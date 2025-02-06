import os
import os.path as osp

import torch
import torch.utils.data.dataset
import numpy as np
import open3d as o3d

import struct
import easydict

import utils

from . import DATASET


@DATASET.register
class KITTIObj3d(torch.utils.data.dataset.Dataset):
    def __init__(self, *args, **kwds):
        super().__init__()
        kwds = easydict.EasyDict(kwds)
        self.args = kwds

        self.root = self.args.root
        self.split = self.args.split
        # kitti doesn't provide ground truth for test set
        assert osp.exists(self.root), f"dataset root {self.root} not exist"
        assert self.split == "train" or self.split == "valid", "invalid split"

        num_frames = len([
            x for x in os.listdir(osp.join(self.root, "training", "velodyne")) if x.endswith(".bin")
        ])
        self.frame_index = list(range(num_frames))

        if self.split == "train":
            self.frame_index = self.frame_index[:int(num_frames * self.args.train_set_ratio)]
        if self.split == "valid":
            self.frame_index = self.frame_index[int(num_frames * self.args.train_set_ratio):]

        # class name and label index
        self.cls2ldx = {v: i for (i, v) in enumerate(self.args.kitti_obj3d_det_cls_names)}
        self.ldx2cls = {i: v for (v, i) in self.cls2ldx.items()}

        self.num_cls = len(self.args.kitti_obj3d_det_cls_names)


    def __len__(self) -> int:
        return len(self.frame_index)


    def __getitem__(self, index):
        points, labels = self.__make_labels(index)

        fmap, gdth, rmap = self.spherical_project(points, labels, fov_h=np.pi / 2.0)
        # do smooth on range and intensity channel
        fmap[0] = utils.fill_blank(fmap[0], 0, 1e-4, 4)
        fmap[1] = utils.fill_blank(fmap[1], 0, 1e-4, 4)
        fmap[2] = utils.fill_blank(fmap[2], 0, 1e-4, 4)
        fmap[3] = utils.fill_blank(fmap[3], 0, 1e-4, 4)
        fmap[4] = utils.fill_blank(fmap[4], 0, 1e-4, 4)
        gdth = utils.fill_blank(gdth, 0, 1e-4, 4)
        fmap = utils.normalized_fmap(fmap, [0, 1, 2, 3, 4])

        return fmap, gdth
    
    def __read_points(self, index: int):
        index = f"{index:06d}" # pad with 0

        bin_path = osp.join(
            self.root, "training", "velodyne", f"{index}.bin"
        )
        bin_size = osp.getsize(bin_path)
        # 4 components and 4 bytes for each component
        assert bin_size % 16 == 0, "invalid binary structure for kitti bin"

        lidar = []
        with open(bin_path, "rb") as f:
            while True:
                byte_data = f.read(4)
                if len(byte_data) < 4:
                    break
                lidar.append(struct.unpack('f', byte_data))
        
        points = np.array(lidar).reshape((-1, 4))

        return points
    
    def __make_labels(self, index: int):
        points = self.__read_points(index)
        coords = points[:, :3]
        labels = np.zeros(len(points))

        index = f"{index:06d}" # pad with 0

        def is_float(item):
            try:
                item = float(item)
            except Exception as e:
                return False
            return True

        label_list = None
        with open(
            osp.join(
                self.root,
                "training",
                "label_2",
                f"{index}.txt"),
                "r"
        ) as f:
            item_list = [
                tuple(
                    [float(item) if is_float(item) else item for item in line.strip().split()]
                ) for line in f.readlines()
            ]
        
        label_list = [
            easydict.EasyDict(
                {field:value for (field, value) in zip(self.args.label_field_name_list, item)}
            ) for item in item_list
        ]

        calib = None
        with open(
            osp.join(
                self.root,
                "training",
                "calib",
                f"{index}.txt",
            ), "r"
        ) as f:
            item_list = [
                line.strip().split() for line in f.readlines() if len(line.strip()) > 0
            ]
        calib = easydict.EasyDict(
            {item[0][:-1]:[float(x) for x in item[1:]] for item in item_list}
        )


        # step 2 - label mask
        for label in label_list:
            if label.object_type == "DontCare":
                continue
            
            R_cam = utils.R_mat([0, 1, 0], label.rotation_y)
            T_cam = utils.T_vec([label.x, label.y, label.z])
            
            # transform from camera coordinate to velodyn coordinate
            vly2cam = np.array(calib.Tr_velo_to_cam).reshape((3, 4))
            R_vly2cam = vly2cam[:3, :3]
            T_vly2cam = vly2cam[: ,  3].reshape((-1, 1))

            T_vly = R_vly2cam.T @ (T_cam - T_vly2cam)
            R_vly = R_vly2cam.T @ R_cam @ R_vly2cam

            bbox_expand_ratio = 5e-2
            w, l, h, b = label.width/2 + label.width * bbox_expand_ratio,\
                         label.length/2 + label.length * bbox_expand_ratio,\
                         label.height + label.height * bbox_expand_ratio,\
                         1e-2

            lb_vly = np.array([-w, -l, +b]).reshape((-1, 1)) # left bottom
            rt_vly = np.array([+w, +l, +h]).reshape((-1, 1)) # right top

            coords_rt = R_vly.T @ (coords.T - T_vly)
            mask = np.all((coords_rt.T >= lb_vly.T) & (coords_rt.T <= rt_vly.T), axis=1)

            labels[mask] = self.cls2ldx[label.object_type]
        
        return points, labels


    def spherical_project(
        self,
        _points: np.ndarray,
        _labels: np.ndarray,
        center: np.ndarray = np.array([0.0, 0.0, 0.0]),
        uppvec: np.ndarray = np.array([0.0, 0.0, 1.0]),
        toward: np.ndarray = np.array([1.0, 0.0, 0.0]),
        fov_v: float = None, # vertical
        fov_h: float = None, # default to 90 degree
        img_h: int = 64, # vertical / height
        img_w: int = 512 # horizontal / width
    ):
        '''
        Spherical Projection with specified range of view and center

        Params
        -
            - points: np.ndarray, [N, 3] coordinates
            - labels: np.ndarray, [N, 1] point class type int
            - center: np.ndarray, [1, 3] center coord
            - uppvec: np.ndarray, [1, 3] world up vector
            - toward: np.ndarray, [1, 3] camera towards
            - fov_v: float, range of vertical field in degree
            - fov_h: float, range of horizontal field in degree
            - img_h: int, number of pixels on height
            - img_w: int, number of pixels on width

        Return
        -
            - fmap: np.ndarray, [5, C, H, W] shape [x, y, z, i, r] feature map
            - gdth: np.ndarray, [H, W] shape class label map of each pixel
            - rmap: list[list], back projection from pixel to pointcloud index
        '''

        points = np.copy(_points)
        labels = np.copy(_labels)
        coords = points[:, :3]

        assert len(center) == 3, "not in 3d?"
        assert len(uppvec) == 3, "not in 3d?"
        assert len(toward) == 3, "not in 3d?"

        if not isinstance(center, np.ndarray):
            center = np.array(center)
        if not isinstance(uppvec, np.ndarray):
            uppvec = np.array(uppvec)
        if not isinstance(toward, np.ndarray):
            toward = np.array(toward)

        uppvec = uppvec / np.linalg.norm(uppvec)
        toward = toward / np.linalg.norm(toward)
        
        # transform point cloud
        coords -= center

        # align uppvec to z-axis
        R_u2z = utils.A2B_R(uppvec, [0, 0, 1]) # treat z-axis as up axis
        coords = np.matmul(R_u2z, coords.T).T
        # align towrad to x-axis
        toward = np.dot(R_u2z, toward)
        toward[2] = 0.0 # projected to x-y plane
        R_t2x = utils.A2B_R(toward, [1, 0, 0])
        coords = np.matmul(R_t2x, coords.T).T

        radian_xy = np.arctan2(coords[:, 1], coords[:, 0]) # tan(y/x)
        radian_zz = np.arctan2(coords[:, 2], np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)) # tan(r/z)
        
        if fov_v is None:
            fov_v = radian_zz.max() - radian_zz.min()
        if fov_h is None:
            fov_h = radian_xy.max() - radian_xy.min()
        
        delta_h = fov_v / img_h # vertical // height
        delta_w = fov_h / img_w # horizontal // width
        

        img_coord_h = (radian_zz / delta_h).astype(np.int32)
        img_coord_w = (radian_xy / delta_w).astype(np.int32)
        
        mask = np.full((len(coords),), True, dtype=bool)
        # mask = mask & (radian_zz > -fov_v / 2) & (radian_zz < +fov_v / 2)
        mask = mask & (radian_xy > -fov_h / 2) & (radian_xy < +fov_h / 2)
        
        # # 创建一个Open3D的点云对象
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(coords[mask])
        # o3d.io.write_point_cloud("dbug.ply", point_cloud)
        
        img_coord_h -= img_coord_h[mask].min()
        img_coord_w -= img_coord_w[mask].min()

        used = np.array(list(range(len(coords))))[mask]
        # print("retained ratio", mask.astype(np.int32).sum() / len(mask))

        gdth = np.zeros((img_h, img_w))
        fmap = np.zeros((5, img_h, img_w))
        # rmap is for back projection usage
        rmap = [[[] for _ in range(img_w)] for _ in range(img_h)]

        # feature vector [x, y, z, i, r], shape [C, H, W]
        for i in used:
            # `i` is already the original index of points
            rmap[img_coord_h[i]][img_coord_w[i]].append(i)

            # select the point with nearest range as the feature pixel
            dist = np.linalg.norm(coords[i])
            feat = np.array([*points[i], dist])
            # empty pixel or a nearer pixel lead to an update
            if fmap[4, img_coord_h[i], img_coord_w[i]] == 0 or dist < fmap[4, img_coord_h[i], img_coord_w[i]]:
                fmap[:, img_coord_h[i], img_coord_w[i]] = feat
                gdth[img_coord_h[i], img_coord_w[i]] = labels[i]
        
        # fmap[fmap < 0] = 0
        # gdth[gdth < 0] = 0
        return fmap, gdth, rmap
