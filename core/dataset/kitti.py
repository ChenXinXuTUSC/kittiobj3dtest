import os
import os.path as osp

import torch
import torch.utils.data.dataset
import numpy as np

import struct
import easydict
import random

import utils


class KITTISpherical(torch.utils.data.dataset.Dataset):
    TRAIN_SET_RATIO = 0.7
    TEST_SET_RATIO = 0.3
    IMG_W = 512
    IMG_H = 64

    kitti_obj3d_det_cls_names = [
        "DontCare",
        "Car",
        "Cyclist",
        "Misc",
        "Pedestrian",
        "Person_sitting",
        "Tram",
        "Truck",
        "Van"
    ]

    label_field_name_list = [
        "object_type",
        "truncation",
        "occlusion",
        "alpha",
        "left", "top", "right", "bottom",
        "height", "width", "length",
        "x", "y", "z",
        "rotation_y"
    ]

    def __init__(self, kitti_root: str, split: str):
        super().__init__()

        assert split == "train" or split == "test", "invalid split"
        
        self.kitti_root = kitti_root

        num_frames = len([
            x for x in os.listdir(osp.join(kitti_root, "training", "velodyne")) if x.endswith(".bin")
        ])
        self.frame_index = list(range(num_frames))
        random.shuffle(self.frame_index)

        if split == "train":
            self.frame_index = self.frame_index[:int(num_frames * self.TRAIN_SET_RATIO)]
        if split == "test":
            self.frame_index = self.frame_index[:int(num_frames * self.TEST_SET_RATIO)]

        # class name and label index
        self.cls2ldx = {v: i for (i, v) in enumerate(self.kitti_obj3d_det_cls_names)}
        self.ldx2cls = {i: v for (v, i) in self.cls2ldx.items()}

        self.num_cls = len(self.kitti_obj3d_det_cls_names)


    def __len__(self) -> int:
        return len(self.frame_index)


    def __getitem__(self, index):
        fmap, gdth, rmap = self.__spherical_project(index)
        fmap = utils.normalized_fmap(fmap, [3, 4])
        # do smooth on range and intensity channel
        fmap[3] = utils.fill_blank(fmap[3], 1e-4, 4)
        fmap[4] = utils.fill_blank(fmap[4], 1e-4, 4)
        gdth = utils.fill_blank(gdth, 1e-4, 4)

        return fmap, gdth
    

    def __spherical_project(self, index: int):
        '''
        - kitti_obj3d_root: path to dataset dir containing training and testing dir
        - index: frame data index
        '''
        assert osp.exists(self.kitti_root), "dataset root not exist"
        assert index >= 0 and index <= 7480,\
            "index out of train samples in kitti object 3d dataset"

        index = f"{index:06d}" # pad with 0

        # step 1 - read lidar points
        bin_path = osp.join(
            self.kitti_root, "training", "velodyne", f"{index}.bin"
        )
        bin_size = osp.getsize(bin_path)
        assert bin_size % 16 == 0, "invalid binary structure for kitti bin"

        lidar = []
        with open(bin_path, "rb") as f:
            while True:
                byte_data = f.read(4)
                if len(byte_data) < 4:
                    break
                lidar.append(struct.unpack('f', byte_data))
        
        lidar = np.array(lidar).reshape((-1, 4))

        points = lidar[:, :3] # xyz coordinates
        reflec = lidar[:,  3] # reflective intensity
        ptscls = np.zeros(len(points)) # points classes




        # step 2 - read bounding box and calibration info
        def is_float(item):
            try:
                item = float(item)
            except Exception as e:
                return False
            return True

        label_list = None
        with open(
            osp.join(
                self.kitti_root,
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
                {field:value for (field, value) in zip(self.label_field_name_list, item)}
            ) for item in item_list
        ]

        calib = None
        with open(
            osp.join(
                self.kitti_root,
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




        # step 3 - label mask
        for label in label_list:
            if label.object_type == "DontCare":
                continue
            
            R_cam = utils.R_mat(label.rotation_y, [0, 1, 0])
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

            points_msk = R_vly.T @ (points.T - T_vly)
            mask = np.all((points_msk.T >= lb_vly.T) & (points_msk.T <= rt_vly.T), axis=1)

            ptscls[mask] = self.cls2ldx[label.object_type]



        # step 4 - spherical projection
        # only project the front 90 degree view range

        mask = points[:, 1] / np.linalg.norm(points[:, :2], axis=1)
        # angle between -45 and +45
        mask = (points[:, 0] > 0) & (mask > np.sin(-np.pi / 4)) & (mask < np.sin(+np.pi / 4))

        front_indics = np.array(list(range(len(points))))[mask]

        x = np.arcsin(points[mask, 1] / np.linalg.norm(points[mask, :2], axis=1))
        y = np.arcsin(points[mask, 2] / np.linalg.norm(points[  mask  ], axis=1))

        field_x_range, res_x = np.pi / 2               , self.IMG_W
        filed_y_range, res_y = y.max() - y.min() + 1e-2, self.IMG_H

        delta_x = field_x_range / res_x
        delta_y = filed_y_range / res_y

        x = (x // delta_x).astype(np.int32)
        y = (y // delta_y).astype(np.int32)

        x = x - x.min()
        y = y - y.min()

        gdth = np.zeros((self.IMG_H, self.IMG_W))
        fmap = np.zeros((5, self.IMG_H, self.IMG_W))
        rmap = [[[] for _ in range(self.IMG_W)] for _ in range(self.IMG_H)]

        # feature vector [x, y, z, r, i], shape [C, H, W]
        # as there might be multiple points projected into one grid
        # I choose not to use np.add.at to parallel projection
        for i in range(len(front_indics)):
            rmap[y[i]][x[i]].append(front_indics[i])
            # select the point with nearest range as the feature pixel
            dist = np.linalg.norm(points[front_indics[i]])
            if  fmap[3, y[i], x[i]] != 0 and dist < fmap[3][y[i], x[i]]:
                fmap[:, y[i], x[i]] = np.array([*points[front_indics[i]], dist, reflec[front_indics[i]]])
                gdth[y[i], x[i]] = ptscls[front_indics[i]]
            else:
                fmap[:, y[i], x[i]] = np.array([*points[front_indics[i]], dist, reflec[front_indics[i]]])
                gdth[y[i], x[i]] = ptscls[front_indics[i]]
        
        return fmap, gdth, rmap
