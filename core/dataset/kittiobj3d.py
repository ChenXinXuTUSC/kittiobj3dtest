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

from .projproc import snapshot_spherical


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
        self.ldx2cls = {i: v for (i, v) in enumerate(self.args.kitti_obj3d_det_cls_names)}

        self.num_cls = len(self.args.kitti_obj3d_det_cls_names)


    def __len__(self) -> int:
        return len(self.frame_index)


    def __getitem__(self, index):
        points, labels = self.__make_labels(index)

        fmap, gdth, rmap = snapshot_spherical(points, labels, fov_h=np.pi / 2.0)
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

        return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    
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

