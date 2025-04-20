import os
import os.path as osp
import easydict

import torch
import torch.utils.data.dataset

import numpy as np

import utils

from .projproc import snapshot_spherical

from . import DATASET

@DATASET.register
class KITTISemantic(torch.utils.data.dataset.Dataset):
    def __init__(self, *args, **kwds):
        super().__init__()
        kwds = easydict.EasyDict(kwds)
        self.args = kwds

        self.split = self.args.split
        assert self.split == "train" or self.split == "valid", f"invalid split {self.split}"

        self.root = osp.join(self.args.root, 'sequences')
        if self.split == "train":
            seq_list = self.args.train_seq_list
        if self.split == "valid":
            seq_list = self.args.valid_seq_list

        self.files = []

        for seq_idx in seq_list:
            seq_dir = osp.join(self.root, seq_idx)
            data_dir = osp.join(seq_dir, 'velodyne')
            gdth_dir = osp.join(seq_dir, 'labels')
            for item in os.listdir(data_dir):
                fname = osp.splitext(item)[0]

                if osp.exists(osp.join(data_dir, fname+".bin")) and osp.exists(osp.join(gdth_dir, fname+".label")):
                    self.files.append((
                        osp.join(data_dir, fname + ".bin"),
                        osp.join(gdth_dir, fname + ".label")
                    ))
        
        self.cls2idx = {cls: idx for (idx, cls) in enumerate(self.args.cls_names)}
        self.idx2cls = {idx: cls for (idx, cls) in enumerate(self.args.cls_names)}
        self.ldx2idx = {ldx: self.cls2idx[cls] for (cls, ldx) in self.args.cls_idx.items()}
        self.pallete = {idx: clr for (idx, clr) in enumerate(self.args.pallete)}

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        points = self.__read_points(self.files[index][0])
        labels = self.__read_labels(self.files[index][1])
        # transform into contiguous index from 0 to n-1
        for k, v in self.ldx2idx.items():
            labels[labels == k] = v

        proj_img_h = self.args.proj_img_h
        proj_img_w = self.args.proj_img_w

        fmap, gdth, rmap = snapshot_spherical(
            points, labels,
            img_h=proj_img_h,
            img_w=proj_img_w
        )
        # do smooth on range and intensity channel
        fmap[0] = utils.image_fill2(fmap[0], 0, 1e-4, 4)
        fmap[1] = utils.image_fill2(fmap[1], 0, 1e-4, 4)
        fmap[2] = utils.image_fill2(fmap[2], 0, 1e-4, 4)
        fmap[3] = utils.image_fill2(fmap[3], 0, 1e-4, 4)
        fmap[4] = utils.image_fill2(fmap[4], 0, 1e-4, 4)
        gdth = utils.image_fill2(gdth, 0, 1e-4, 4)
        fmap = utils.normalized_fmap(fmap, [0, 1, 2, 3, 4])

        return fmap, gdth.astype(np.int64)
    
    def __read_points(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    

    def __read_labels(self, path):
        labels = np.fromfile(path, dtype=np.uint32).reshape(-1)
        upper_half = labels >> 16      # get upper half for instances
        lower_half = labels & 0xFFFF   # get lower half for semantics
        return lower_half.astype(np.int32)

