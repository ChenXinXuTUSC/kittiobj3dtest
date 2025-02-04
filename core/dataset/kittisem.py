import os
import os.path as osp
import easydict

import torch
import torch.utils.data.dataset

import numpy as np


from . import DATASET

@DATASET.register
class KITTISemantic(torch.utils.data.dataset.Dataset):
    def __init__(self, *args, **kwds):
        super().__init__()
        kwds = easydict.EasyDict(kwds)
        self.args = kwds

        self.root = osp.join(self.args.root, 'sequences') 
        seq_list = self.args.seq_list

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
        

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        points = self.__read_kitti_bin(self.files[index][0])
        labels = self.__read_kitti_label(self.files[index][1])

        proj_img_h = self.args.proj_img_h
        proj_img_w = self.args.proj_img_w

        return points, labels
    
    def __read_kitti_bin(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    

    def __read_kitti_label(self, path):
        labels = np.fromfile(path, dtype=np.uint32).reshape(-1)
        upper_half = labels >> 16      # get upper half for instances
        lower_half = labels & 0xFFFF   # get lower half for semantics
        return lower_half

