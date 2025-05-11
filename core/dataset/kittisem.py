import os
import os.path as osp
import easydict

import numpy as np

import torch

import utils

from . import DATASET
from .base import BaseDataset
from .projproc import snapshot_spherical

@DATASET.register
class KITTISemantic(BaseDataset):
	def __init__(self, *args, **kwds):
		super().__init__()
		kwds = easydict.EasyDict(kwds)
		self.args = kwds

		self.split = self.args.split
		assert self.split in self.args.seqs_split, f"{self.split} not in dataset conf"

		self.root = osp.join(self.args.root, 'sequences')
		seq_list = self.args.seqs_split[self.split]

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
		# cls 原始类名 class
		# idx 连续索引 index
		# lbl 离散索引 label
		self.cls2idx = {cls: idx for (idx, cls) in enumerate(self.args.cls_names)}
		self.idx2cls = {idx: cls for (idx, cls) in enumerate(self.args.cls_names)}
		self.ldx2idx = {ldx: self.cls2idx[cls] for (cls, ldx) in self.args.cls_lbl.items()}
		self.pallete = {idx: clr for (idx, clr) in enumerate(self.args.pallete)}

		# 更新批次样本处理函数，填充每个批次的内存布局
		self.collate_fn_registry["train"] = self.__batch_padding__
		self.collate_fn_registry["valid"] = self.__batch_padding__
		self.collate_fn_registry["testt"] = self.__batch_padding__

	def __len__(self):
		return len(self.files)
	
	def __getitem__(self, index):
		points = self.__read_points__(self.files[index][0])
		labels = self.__read_labels__(self.files[index][1])
		# transform original discrete label
		# into contiguous index (from 0 to n-1)
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

		return fmap, gdth, rmap
	
	def __read_points__(self, path):
		return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
	
	def __read_labels__(self, path):
		labels = np.fromfile(path, dtype=np.uint32).reshape(-1)
		upper_half = labels >> 16	  # get upper half for instances
		lower_half = labels & 0xFFFF   # get lower half for semantics
		return lower_half.astype(np.int32)

	def __batch_padding__(self, batch: list):
		# all are batched data, not single sample
		data, gdth, rmap = zip(*batch)
		data = torch.tensor(np.array(data), dtype=torch.float32)
		gdth = torch.tensor(np.array(gdth), dtype=torch.long)
		# padding
		max_rmap_len = max([len(x) for x in rmap])
		rmap = np.array(
			[
				(
					len(x),
					np.pad(
						x,
						((0, max_rmap_len - len(x)), (0, 0)),
						mode="constant", constant_values=0
					)
				) for x in rmap
			],
			dtype=np.dtype([
				('valid', np.int32),
				('ptpix', np.float32, (max_rmap_len, rmap[0].shape[1]))  # 不知道 dataset 传出来的逆投影关系映射一个关系包含多少特征
			])
		)
		return {
			"data": data,
			"gdth": gdth,
			"rmap": rmap
		}
