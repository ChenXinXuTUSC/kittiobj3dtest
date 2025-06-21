import os
import os.path as osp

import numpy as np

import torch

import easydict

import utils

from . import DATASET
from .dataset_base import BaseDataset

from utils.projproc import snapshot_voxelized


@DATASET.register
class PartAnno(BaseDataset):
	def __init__(self, *args, **kwds):
		super().__init__()

		kwds = easydict.EasyDict(kwds)
		self.args = kwds

		self.root = kwds.root
		assert osp.exists(self.root), \
			f"{self.root} not exists"

		cls2syn = {v: k for k, v in kwds.syn2cls.items()}

		self.split = kwds.split
		assert self.split in kwds.seqs_split, \
			f"{self.split} not in dataset conf"

		self.selected_cls_name = kwds.selected_cls_name
		assert self.selected_cls_name in kwds.cls_names, \
			f"{self.selected_cls_name} not in dataset conf"
		self.selecte_syn_name = cls2syn[self.selected_cls_name]

		self.points_root = osp.join(self.root, self.selecte_syn_name, "points")
		assert osp.exists(self.points_root), \
			f"{self.root} points not exists"
		self.labels_root = osp.join(self.root, self.selecte_syn_name, "points_label")
		assert osp.exists(self.labels_root), \
			f"{self.root} labels not exists"
		
		assert len(os.listdir(self.points_root)) == len(os.listdir(self.labels_root)), \
			f"{self.selected_cls_name} number of points and labels file mismatched"

		assert sum(self.args.seqs_split.values()) <= 1.0, \
			f"sum of seqs_split ratio should be <= 1.0"
		
		self.all_samples = [osp.splitext(x)[0] for x in  sorted(os.listdir(self.points_root))]
		np.random.shuffle(self.all_samples)

		# 根据每个 split 的比例，划分数据集
		self.split_samples = {}
		start = 0
		for split, ratio in self.args.seqs_split.items():
			end = start + int(len(self.all_samples) * ratio)
			self.split_samples[split] = self.all_samples[start:end]
			start = end
			if end >= len(self.all_samples):
				break
		
		self.samples = self.split_samples[self.split]
		self.files = []
		for sample in self.samples:
			self.files.append((
				osp.join(self.points_root, sample + ".pts"),
				osp.join(self.labels_root, sample + ".seg")
			))
		

		# 其他属性获取，体素投影分辨率尺寸，体素大小，各个投影轴位置
		self.voxel_size = kwds.voxel_size
		self.proj_img_h = kwds.proj_img_h
		self.proj_img_w = kwds.proj_img_w
		self.proj_axis_list = kwds.proj_axis_list

	def __getitem__(self, index):
		# return super().__getitem__(index)
		points = np.loadtxt(self.files[index][0], delimiter=" ").astype(np.float32)
		labels = np.loadtxt(self.files[index][1], delimiter=" ").astype(np.int32)

		data, gdth, rmap = list(zip(*[
			(data, gdth) for data, gdth, rmap in [snapshot_voxelized(
				points[:, :3], labels,
				self.voxel_size, (self.proj_img_h, self.proj_img_w),
				proj_axis, (0, 0, 0) # partanno 是小数据集，直接对模型中央投影即可
			) for proj_axis in self.proj_axis_list]
		]))

		return (
			np.array(data, dtype=np.float32),
			np.array(gdth, dtype=np.int32),
			np.array(rmap, dtype=np.int32)
		)
	
	def __len__(self):
		return len(self.files)

	def __batch_padding__(self, batch: list):
		# all are batched data, not single sample
		data, gdth, rmap = zip(*batch)
		data = torch.tensor(np.array(data), dtype=torch.float32)
		gdth = torch.tensor(np.array(gdth), dtype=torch.long)
		# 由于投影区域的不同，每幅投影图的逆映射涉及到的原始点云数量不同，需要统一填充到一致长度
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
