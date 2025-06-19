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

		data, gdth = list(zip(*[
			(data, gdth) for data, gdth, _ in [snapshot_voxelized(
				points[:, :3], labels,
				self.voxel_size, self.project_res, project_axis
			) for project_axis in self.project_axis_list]
		]))

		return data, gdth
	
	def __len__(self):
		return len(self.files)
