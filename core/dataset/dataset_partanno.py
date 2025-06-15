import os
import os.path as osp

import numpy as np

import torch

import easydict

import utils

from . import DATASET
from .dataset_base import BaseDataset

from .projproc import snapshot_spherical


@DATASET.register
class PartAnno(BaseDataset):
	def __init__(self, *args, **kwds):
		super().__init__()

		kwds = easydict.EasyDict(kwds)
		self.args = kwds

		self.split = self.args.split
		assert self.split in self.args.seqs_split, \
			f"{self.split} not in dataset conf"

		self.selecte_cls_name = self.args.selected_cls_name
		assert self.selecte_cls_name in self.args.cls_names, \
			f"{self.selecte_cls_name} not in dataset conf"

		self.points_root = osp.join(self.args.root, self.selecte_cls_name, "points")
		assert osp.exists(self.points_root), \
			f"{self.root} points not exists"
		self.labels_root = osp.join(self.args.root, self.selecte_cls_name, "points_label")
		assert osp.exists(self.labels_root), \
			f"{self.root} labels not exists"
		
		assert sum(self.args.seqs_split.values()) <= 1.0, \
			f"sum of seqs_split should be <= 1.0"
		
		self.file_list = sorted(os.listdir(self.root))
		np.random.shuffle(self.file_list)

		# 根据每个 split 的比例，划分数据集
		self.split_files = {}
		start = 0
		for split, ratio in self.args.seqs_split.items():
			end = start + int(len(self.file_list) * ratio)
			self.split_files[split] = self.file_list[start:end]
			start = end
			if end >= len(self.file_list):
				break
		
		self.files = self.split_files[self.split]
		for i in range(len(self.files)):
			self.files[i] = osp.join(self.root, self.files)

	def __getitem__(self, index):
		# return super().__getitem__(index)
		points = np.loadtxt(osp.join(PARTNO_ROOT, synset, "points", f"{sample}.pts"), delimiter=" ")
		labels = np.loadtxt(osp.join(PARTNO_ROOT, synset, "points_label", f"{sample}.seg"), delimiter=" ").astype(np.int32)
