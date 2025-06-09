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

		self.split = self.args.split
		assert self.split in self.args.seqs_split, f"{self.split} not in dataset conf"

		self.root = osp.join(self.args.root, 'sequences')

		self.data_dir = osp.join('partanno')
		self.file_list = sorted(os.listdir(self.data_dir))
		np.random.shuffle(self.file_list)

		train_size = int(len(self.file_list) * self.args.split)
		self.train_files = self.file_list[:train_size]
		self.val_files = self.file_list[train_size:]


