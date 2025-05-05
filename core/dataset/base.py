import os
import os.path as osp

import numpy as np

import torch
import torch.utils.data.dataset

from collections import defaultdict

def __return_none__():
	return None

class BaseDataset(torch.utils.data.dataset.Dataset):
	def __init__(self):
		super().__init__()

		self.collate_fn_registry = defaultdict(__return_none__)
	
	def collate_fn(self, func_name: str):
		return self.collate_fn_registry[func_name]
