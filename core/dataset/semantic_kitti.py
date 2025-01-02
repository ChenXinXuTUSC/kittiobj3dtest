import os
import os.path as osp

import torch
import torch.utils.data.dataset
import numpy as np

import struct
import easydict

import utils

from . import register_dataset

@register_dataset
class KITTISem (torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, split: str, conf):
        super().__init__()
