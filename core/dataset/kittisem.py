import os
import os.path as osp
import easydict

import torch
import torch.utils.data.dataset


from . import DATASET

@DATASET.register
class KITTISemantic (torch.utils.data.dataset.Dataset):
    def __init__(self, *args, **kwds):
        super().__init__()
        kwds = easydict.EasyDict(kwds)
        self.args = kwds
