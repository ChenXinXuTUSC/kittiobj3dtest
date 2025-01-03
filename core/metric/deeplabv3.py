from . import METRIC

import easydict

import torch

@METRIC.register
class DeepLabV3Metric:
    def __init__(self, *args, **kwds):
        kwds = easydict.EasyDict(kwds)
        self.args = kwds

    def __call__(self, pred: torch.Tensor, gdth:torch.Tensor):
        pass
