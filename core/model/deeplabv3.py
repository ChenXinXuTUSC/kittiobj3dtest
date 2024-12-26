import os
import os.path as osp
import sys
sys.path.append("..")

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model

@register_model
class DeepLabV3(nn.Module):
    def __init__(self, num_cls: int, stretch_shape: tuple, in_channels: int):
        super(DeepLabV3, self).__init__()
        
        self.stretch = nn.Upsample(size=stretch_shape, mode='bilinear', align_corners=True)
        self.squeeze = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)

        self.deeplab = torch.hub.load('hub/deeplabv3', source="local", model='deeplabv3_resnet50', pretrained=True)
        self.deeplab.classifier[-1] = nn.Conv2d(256, num_cls, kernel_size=(1, 1), stride=(1, 1))
        self.deeplab.aux_classifier[-1] = nn.Conv2d(256, num_cls, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x: torch.Tensor):
        original_shape = x.shape[2:]
        x = self.stretch(x)
        x = self.squeeze(x)
        x = self.deeplab(x)
        return (
            F.upsample(x["out"], size=original_shape, mode="bilinear", align_corners=True),
            F.upsample(x["aux"], size=original_shape, mode="bilinear", align_corners=True)
        )
