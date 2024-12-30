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
    def __init__(self, in_channels: int, out_channels: int):
        super(DeepLabV3, self).__init__()
        
        self.squeeze = None
        if in_channels != 3:
            self.squeeze = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)

        self.deeplab = torch.hub.load("hub/deeplabv3", source="local", model='deeplabv3_resnet50', pretrained=True)
        self.deeplab.classifier[-1] = nn.Conv2d(256, out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.deeplab.aux_classifier[-1] = nn.Conv2d(256, out_channels, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x: torch.Tensor):
        old_shape = x.shape[2:]

        x = F.interpolate(x, scale_factor=((224-1)/old_shape[0]+1, (224-1)/old_shape[1]+1), mode="bilinear", align_corners=True)
        if self.squeeze is not None:
            x = self.squeeze(x)
        
        x = self.deeplab(x)
        return ( # keep the same if x not strethced
            F.interpolate(x["out"], size=old_shape, mode="bilinear", align_corners=True),
            F.interpolate(x["aux"], size=old_shape, mode="bilinear", align_corners=True)
        )
