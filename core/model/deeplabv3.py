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
        super().__init__()
        
        self.squeeze = None
        if in_channels != 3:
            self.squeeze = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)

        self.deeplab = torch.hub.load("hub/deeplabv3", source="local", model='deeplabv3_resnet50', pretrained=True)
        self.deeplab.classifier[-1] = nn.Conv2d(256, out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.deeplab.aux_classifier[-1] = nn.Conv2d(256, out_channels, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, data: torch.Tensor, gdth: torch.Tensor, criterion: nn.Module):
        old_shape = data.shape[2:]

        data = F.interpolate(data, scale_factor=((224-1)/old_shape[0]+1, (224-1)/old_shape[1]+1), mode="bilinear", align_corners=True)
        if self.squeeze is not None:
            data = self.squeeze(data)
        
        data = self.deeplab(data)
        out = F.interpolate(data["out"], size=old_shape, mode="bilinear", align_corners=True)
        aux = F.interpolate(data["aux"], size=old_shape, mode="bilinear", align_corners=True)
        data = (out, aux)
        loss = self.loss_fn(data, gdth, criterion)
        pred = self.predict(data)
        
        return loss, pred

    def loss_fn(self, pred, gdth: torch.Tensor, criterion: nn.Module):
        out = pred[0] # main prediction
        aux = pred[1] # auxiliary prediction

        loss_out = criterion(out, gdth)
        loss_aux = criterion(aux, gdth)
        return loss_out + 0.5 * loss_aux

    def predict(self, model_out):
        return model_out[0] # main output from deeplabv3
