import os
import os.path as osp
import sys
sys.path.append("..")

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import easydict

from . import MODEL

@MODEL.register
class DeepLabV3(nn.Module):
	def __init__(self, *args, **kwds):
		super().__init__()
		kwds = easydict.EasyDict(kwds)
		self.args = kwds

		in_channels = kwds.in_channels
		out_channels = kwds.out_channels
		
		self.squeeze = None
		if in_channels != 3:
			self.squeeze = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)

		self.deeplab = torch.hub.load("hub/deeplabv3", source="local", model='deeplabv3_resnet50', pretrained=True)
		self.deeplab.classifier[-1] = nn.Conv2d(256, out_channels, kernel_size=(1, 1), stride=(1, 1))
		self.deeplab.aux_classifier[-1] = nn.Conv2d(256, out_channels, kernel_size=(1, 1), stride=(1, 1))
	
	def forward(self, batch: torch.Tensor):
		data = batch["data"]
		
		old_shape = data.shape[2:]

		data = F.interpolate(data, scale_factor=((224-1)/old_shape[0]+1, (224-1)/old_shape[1]+1), mode="bilinear", align_corners=True)
		if self.squeeze is not None:
			data = self.squeeze(data)
		
		pred = self.deeplab(data)
		out = F.interpolate(pred["out"], size=old_shape, mode="bilinear", align_corners=True)
		aux = F.interpolate(pred["aux"], size=old_shape, mode="bilinear", align_corners=True)

		return {"out": out, "aux": aux}
