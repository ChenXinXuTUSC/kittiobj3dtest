import torch
import torch.nn as nn

import sys
sys.path.append("..")

import easydict

from .embd import Embedding
from .encd import CascadedEncoder

class ConvDn(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		# 全尺寸卷积 + 池化 + ReLU
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.norm = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, x: torch.Tensor):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.pool(x)
		x = self.norm(x)
		x = self.relu(x)
		return x

class ConvUp(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		# 通过 ConvTranspose2d 完成 2x 倍率上采样
		self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
		self.norm = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, x: torch.Tensor):
		x = self.up(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.norm(x)
		x = self.relu(x)
		return x


from . import MODEL
@MODEL.register
class TransUNetMini(nn.Module):
	def __init__(self, *args, **kwds):
		super().__init__()
		# 1. 编码器 - Transformer
		# 2. 解码器 - 级联反卷积上采样

		kwds = easydict.EasyDict(kwds)
		self.args = kwds

		self.dn_channels = kwds.dn_channels # channels during downsample
		self.up_channels = kwds.up_channels # channels during upsample
		assert len(self.dn_channels) == len(self.up_channels), \
			"UNet structure requires same number of up and down sample"


		# 注意图块嵌入器的图像大小是降采样最后输出的尺寸，不是图像原始尺寸
		self.embeder = Embedding(
			kwds.embd_image_shape,
			kwds.embd_patch_shape,
			kwds.in_channels,
			kwds.out_channels
		)
		# 级联 Transformer 编码器
		self.encoder = CascadedEncoder(
			num_casd=kwds.num_encds,
			num_heads=kwds.num_heads,
			hidn_size=kwds.out_channels
		)
		# 语义分割头
		self.seg_head = None # todo: finish
	
	def forward(self, x: torch.Tensor):
		# 进入的样本批次形状为 [B, N, H, W, C]，压缩为 [B*N, H, W, C]
		# 一个样本具有多个视角投影图，现在并行卷积，不区分每个特征图具体来自哪个样本
		x = x.view(-1, *x.shape[2:])
