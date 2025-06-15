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
		assert self.dn_channels[-1] == self.up_channels[0], \
			"UNet structure requires same feature channels between downsample and upsample end point"

		self.dn_channels = [kwds.in_channels] + self.dn_channels
		self.up_channels = self.up_channels + [kwds.out_channels]

		self.dn_convs = [ConvDn(self.dn_channels[i], self.dn_channels[i+1]) for i in range(len(self.dn_channels)-1)]
		self.up_convs = [ConvUp(self.up_channels[i], self.up_channels[i+1]) for i in range(len(self.up_channels)-1)]

		# 注意图块嵌入器的图像大小是降采样最后输出的尺寸，不是图像原始尺寸
		# 将一幅特征图转换为一个 patch-token 序列
		self.embeder = Embedding(
			kwds.embd_image_shape,
			(1, 1),
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
		# 一个样本具有多个视角投影图，现在并行卷积，不区分每个特征图具体来自哪个样本
		# 进入的样本批次形状为 [B, N, H, W, C]，压缩为 [B*N, H, W, C]
		B, I, C, H, W = x.size()
		x = x.view(B * I, C, H, W)

		# 两层下采样
		dn_out_list = []
		for dn_conv in self.dn_convs:
			dn_out_list.append(dn_conv(x))
		_, _, CONV_DN_H, CONV_DN_W = x.size()

		# Transformer 编码器
		x = self.encoder(x)

		# 编码器输出重新转换为上采样所需维度
		_, S, E = x.size() # [忽略样本批次，序列长度，嵌入维度]
		x = x.view(B, I, S, E) # [样本批次，每个样本视图数量，每个视图的序列长度，嵌入维度]
		# 交换最后两个维度
		x = x.permute(0, 1, 3, 2)
		x = x.view(B, I, E, CONV_DN_H, CONV_DN_W)

		# 两层上采样
		up_out_list = []
		for up_conv in self.up_convs:
			up_out_list.append(up_conv(x))
		
		# 最终输出


