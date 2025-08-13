import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import sys
sys.path.append("..")

import easydict

from .embd import Embedding
from .encd import CascadedEncoder

class ConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels), # 插入 BatchNorm
			nn.ReLU(inplace=True)         # 插入 ReLU
		)
	
	def forward(self, x: torch.Tensor):
		return self.conv(x)

class ConvDn(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		# 改进：第一个卷积层后插入归一化和激活
		self.conv1 = ConvBlock(in_channels, out_channels)
		# 改进：第二个卷积层后插入归一化和激活
		self.conv2 = ConvBlock(out_channels, out_channels)
		# 池化层通常在卷积和激活之后进行下采样
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
	def forward(self, x: torch.Tensor):
		x = self.conv1(x) # conv -> norm -> relu
		x = self.conv2(x) # conv -> norm -> relu
		x = self.pool(x)        # 下采样
		return x

class ConvUp(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		# 通过 ConvTranspose2d 完成 2x 倍率上采样
		self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 改进：插入归一化和激活
		self.conv1 = ConvBlock(out_channels, out_channels)
        # 改进：插入归一化和激活
		self.conv2 = ConvBlock(out_channels, out_channels)
	
	def forward(self, x: torch.Tensor):
		x = self.up(x)
		x = self.conv1(x)
		x = self.conv2(x)
		return x


from . import MODEL
@MODEL.register
class TransUNetMini(nn.Module):
	def __init__(self, *args, **kwds):
		super().__init__()
		# 1. 编码器 - Transformer
		# 2. 解码器 - 级联反卷积上采样生成

		kwds = easydict.EasyDict(kwds)
		self.args = kwds

		# self.dn_channels = kwds.dn_channels # channels during downsample
		# self.up_channels = kwds.up_channels # channels during upsample
		assert len(kwds.dn_channels) == len(kwds.up_channels), \
			"UNet structure requires same number of up and down sample"
		
		# for i, (dn, up) in enumerate(zip(self.dn_channels, self.up_channels)):
		# 	self.up_channels[i] = dn + up # UNet skip connection concate
		# self.dn_channels = [kwds.in_channels] + self.dn_channels
		# self.up_channels = [kwds.attn_hidn_size] + self.up_channels

		# 多级卷积请注册为 ModuleList ，否则 to(device) 不会检测到非 nn.Module 属性的参数
		self.dn_convs = nn.ModuleList(
			[ConvBlock(kwds.in_channels, kwds.dn_channels[0], kernel_size=1, stride=1, padding=0)] + \
			[ConvDn(kwds.dn_channels[i], kwds.dn_channels[i+1]) for i in range(len(kwds.dn_channels)-1)]
		)
		self.up_convs = nn.ModuleList(
			[ConvBlock(kwds.attn_hidn_size, kwds.up_channels[0], kernel_size=1, stride=1, padding=0)] + \
			[ConvUp(kwds.up_channels[i] + kwds.dn_channels[-(i+1)], kwds.up_channels[i+1]) for i in range(len(kwds.up_channels)-1)]
		)

		# 注意图块嵌入器的图像大小是降采样最后输出的尺寸，不是图像原始尺寸
		# 将一幅特征图转换为一个 patch-token 序列
		self.embeder = Embedding(
			image_shape=[
				kwds.proj_img_h // (2**(len(kwds.dn_channels)-1)),
				kwds.proj_img_w // (2**(len(kwds.dn_channels)-1))
			],
			patch_shape=(1, 1),
			in_channels=kwds.dn_channels[-1],
			out_channels=kwds.attn_hidn_size
		)
		# 级联 Transformer 编码器
		self.encoder = CascadedEncoder(
			num_casd=kwds.num_encds,
			num_heads=kwds.num_heads,
			hidn_size=kwds.attn_hidn_size,
		)
		# 分类输出头
		self.seg_head = nn.Conv2d(
			in_channels=kwds.up_channels[-1] + kwds.dn_channels[0],
			out_channels=kwds.out_channels,
			kernel_size=1,
			stride=1,
			padding=0
		)

		# 其他在后续需要使用到的属性
		self.in_channels = kwds.in_channels
		self.out_channels = kwds.out_channels
	
	def forward(self, batch: dict):
		# batch 是一个数据集输出的自定义的字典，存储了所有相关的训练所需的数据，不只是训练集的输入
		x = batch["data"]
		# 一个样本具有多个视角投影图，现在并行卷积，不区分每个特征图具体来自哪个样本
		# 进入的样本批次形状为 [B, N, H, W, C]，压缩为 [B*N, H, W, C]
		B, I, C, H, W = x.size()
		x = x.view(B * I, C, H, W)

		# 下采样链路
		dn_out_list = []
		for dn_conv in self.dn_convs:
			x = dn_conv(x)
			dn_out_list.append(x)
		# 记录送入编码器之前最后的 patch shape
		_, _, CONV_DN_H, CONV_DN_W = x.size() 

		# Vision Transformer 编码器
		x = self.embeder(x) # 从图块转变为序列数据
		x = self.encoder(x)

		# 编码器输出重新转换为上采样所需维度
		_, S, E = x.size() # [忽略样本批次，序列长度，嵌入维度]
		x = x.view(B * I, S, E) # [样本批次，每个样本视图数量，每个视图的序列长度，嵌入维度]
		# 交换最后两个维度
		x = x.permute(0, 2, 1)
		x = x.view(B * I, E, CONV_DN_H, CONV_DN_W)

		# 上采样链路 和 下采样链路对应每层进行拼接
		for up_conv in self.up_convs:
			x = up_conv(x)
			x = torch.concat([x, dn_out_list.pop()], dim=1)
		
		# 最终输出语义分割 logit
		x = self.seg_head(x)
		# 还原输入，便于后续指标计算
		x = x.view(B, I, self.out_channels, H, W)
		return x
