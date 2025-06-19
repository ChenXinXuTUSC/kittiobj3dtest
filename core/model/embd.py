import torch
import torch.nn as nn

import torch.nn.functional as F

class PatchEmbedding(nn.Module):
	def __init__(self,
		patch_shape: tuple,
		in_channels: int,
		out_channels: int
	):
		super().__init__()
		assert isinstance(patch_shape, tuple) \
			and len(patch_shape) == 2, f"wrong shape type {patch_shape}"

		self.conv_proj = nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=patch_shape,
			stride=patch_shape
		)
	
	def forward(self, x: torch.Tensor):
		# [B, C, H, W] x
		# [B, E, H, W] projected
		# [B, E, S] S for number of patch embeddings
		# [B, S, E] E for number of feature channel of one embedding
		x = self.conv_proj(x)
		x = x.flatten(2).transpose(1, 2)
		return x

class Embedding(nn.Module):
	def __init__(self,
		image_shape: tuple,
		patch_shape: tuple,
		in_channels: int,
		out_channels: int
	):
		super().__init__()

		assert isinstance(image_shape, tuple) or isinstance(image_shape, list) and len(image_shape) == 2, \
			f"wrong shape type {type(image_shape)}: {image_shape}"

		self.patch_embedding = PatchEmbedding(
			patch_shape,
			in_channels,
			out_channels
		)
		num_patches = (image_shape[0] // patch_shape[0]) * (image_shape[1] // patch_shape[1])
		
		# # 分类信息聚合 token 是单独一个特征值维度，需要拼接
		# self.cls_token = nn.Parameter(
		#	 torch.randn(1, 1, out_channels)
		# )
		# # 位置编码是一种加性影响，直接增加位置影响到嵌入向量上
		# self.pos_encod = nn.Parameter(
		#	 torch.randn(1, num_patches + 1, out_channels)
		# )
		
		self.pos_encod = nn.Parameter(
			torch.randn(1, num_patches, out_channels)
		)
	
	def forward(self, x: torch.Tensor):
		x = self.patch_embedding(x)
		B, _, _ = x.size()

		# cls_token = self.cls_token.expand(B, -1, -1)
		# x = torch.cat([cls_token, x], dim=1)
		x = x + self.pos_encod
		x = F.dropout(x, p=1e-2)

		return x
