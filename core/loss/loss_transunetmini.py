import torch
import torch.nn as nn

import easydict

from . import LOSS

@LOSS.register
class TransUNetMiniLoss(nn.Module):
	def __init__(self, *args, **kwds):
		super().__init__()

		kwds = easydict.EasyDict(kwds)
		self.args = kwds

		self.cls_weight = kwds.get("cls_weight", None)
		self.ignore_index = kwds.get("ignore_index", -1)
		if self.cls_weight is not None and not isinstance(self.cls_weight, torch.Tensor):
			self.cls_weight = torch.tensor(self.cls_weight)

		self.loss_fn = nn.CrossEntropyLoss(
			weight=self.cls_weight,
			ignore_index=self.ignore_index
		)
	
	def forward(self, batch):
		pred = batch["pred"]
		gdth = batch["gdth"]

		# 转换为适合多分类交叉熵损失函数的计算形式
		B, I, C, H, W = pred.size()
		pred = pred.view(B, C, I, H, W)

		loss= self.loss_fn(pred, gdth)
		# 还原方便后续指标计算？
		pred = pred.view(B, I, C, H, W)
		return loss
