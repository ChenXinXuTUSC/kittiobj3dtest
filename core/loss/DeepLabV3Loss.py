import torch
import torch.nn as nn

import easydict

from . import LOSS

@LOSS.register
class DeepLabV3Loss(nn.Module):
	def __init__(self, *args, **kwds):
		super().__init__()
		kwds = easydict.EasyDict(kwds)
		self.args = kwds

		self.weight = kwds.get("weight", None)
		self.ignore_index = kwds.get("ignore_index", -100)
		if self.weight is not None:
			self.weight = torch.tensor(self.weight)

		self.loss_fn = nn.CrossEntropyLoss(
			weight=self.weight,
			ignore_index=self.ignore_index
		)
	
	def forward(self, batch):
		pred = batch["pred"]
		gdth = batch["gdth"]

		loss_out = self.loss_fn(pred["out"], gdth)
		loss_aux = self.loss_fn(pred["aux"], gdth)
		loss = loss_out + 0.5 * loss_aux
		return loss
