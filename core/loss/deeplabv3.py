import torch
import torch.nn as nn
import torch.nn.functional as F

import easydict

from . import LOSS

@LOSS.register
class DeepLabV3Loss(nn.Module):
    def __init__(self, *args, **kwds):
        super().__init__()
        kwds = easydict.EasyDict(kwds)
        self.args = kwds

        self.weight = kwds.get("weight", None)
        self.ignore_index = kwds.get("ignore_index", 255)

        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(self.weight),
            ignore_index=self.ignore_index
        )
    
    def forward(self, preds, target):
        loss_out = self.loss_fn(preds["out"], target)
        loss_aux = self.loss_fn(preds["aux"], target)
        loss = loss_out + 0.5 * loss_aux
        return loss
