import torch

import easydict
from collections import defaultdict

import utils

from . import METRIC
@METRIC.register
class DeepLabV3Metric:
    def __init__(self, *args, **kwds):
        kwds = easydict.EasyDict(kwds)
        self.args = kwds

        self.all_iou = defaultdict(list)
        self.all_acc = defaultdict(list)
        self.all_rec = defaultdict(list)

        self.best_metric = 0.0
        self.mean_metric = 0.0

        # other attributes
        self.ignore_cls = self.args.ignore_cls
        self.bkgcls_idx = self.args.bkgcls_idx

    def __call__(
        self,
        logger: utils.logger.LoggerTXTFX,
        data: torch.Tensor, pred: torch.Tensor, gdth:torch.Tensor,
        prefix: str, tag: str, step: int, **misc_metrics
    ):
        assert isinstance(logger, utils.logger.LoggerTXTFX)

        iou, acc, rec = self.__compute_metrics(pred, gdth)
        for c in iou:
            self.all_iou[c].append(iou[c])
        for c in acc:
            self.all_acc[c].append(acc[c])
        for c in rec:
            self.all_rec[c].append(rec[c])
        
        mean_iou = [sum(self.all_acc[c]) / len(self.all_acc[c]) for c in self.all_acc]
        mean_iou = sum(mean_iou) / len(mean_iou)
        self.mean_metric = mean_iou
        if mean_iou > self.best_metric:
            self.best_metric = mean_iou
        
        logger.txt.info(" ".join([
            str(x) for x in [
                prefix,
                "loss", f"{misc_metrics['loss']:.3f}",
                "miou", f"{mean_iou:.3f}",
                "iou", iou,
                "acc", acc,
                "rec", rec
            ]
        ]))

        iou = {f"{k}": v for k, v in iou.items()}
        acc = {f"{k}": v for k, v in acc.items()}
        rec = {f"{k}": v for k, v in rec.items()}
        logger.tfx.add_scalar(tag=f"{tag}/miou", scalar_value=mean_iou, global_step=step)
        logger.tfx.add_scalar(tag=f"{tag}/loss", scalar_value=misc_metrics["loss"], global_step=step)
        logger.tfx.add_scalars(main_tag=f"{tag}/iou", tag_scalar_dict=iou, global_step=step)
        logger.tfx.add_scalars(main_tag=f"{tag}/acc", tag_scalar_dict=acc, global_step=step)
        logger.tfx.add_scalars(main_tag=f"{tag}/rec", tag_scalar_dict=rec, global_step=step)
    
    def reset(self):
        self.all_iou = defaultdict(list)
        self.all_acc = defaultdict(list)
        self.all_rec = defaultdict(list)
        self.mean_metric = 0.0

    def mean(self):
        return self.mean_metric

    def best(self):
        return self.best_metric
    
    def __compute_metrics(self, pred: torch.Tensor, gdth: torch.Tensor):
        # pred 和 gdth 是形状为 (batch_size, height, width) 的张量
        pred = pred["out"] # 预测结果具体使用方式见模型的 forward 函数返回值
        if len(pred.shape) > 3:
            pred = torch.argmax(pred, dim=1)
        
        iou = {}
        acc = {}
        rec = {}
        # 只计算在 gdth 中出现的类别
        for c in torch.unique(gdth):
            c = c.item()
            if c == self.ignore_cls or c == self.bkgcls_idx:
                continue
            # 计算 TP, TN, FP, FN
            tp = ((pred == c) & (gdth == c)).sum().float().item()
            tn = ((pred != c) & (gdth != c)).sum().float().item()
            fp = ((pred == c) & (gdth != c)).sum().float().item()
            fn = ((pred != c) & (gdth == c)).sum().float().item()

            # accuracy
            acc[c] = tp / (tp + fn)

            # recall
            rec[c] = tp / (tp + fn)

            # Intersection over Union
            iou[c] = tp / ((tp + fn) + fp)

        return iou, acc, rec
