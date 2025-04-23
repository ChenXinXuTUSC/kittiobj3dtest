import torch

import easydict
from collections import defaultdict

from tabulate import tabulate

import utils
from .base import Metric

from . import METRIC
@METRIC.register
class DeepLabV3Metric(Metric):
    def __init__(self, *args, **kwds):
        super().__init__()

        kwds = easydict.EasyDict(kwds)
        self.args = kwds

        self.all_iou = defaultdict(list)
        self.all_acc = defaultdict(list)
        self.all_rec = defaultdict(list)

        self.best_metric = defaultdict(list)
        self.mean_metric = defaultdict(list)

        # other attributes
        self.ignore_cls = self.args.ignore_cls
        self.bkgcls_idx = self.args.bkgcls_idx
    
    def reset(self):
        self.all_iou = defaultdict(lambda: [0])
        self.all_acc = defaultdict(lambda: [0])
        self.all_rec = defaultdict(lambda: [0])
        self.mean_metric = defaultdict(lambda: [0])
        self.best_metric = defaultdict(int)

    # should return a scalar
    def mean(self):
        return self.mean_metric["iou"][-1]

    # should return a scalar
    def best(self):
        return self.best_metric["iou"][-1]

    # should only be responsible for computing new metrics and store
    def mct(self,
        data: torch.Tensor,
        pred: torch.Tensor,
        gdth: torch.Tensor,
    ):
        iou, acc, rec = self.__compute_metrics(pred, gdth)
        for c, v in iou.items():
            self.all_iou[c].append(v)
        for c, v in acc.items():
            self.all_acc[c].append(v)
        for c, v in rec.items():
            self.all_rec[c].append(v)
        
        mean_iou = sum(iou.values()) / len(iou.keys())
        mean_acc = sum(acc.values()) / len(acc.keys())
        mean_rec = sum(rec.values()) / len(rec.keys())
        self.mean_metric["iou"].append(mean_iou)
        self.mean_metric["acc"].append(mean_acc)
        self.mean_metric["rec"].append(mean_rec)

        if mean_iou > self.best_metric["iou"]:
            self.best_metric["iou"] = mean_iou
        if mean_acc > self.best_metric["acc"]:
            self.best_metric["acc"] = mean_acc
        if mean_rec > self.best_metric["rec"]:
            self.best_metric["rec"] = mean_rec
    
    # console and tensorboard log
    # print the last computed metric
    def log(self,
        logger: utils.logger.LoggerTXTFX,
        prefix: str, 
        tfbtag: str,
        step: int,
        **misc_metrics
    ):
        assert isinstance(logger, utils.logger.LoggerTXTFX)
        # console log
        logmsg = [prefix]
        logmsg.append(f"loss {misc_metrics['loss']:.3f}")
        table_headers = sorted(self.all_iou.keys())
        logmsg.append(tabulate(
            tabular_data=[
                ["iou"] + [self.all_iou[key][-1] for key in table_headers],
                ["acc"] + [self.all_acc[key][-1] for key in table_headers],
                ["rec"] + [self.all_rec[key][-1] for key in table_headers],
            ],
            headers=["class"] + table_headers, tablefmt="fancy_grid", floatfmt=".3f")
        )
        logmsg = "\n".join(logmsg)
        logger.txt.info(logmsg)

        # tensorboard visualization
        iou = {f"{k}": v_list[-1] for k, v_list in self.all_iou.items()}
        acc = {f"{k}": v_list[-1] for k, v_list in self.all_acc.items()}
        rec = {f"{k}": v_list[-1] for k, v_list in self.all_rec.items()}
        logger.tfx.add_scalars(main_tag=f"{tfbtag}/iou", tag_scalar_dict=iou, global_step=step)
        logger.tfx.add_scalars(main_tag=f"{tfbtag}/acc", tag_scalar_dict=acc, global_step=step)
        logger.tfx.add_scalars(main_tag=f"{tfbtag}/rec", tag_scalar_dict=rec, global_step=step)
        
        logger.tfx.add_scalar(tag=f"{tfbtag}/miou", scalar_value=self.mean_metric["iou"][-1], global_step=step)
        logger.tfx.add_scalar(tag=f"{tfbtag}/loss", scalar_value=misc_metrics["loss"], global_step=step)

    
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

            # Intersection over Union
            iou[c] = tp / ((tp + fn) + fp)
            # accuracy
            acc[c] = (tp + tn) / (tp + tn + fp + fn)
            # recall
            rec[c] = tp / (tp + fn)

        return iou, acc, rec
