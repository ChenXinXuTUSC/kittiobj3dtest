import torch
import numpy as np

import easydict
from collections import defaultdict
from tabulate import tabulate

import utils

from .metric_base import BaseMetricLog
from . import METRIC
@METRIC.register
class TransUNetMiniMetric(BaseMetricLog):
	def __init__(self, *args, **kwds):
		super().__init__()

		kwds = easydict.EasyDict(kwds)
		self.args = kwds

		self.pallete = self.args.pallete

		self.all_iou = defaultdict(list)
		self.all_acc = defaultdict(list)
		self.all_rec = defaultdict(list)

		# 用于指示训练验证过程中是否保存模型参数的最好指标
		# 应该是一个标量？否则的话训练主框架中会引入判断指标的逻辑
		# 有代码侵入的风险
		self.best_metric = -1
		self.mean_metric = defaultdict(list)

		# other attributes
		self.ignore_cls = self.args.ignore_cls
		self.bkgcls_idx = self.args.bkgcls_idx

		# snapshot of the log data
		self.data = None
		self.prec = None
		self.gdth = None
	
	def reset(self):
		self.all_iou = defaultdict(list)
		self.all_acc = defaultdict(list)
		self.all_rec = defaultdict(list)
		self.mean_metric = defaultdict(list)
		# self.best_metric = float("inf")

	# should return a scalar
	def mean(self):
		if len(self.mean_metric["iou"]) == 0:
			return 0
		# 返回到目前位置积累的所有样本批次的所有类别的平均交并比的均值
		ret = sum(self.mean_metric["iou"]) / (len(self.mean_metric["iou"]))
		return ret

	# should return a scalar
	def best(self) -> bool:
		curr_metric = self.mean()
		new_best = False
		if curr_metric > self.best_metric:
			new_best = True
			self.best_metric = curr_metric
		
		return new_best

	# should only be responsible for computing new metrics and store
	def update_metrics(self,
		# data: torch.Tensor,
		# pred: torch.Tensor,
		# gdth: torch.Tensor,
		batch
	):
		self.data = batch["data"]
		self.pred = batch["pred"]
		self.gdth = batch["gdth"]

		iou, acc, rec = self.__compute_metrics(self.pred, self.gdth)
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
	
	# console and tensorboard log
	# print the last computed metric
	def log_metrics(self,
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
		logmsg.append(
			tabulate(
				tabular_data=[
					["iou"] + [self.all_iou[key][-1] for key in table_headers],
					["acc"] + [self.all_acc[key][-1] for key in table_headers],
					["rec"] + [self.all_rec[key][-1] for key in table_headers],
				],
				headers=["class"] + table_headers, tablefmt="fancy_grid", floatfmt=".3f"
			)
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

		# visualize pred and gdth mask image
		data = self.data[0][0][-1] # 第一个样本的第一个投影视图
		pred = self.pred[0][0] # 第一个样本的第一个投影视图
		gdth = self.gdth[0][0] # 第一个样本的第一个投影视图

		# 原始特征图数据特征通道是 [x, y, z, r] ，最后一维是深度特征
		fmap_img = data.cpu().numpy()
		# 用户应该知晓自己的模型输出是什么，然后在这里自己处理
		pred_img = torch.argmax(pred, dim=0).cpu().numpy()
		gdth_img = gdth.cpu().numpy()

		# add color
		pred_img = self.visualize_fmap(pred_img)
		gdth_img = self.visualize_fmap(gdth_img)
		logger.tfx.add_image(
			tag=f"{tfbtag}/fmap_img", img_tensor=fmap_img, dataformats="HW",
			global_step=step
		)
		logger.tfx.add_image(
			tag=f"{tfbtag}/pred_img", img_tensor=pred_img, dataformats="HWC",
			global_step=step
		)
		logger.tfx.add_image(
			tag=f"{tfbtag}/gdth_img", img_tensor=gdth_img, dataformats="HWC",
			global_step=step
		)

	def __compute_metrics(self, pred: torch.Tensor, gdth: torch.Tensor):
		# pred: [B, I, C, H, W]
		# gdth: [B, I, H, W]
		pred = torch.argmax(pred, dim=2)

		iou = {}
		acc = {}
		rec = {}
		# 只计算在 gdth 中出现的类别
		for c in torch.unique(gdth):
			c = c.item()
			# 忽略指定的类别，如背景
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

	def visualize_fmap(self, fmap: np.ndarray):
		'''
		- fmap: [H, W] shape
		'''
		img = np.zeros((*fmap.shape[:2], 3), dtype=np.uint8)
		for idx, clr in enumerate(self.pallete):
			img[fmap == idx] = np.array(clr)
		return img
