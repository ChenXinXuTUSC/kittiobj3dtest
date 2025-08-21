import os
import os.path as osp
os.environ["TORCH_HOME"] = "."

import matplotlib.pyplot as plt

import argparse
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import core

parser = argparse.ArgumentParser("train.py")
parser.add_argument("--conf", type=str, help="main config file path")


def main():
	args = parser.parse_args()
	conf_main = core.conf.read(args.conf)
	pprint(conf_main)

	args_dataset = core.conf.read(conf_main.dataset.conf)
	args_model   = core.conf.read(conf_main.model.conf)
	args_loss	 = core.conf.read(conf_main.loss.conf)
	args_metric  = core.conf.read(conf_main.metric.conf)
	# pprint(args_dataset)
	# pprint(args_model)
	# pprint(args_loss)
	# pprint(args_metric)

	os.environ['MASTER_ADDR'] = 'localhost' # 设置一个本地地址
	os.environ['MASTER_PORT'] = '29500' # 设置一个本地端口
	torch.distributed.init_process_group(
		backend="nccl",
		world_size=1,
		rank=0
	)
	device = torch.device(0)

	# create corresponding dataset
	valid_dataset = core.dataset.DATASET[conf_main.dataset.name](**args_dataset, split="valid")
	testt_dataloader = torch.utils.data.DataLoader(
		dataset=valid_dataset,
		collate_fn=valid_dataset.collate_fn("valid"),
		batch_size=1,
		shuffle=True,
		pin_memory=True,
		drop_last=True
	)

	# create corresponding model
	# 多卡训练保存的参数名称和单卡训练不一样，需要添加 module 前缀，这里直接换成多卡模式加载模型参数，但是设置成单张卡环境
	model = nn.parallel.DistributedDataParallel(
		module=core.model.MODEL[conf_main.model.name](**args_model).to(device),
		device_ids=[0]
	)
	model.load_state_dict(torch.load("log/test/partanno+transunetmini/20250814120729/pth/best_valid.pth"))

	model.eval()
	for iter, batch in enumerate(testt_dataloader):
		data = batch["data"]
		gdth = batch["gdth"]
		rmap = batch["rmap"]

		pred = model(batch)
		# pred = torch.argmax(pred["out"], dim=1)
		# print(pred.shape)
		# pred_img = torch.argmax(pred["out"][0], dim=0).cpu().numpy()
		# pred_img = metriclog.visualize_fmap(pred_img)

		# retrive reverse map relationship
		# rmap = [x["ptpix"][:x["valid"]] for x in rmap]
		attn_list = model.module.encoder.attn_list # 分布式多卡训练的参数属性会多一个 module 前缀
		for i, _ in enumerate(attn_list):
			attn_list[i] = attn_list[i].squeeze(0)
			attn_list[i] = attn_list[i].mean(dim=0)
		
		# print(attn_list[-1][0].shape) # [384, 384]
		patch_attn = attn_list[0][0] # 最后一层编码器的第一个 patch 对其他 patch 的注意力
		patch_attn = patch_attn.view(6, 8, 8)
		patch_attn = F.interpolate(patch_attn.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
		# 因为 interpolate 期望输入是 (N, C, H, W) 或 (N, C, D, H, W)
		# 我们需要先给 x 增加一个批次维度（N=1），使其变成 [1, 6, 8, 8]
		# 插值完成后，再将这个批次维度移除
		patch_attn = patch_attn.squeeze(0)
		# print(patch_attn.shape) # [6, 64, 64]

		# 保存注意力图
		fig = plt.figure(figsize=(9, 6))
		for i, view_attn in enumerate(patch_attn):
			ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[])
			ax.set_title(f"view_{i+1}_attn")

			image = data[0][i][-1].detach().cpu().numpy()
			mask = np.concatenate([
				np.ones(image.shape),
				np.zeros(image.shape),
			], axis=1)

			image = np.concatenate([image, image], axis=1) # concate on row
			ax.imshow(image, cmap="gray", interpolation="nearest")

			view_attn = view_attn.detach().cpu().numpy()
			view_attn = np.concatenate([np.zeros(view_attn.shape), view_attn], axis=1)
			view_attn = np.ma.masked_where(mask == 1, view_attn)
			ax.imshow(view_attn, alpha=0.5, cmap='jet', interpolation='nearest')
		# 自动调整子图参数，使之填充整个图像区域，并避免重叠
		plt.tight_layout()
		# dpi=300 可以设置图片的分辨率，提高清晰度
		fig.savefig('attention_maps.png', bbox_inches='tight', dpi=150)

		# input("Press Enter to continue...")
		break


if __name__ == "__main__":
	main()
