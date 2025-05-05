# %%
import sys
sys.path.append("..")

import os
import os.path as osp
os.environ["TORCH_HOME"] = "."

import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torch.nn.parallel
import torch.distributed

import numpy as np

# %%
import core

# %%
conf_main = core.readconfyaml.read("conf/pipe/deeplab+kittisemantic.yaml")

args_dataset = core.readconfyaml.read(conf_main.dataset.conf_yaml)
args_model   = core.readconfyaml.read(conf_main.model.conf_yaml)
args_loss	= core.readconfyaml.read(conf_main.loss.conf_yaml)
args_metric  = core.readconfyaml.read(conf_main.metric.conf_yaml)

# %%
testt_dataset = core.dataset.KITTISemantic(
	split="testt",
	**args_dataset
)

testt_dataloader = torch.utils.data.DataLoader(
	dataset=testt_dataset,
	batch_size=1,
	shuffle=True,
	pin_memory=True,
	drop_last=True
)

# %%
metriclog = core.metric.DeepLabV3Metric(
	**args_metric
)

# %%
os.environ['MASTER_ADDR'] = 'localhost' # 设置一个本地地址
os.environ['MASTER_PORT'] = '29500' # 设置一个本地端口

torch.distributed.init_process_group(
	backend="nccl",
	world_size=1,
	rank=0
)
device = torch.device(0)

model = nn.parallel.DistributedDataParallel(
	core.model.DeepLabV3(
		in_channels=5,
		out_channels=34
	).to(device),
	device_ids=[0]
)
model.load_state_dict(torch.load("log/test/deeplabv3+kittisem/20250427194725/pth/best_valid.pth"))

# %%


model.eval()
for iter, (data, gdth) in enumerate(testt_dataloader):
	data = data.to(device).float()
	gdth = gdth.to(device).long()

	pred = model(data)
	pred_img = torch.argmax(pred["out"][0], dim=0).cpu().numpy()
	pred_img = metriclog.visualize_fmap(pred_img)
	break

# %%
import matplotlib.pyplot as plt

HEI, WID = pred_img.shape[:2]
plt.figure(figsize=(12, 6), dpi=128, tight_layout=True)
plt.axis([WID, 0, 0, HEI])
plt.axis("off")
plt.imshow(pred_img, interpolation='nearest')

# %%



