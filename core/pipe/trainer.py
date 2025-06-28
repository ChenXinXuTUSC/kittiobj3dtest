import os
import os.path as osp
import sys
sys.path.append("..")

import torch
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import utils
import core

class Trainer:
	def __init__(
		self,
		# distributed training args
		device_mod: str,
		world_size: int,
		# model and data args
		model: torch.nn.Module,
		train_dataset: core.dataset.BaseDataset,
		valid_dataset: core.dataset.BaseDataset,
		criterion: torch.nn.Module,
		metriclog: core.metric.BaseMetricLog,
		# training args
		num_epochs: int,
		batch_size: int,
		lr: float,
		# logging args
		log_exname: str,
		log_alldir: str,
		log_interv: int = 10
	):
		# model and data
		assert device_mod == "cpu" or \
			   device_mod == "cuda", \
			   f"invalid device {device_mod}"
		self.device_mod = device_mod
		self.world_size = world_size

		self.model = model
		self.train_dataset = train_dataset
		self.valid_dataset = valid_dataset
		self.criterion = criterion
		self.metriclog = metriclog

		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.lr = lr

		# log staff
		self.log_interv = log_interv
		self.log_alldir = log_alldir
		self.log_exname = log_exname

		print(device_mod, world_size)

	def ddp_setup(self, world_rank: int, world_size: int):
		os.environ['MASTER_ADDR'] = 'localhost'
		os.environ['MASTER_PORT'] = '12355'

		# Initialize the process group
		backend = "nccl" if self.device_mod == "cuda" else "gloo"
		dist.init_process_group(backend=backend, rank=world_rank, world_size=world_size)

		self.device = torch.device(world_rank if self.device_mod == "cuda" else "cpu")

		self.model = self.model.to(self.device)
		self.model = DDP(self.model, device_ids=[world_rank] if self.device_mod == "cuda" else None)
		self.criterion = self.criterion.to(self.device)

		self.train_sampler = DistributedSampler(
			self.train_dataset, num_replicas=world_size, rank=world_rank
		)
		self.train_dataloader = DataLoader(
			dataset=self.train_dataset,
			collate_fn=self.train_dataset.collate_fn("train"),
			batch_size=self.batch_size,
			sampler=self.train_sampler,
			num_workers=4,
			pin_memory=True,
			drop_last=True # 防止批次归一化 BN 的时候，最后一个批次只有一个样本导致抛出异常
		)

		self.valid_sampler = DistributedSampler(
			dataset=self.valid_dataset, num_replicas=world_size, rank=world_rank
		)
		self.valid_dataloader = DataLoader(
			dataset=self.valid_dataset,
			collate_fn=self.valid_dataset.collate_fn("valid"),
			batch_size=self.batch_size,
			sampler=self.valid_sampler,
			num_workers=4,
			pin_memory=True,
			drop_last=True
		)

		if self.world_rank == 0:
			self.loggertfx = utils.logger.LoggerTXTFX(
				self.log_alldir, name=self.log_exname
			)
			self.ckpt_save_path = osp.join(self.loggertfx.get_root(), "pth")
			os.makedirs(self.ckpt_save_path, exist_ok=True)
		else:
			self.loggertfx = None
			self.ckpt_save_path = None

	def ddp_cleanup(self):
		dist.destroy_process_group()

	def ddp_entry(self, world_rank: int, world_size: int):
		self.world_rank = world_rank
		self.world_size = world_size

		# =====================================================================
		# ============================= ddp setup =============================
		self.ddp_setup(world_rank, world_size)
		# ============================= ddp setup =============================
		# =====================================================================


		# =====================================================================
		# ============================ entry point ============================
		self.pipe()
		# ============================ entry point ============================
		# =====================================================================


		# =====================================================================
		# ============================ ddp cleanup ============================
		self.ddp_cleanup()
		# ============================ ddp cleanup ============================
		# =====================================================================
	
	def run(self):
		mp.spawn(
			fn=self.ddp_entry,
			args=(self.world_size,),
			nprocs=self.world_size,
			join=True
		)

	def pipe(self):
		self.criterion = self.criterion
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

		self.metriclog.reset()
		for epoch in range(self.num_epochs):
			self.train_epoch(epoch)
			self.valid_epoch(epoch)
			self.scheduler.step()

	def train_epoch(self, epoch: int):
		lossf = self.criterion
		mtlog = self.metriclog
		tflog = self.loggertfx

		train_sampler = self.train_sampler
		train_dataloader = self.train_dataloader
		train_sampler.set_epoch(epoch)

		optimizer = self.optimizer

		model = self.model
		model.train()

		for iter, batch in enumerate(train_dataloader):
			batch = self.__to_device__(batch, self.device)

			pred = model(batch)
			batch["pred"] = pred # add prediction to batched data dict
			loss = lossf(batch)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			mtlog.update_metrics(batch)
			if self.world_rank == 0 and (iter + 1) % self.log_interv == 0:
				mtlog.log_metrics(
					tflog,
					prefix=f"train [{epoch+1}/{self.num_epochs} {iter / len(self.train_dataloader) * 100:.2f}%]",
					tfbtag="train",
					step=epoch * len(self.train_dataloader) + iter,
					loss=loss.item()
				)
		
		if self.world_rank == 0:
			# only main process have file access
			torch.save(
				self.model.state_dict(),
				osp.join(self.ckpt_save_path, "last_train.pth")
			)

	def valid_epoch(self, epoch: int):
		# just for abbreviation
		lossf = self.criterion
		mtlog = self.metriclog
		tflog = self.loggertfx

		valid_sampler = self.valid_sampler
		valid_dataloader = self.valid_dataloader

		model = self.model
		model.eval()

		mtlog.reset() # 重置指标，去除训练阶段累积
		with torch.no_grad():
			valid_sampler.set_epoch(epoch)
			for iter, batch in enumerate(valid_dataloader):
				batch = self.__to_device__(batch, self.device)

				pred = model(batch)
				batch["pred"] = pred
				loss = lossf(batch)

				# record current iter's metric info
				mtlog.update_metrics(batch)

				if self.world_rank == 0 and (iter + 1) % self.log_interv == 0:
					mtlog.log_metrics(
						tflog,
						prefix=f"valid [{epoch+1}/{self.num_epochs} {iter / len(self.valid_dataloader) * 100:.2f}%]",
						tfbtag="valid",
						step=epoch * len(self.valid_dataloader) + iter,
						loss=loss.item()
					)

		if self.world_rank == 0:
			if mtlog.best():
				tflog.txt.info(
					f"save a new best metric ckp: {self.metriclog.mean()}"
				)
				torch.save(self.model.state_dict(), osp.join(self.ckpt_save_path, "best_valid.pth"))
		# 每个 epoch 重制一次指标累积
		mtlog.reset()

	def __to_device__(self, batch, device: torch.device):
		'''
		[deepseek generated] recursively move all torch.Tensor type object
		to specified device.
		'''
		if isinstance(batch, torch.Tensor):
			return batch.to(device)
		# 处理字典类型（Key保持不变，只移动Value）
		elif isinstance(batch, dict):  # 包括 dict 及其他映射类型
			return {k: self.__to_device__(v, device) for k, v in batch.items()}
		# 处理其他可迭代对象（list/tuple/set等）
		elif isinstance(batch, list) and not isinstance(batch, (str, bytes)):
			return [self.__to_device__(x, device) for x in batch]
		# 非迭代、非Tensor对象直接返回
		else:
			return batch
