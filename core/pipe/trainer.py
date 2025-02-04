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
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        criterion: torch.nn.Module,
        metriclog: core.metric.Metric,
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
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
        )

        self.valid_sampler = DistributedSampler(
            dataset=self.valid_dataset, num_replicas=world_size, rank=world_rank
        )
        self.valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            sampler=self.valid_sampler,
            num_workers=4,
            pin_memory=True,
        )

        if self.world_rank == 0:
            self.loggertfx = utils.logger.LoggerTXTFX(
                self.log_alldir, name=self.log_exname
            )
        else:
            self.loggertfx = None

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
        model = self.model
        model.train()

        train_sampler = self.train_sampler
        train_dataloader = self.train_dataloader
        train_sampler.set_epoch(epoch)

        for iter, (data, gdth) in enumerate(train_dataloader):
            data = data.to(self.device).float()
            gdth = gdth.to(self.device).long()

            pred = model(data)
            loss = self.criterion(pred, gdth)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            if self.world_rank == 0 and (iter + 1) % self.log_interv == 0:
                self.metriclog(
                    self.loggertfx, data, pred, gdth,
                    prefix=f"train [{epoch+1}/{self.num_epochs} {iter / len(self.train_dataloader):.2f}%]",
                    tag="train", step=epoch * len(self.train_dataloader) + iter,
                    loss=loss.item()
                )

    def valid_epoch(self, epoch: int):
        model = self.model
        model.eval()

        valid_sampler = self.valid_sampler
        valid_dataloader = self.valid_dataloader

        self.metriclog.reset()
        with torch.no_grad():
            valid_sampler.set_epoch(epoch)
            for iter, (data, gdth) in enumerate(valid_dataloader):
                data = data.to(self.device).float()
                gdth = gdth.to(self.device).long()

                pred = model(data)
                loss = self.criterion(pred, gdth)

                if self.world_rank == 0 and (iter + 1) % self.log_interv == 0:
                    # self.metriclog(

                    # )
                    pass

        if self.world_rank == 0:
            curr_metric = self.metriclog.mean()
            if curr_metric > self.metriclog.best():
                self.loggertfx.txt.info(f"best model saved with metric: {curr_metric}")
                torch.save(self.model.state_dict(), osp.join(self.ckpt_saved, "best.pth"))

    # def log_images(
    #         self, data: torch.Tensor, pred: torch.Tensor, gdth: torch.Tensor,
    #         epoch: int, iter: int, tfx_tag: str
    #     ):
    #     # visualize pred and gdth mask image
    #     fmap_img = data[0][3].cpu().numpy()
    #     pred_img = torch.argmax(pred[0], dim=0).cpu().numpy()
    #     gdth_img = gdth[0].cpu().numpy()

    #     # add color
    #     pred_img = utils.visualize_fmap(pred_img)
    #     gdth_img = utils.visualize_fmap(gdth_img)
    #     self.tfx_logger.add_image(
    #         tag=f"{tfx_tag}/fmap_img", img_tensor=fmap_img, dataformats="HW",
    #         global_step=epoch * len(self.train_dataloader) + iter
    #     )
    #     self.tfx_logger.add_image(
    #         tag=f"{tfx_tag}/train/pred_img", img_tensor=pred_img, dataformats="HWC",
    #         global_step=epoch * len(self.train_dataloader) + iter
    #     )
    #     self.tfx_logger.add_image(
    #         tag=f"{tfx_tag}/train/gdth_img", img_tensor=gdth_img, dataformats="HWC",
    #         global_step=epoch * len(self.train_dataloader) + iter
    #     )
