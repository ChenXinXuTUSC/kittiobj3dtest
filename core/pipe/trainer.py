import os
import os.path as osp
import sys
sys.path.append("..")
from datetime import datetime

import torch
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict

import utils

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
        metric_fn: callable,
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
        self.metric_fn = metric_fn

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        # log staff
        self.log_interv = log_interv
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.log_alldir = osp.join(log_alldir, log_exname, timestamp)
        self.log_exname = log_exname

    def ddp_setup(self, rank: int, world_size: int):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # Initialize the process group
        backend = "nccl" if self.device_mod == "cuda" else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    def ddp_cleanup(self):
        dist.destroy_process_group()

    
    def run(self):
        mp.spawn(
            fn=self.ddp_entry,
            args=(self.world_size,),
            nprocs=self.world_size,
            join=True
        )
    
    def ddp_entry(self, world_rank: int, world_size: int):
        self.world_rank = world_rank
        self.world_size = world_size

        # =====================================================================
        # ============================= ddp setup =============================
        self.ddp_setup(world_rank, world_size)
        # ============================= ddp setup =============================
        # =====================================================================

        self.tfx_logger = None
        self.txt_logger = None
        
        if world_rank == 0:
            tfx_logdir = osp.join(self.log_alldir, "tfx")
            if not osp.exists(tfx_logdir):
                os.makedirs(tfx_logdir, exist_ok=True)
            self.tfx_logger = SummaryWriter(log_dir=tfx_logdir)
            self.txt_logger = utils.get_logger(
                self.log_exname, True,
                osp.join(self.log_alldir, "run.log")
            )
            self.ckpt_saved = osp.join(self.log_alldir, "ckpt")
            if not osp.exists(self.ckpt_saved):
                os.makedirs(self.ckpt_saved, exist_ok=True)

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



        # =====================================================================
        # ============================ entry point ============================
        if world_rank == 0:
            self.txt_logger.info("start training...")
            self.txt_logger.info(f"model: {self.model}")
        self.train()
        # ============================ entry point ============================
        # =====================================================================



        # =====================================================================
        # ============================ ddp cleanup ============================
        self.ddp_cleanup()
        # ============================ ddp cleanup ============================
        # =====================================================================


    def train(self):
        train_sampler = self.train_sampler
        train_dataloader = self.train_dataloader
        model = self.model
        criterion = self.criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        self.best_miou = 0.0

        model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_macc = defaultdict(list)
            epoch_mrec = defaultdict(list)
            epoch_miou = defaultdict(list)

            train_sampler.set_epoch(epoch)
            for batch_idx, (data, gdth) in enumerate(train_dataloader):
                data = data.to(self.device).float()
                gdth = gdth.to(self.device).long()

                pred = model(data)
                loss = criterion(pred, gdth)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.item()
                epoch_loss += loss
            
                acc, rec, iou = self.compute_metrics(pred, gdth)
                for k, v in acc.items():
                    epoch_macc[k].append(v)
                for k, v in rec.items():
                    epoch_mrec[k].append(v)
                for k, v in iou.items():
                    epoch_miou[k].append(v)

                if self.world_rank == 0 and (batch_idx + 1) % self.log_interv == 0:
                    self.log_metrics(
                        metric_dict={
                            "loss": loss,
                            "miou": sum([v for v in iou.values()]) / len(iou),
                            "mrec": sum([v for v in rec.values()]) / len(rec),
                            "macc": sum([v for v in acc.values()]) / len(acc),
                            "iou": iou, # already a dict
                            "rec": rec, # already a dict
                            "acc": acc  # already a dict
                        }, epoch=epoch, iter=batch_idx, to_tfb=True, tfx_tag="train"
                    )


            if self.world_rank == 0:
                epoch_loss /= len(train_dataloader)
                for k, v in epoch_macc.items():
                    epoch_macc[k] = sum(v) / len(v)
                for k, v in epoch_mrec.items():
                    epoch_mrec[k] = sum(v) / len(v)
                for k, v in epoch_miou.items():
                    epoch_miou[k] = sum(v) / len(v)
                self.log_metrics(
                    metric_dict={
                        "epoch_mlss": epoch_loss,
                        "epoch_miou": epoch_miou,
                        "epoch_mrec": epoch_mrec,
                        "epoch_macc": epoch_macc
                    }, epoch=epoch, iter=len(self.train_dataloader), tfx_tag="train"
                )

            self.valid(epoch)

            scheduler.step()


    def valid(self, epoch: int):
        valid_sampler = self.valid_sampler
        valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=4,
            pin_memory=True,
        )
        model = self.model

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_macc = defaultdict(list)
            valid_mrec = defaultdict(list)
            valid_miou = defaultdict(list)

            valid_sampler.set_epoch(epoch)
            for batch_idx, (fmap, gdth) in enumerate(valid_dataloader):
                fmap = fmap.to(self.device).float()
                gdth = gdth.to(self.device).long()

                loss, pred = model(fmap, gdth)

                loss = loss.item()
                valid_loss += loss

                acc, rec, iou = self.compute_metrics(pred, gdth)
                for k, v in acc.items():
                    valid_macc[k].append(v)
                for k, v in rec.items():
                    valid_mrec[k].append(v)
                for k, v in iou.items():
                    valid_miou[k].append(v)

                if (batch_idx + 1) % self.log_interv == 0:
                    self.log_metrics(
                        metric_dict={
                            "loss": loss,
                            "miou": sum([v for v in iou.values()]) / len(iou),
                            "mrec": sum([v for v in rec.values()]) / len(rec),
                            "macc": sum([v for v in acc.values()]) / len(acc)
                        }, epoch=epoch, iter=batch_idx, to_tfb=False, tfx_tag="valid"
                    )
                    self.log_images(fmap, pred, gdth, epoch, batch_idx, "valid")
            
            valid_loss /= len(valid_dataloader)
            for k, v in valid_macc.items():
                valid_macc[k] = sum(v) / len(v)
            for k, v in valid_mrec.items():
                valid_mrec[k] = sum(v) / len(v)
            for k, v in valid_miou.items():
                valid_miou[k] = sum(v) / len(v)
            self.log_metrics(
                metric_dict={
                    "epoch_mlss": valid_loss,
                    "epoch_miou": valid_miou,
                    "epoch_mrec": valid_mrec,
                    "epoch_macc": valid_macc
                }, epoch=epoch, iter=len(self.train_dataloader), to_tfb=True, tfx_tag="valid"
            )

            curr_miou = sum([v for v in valid_miou.values()]) / len(valid_miou)
            if curr_miou > best_miou:
                self.txt_logger.info(f"best model saved with macc: {curr_miou:.3f}")
                best_miou = curr_miou
                torch.save(self.model.state_dict(), osp.join(self.ckpt_saved, "best.pth"))
    

    def compute_metrics(self, pred, gdth):
        # pred 和 gdth 是形状为 (batch_size, height, width) 的张量
        if len(pred.shape) > 3:
            pred = torch.argmax(pred, dim=1)
        
        acc = {}
        rec = {}
        iou = {}
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

        return acc, rec, iou


    def log_metrics(
            self, metric_dict: dict,
            epoch: int, iter: int, to_tfb: bool = True, tfx_tag: str = "train"
        ):

        log_str = " ".join([
            f"{tfx_tag} [{epoch+1}/{self.num_epochs} {iter / len(self.train_dataloader):.2f}%]",
            *[f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metric_dict.items()]
        ])
        self.txt_logger.info(log_str)
        
        if not to_tfb:
            return
        global_step = epoch * len(self.train_dataloader) + iter
        for k, v in metric_dict.items():
            if isinstance(v, dict):
                self.tfx_logger.add_scalars(
                    main_tag=f"{tfx_tag}/{k}",
                    tag_scalar_dict={str(ik): iv for (ik, iv) in v.items()},
                    global_step=global_step
                )
            else:
                self.tfx_logger.add_scalar(f"{tfx_tag}/{k}", v, global_step)
            self.tfx_logger.flush()


    def log_images(
            self, data: torch.Tensor, pred: torch.Tensor, gdth: torch.Tensor,
            epoch: int, iter: int, tfx_tag: str
        ):
        # visualize pred and gdth mask image
        fmap_img = data[0][3].cpu().numpy()
        pred_img = torch.argmax(pred[0], dim=0).cpu().numpy()
        gdth_img = gdth[0].cpu().numpy()

        # add color
        pred_img = utils.visualize_fmap(pred_img)
        gdth_img = utils.visualize_fmap(gdth_img)
        self.tfx_logger.add_image(
            tag="train/fmap_img", img_tensor=fmap_img, dataformats="HW",
            global_step=epoch * len(self.train_dataloader) + iter
        )
        self.tfx_logger.add_image(
            tag="train/pred_img", img_tensor=pred_img, dataformats="HWC",
            global_step=epoch * len(self.train_dataloader) + iter
        )
        self.tfx_logger.add_image(
            tag="train/gdth_img", img_tensor=gdth_img, dataformats="HWC",
            global_step=epoch * len(self.train_dataloader) + iter
        )
