import os
import os.path as osp
import sys
sys.path.append("..")
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict

import utils

class Trainer:
    '''
    - tfx_logdir: will add timestamp subdir
    '''
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        cls_weight: list, # may be None
        ignore_cls: int, # don't ignore the background class
        bkgcls_idx: int, # should be the background class index
        num_epochs: int,
        log_exname: str,
        log_alldir: str,
        log_interv: int = 10
    ):
        # model and data
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.cls_weight = cls_weight 
        if cls_weight is not None:
            self.cls_weight = torch.tensor(cls_weight, dtype=torch.float)
        
        self.ignore_cls = ignore_cls
        self.bkgcls_idx = bkgcls_idx

        # log staff
        self.log_interv = log_interv
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        log_alldir = osp.join(log_alldir, log_exname, timestamp)
        tfx_logdir = osp.join(log_alldir, "tfx")
        if not osp.exists(tfx_logdir):
            os.makedirs(tfx_logdir, exist_ok=True)
        self.tfx_logger = SummaryWriter(log_dir=tfx_logdir)
        self.txt_logger = utils.get_logger(
            log_exname, True,
            osp.join(log_alldir, "run.log")
        )
        self.ckpt_saved = osp.join(log_alldir, "ckpt")
        if not osp.exists(self.ckpt_saved):
            os.makedirs(self.ckpt_saved, exist_ok=True)

        # misc
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_miou = -1.0


    def train(self):
        model = self.model
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.parallel.DataParallel(self.model)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss(
            weight=self.cls_weight,
            ignore_index=self.ignore_cls
        ).to(self.device)

        self.txt_logger.info("start training...")
        self.txt_logger.info(f"CELoss weight: {self.cls_weight}, ignore_cls: {self.ignore_cls}")
        model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_macc = defaultdict(list)
            epoch_mrec = defaultdict(list)
            epoch_miou = defaultdict(list)

            for batch_idx, (fmap, gdth) in enumerate(self.train_dataloader):
                fmap = fmap.to(self.device).float()
                gdth = gdth.to(self.device).long()

                loss, pred = model(fmap, gdth, criterion)

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

                if (batch_idx + 1) % self.log_interv == 0:
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

            epoch_loss /= len(self.train_dataloader)
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
        criterion = torch.nn.CrossEntropyLoss(
            weight=self.cls_weight,
            ignore_index=self.ignore_cls
        ).to(self.device)
        model = self.model.to(device=self.device)
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_macc = defaultdict(list)
            valid_mrec = defaultdict(list)
            valid_miou = defaultdict(list)

            for batch_idx, (fmap, gdth) in enumerate(self.valid_dataloader):
                fmap = fmap.to(device=self.device).float()
                gdth = gdth.to(device=self.device).long()

                loss, pred = model(fmap, gdth, criterion)

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
            
            valid_loss /= len(self.valid_dataloader)
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
            if curr_miou > self.best_macc:
                self.txt_logger.info(f"best model saved with macc: {curr_miou:.3f}")
                self.best_macc = curr_miou
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
