import os
import os.path as osp
import sys
sys.path.append("..")
from datetime import datetime

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict

import core
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
        log_alldir: str,
        log_exname: str,
        log_interv: int = 10
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.cls_weight = cls_weight 
        if cls_weight is not None:
            self.cls_weight = torch.tensor(cls_weight, dtype=torch.float)
        
        self.ignore_cls = ignore_cls
        self.bkgcls_idx = bkgcls_idx

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        log_alldir = osp.join(log_alldir, log_exname, timestamp)
        tfx_logdir = osp.join(log_alldir, "tfx")
        if not osp.exists(tfx_logdir):
            os.makedirs(tfx_logdir, exist_ok=True)
        self.tfx_logger = SummaryWriter(log_dir=tfx_logdir)
        
        self.txt_logger = utils.get_logger(
            log_exname, True,
            osp.join(log_alldir, "log.txt")
        )


        self.num_epochs = num_epochs
        self.log_interv = log_interv

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train(self):
        self.txt_logger.info(f"CELoss weight: {self.cls_weight}, ignore_cls: {self.ignore_cls}")

        model = self.model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss(
            weight=self.cls_weight,
            ignore_index=self.ignore_cls
        ).to(self.device)

        self.txt_logger.info("Start training...")
        model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_macc = defaultdict(list)
            epoch_mrec = defaultdict(list)
            epoch_miou = defaultdict(list)

            for batch_idx, (fmap, gdth) in enumerate(self.train_dataloader):
                fmap = fmap.to(self.device).float()
                gdth = gdth.to(self.device).long()

                pred = model(fmap)
                loss = criterion(pred, gdth)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                loss = loss.item()
                epoch_loss += loss
                # accu = (torch.argmax(pred, dim=1) == gdth).float()
                # accu = accu[gdth != 0].mean().item()
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
                            "macc": sum([v for v in acc.values()]) / len(acc),
                            "mrec": sum([v for v in rec.values()]) / len(rec),
                            "miou": sum([v for v in iou.values()]) / len(iou),
                            "acc": acc, # already a dict
                            "rec": rec, # already a dict
                            "iou": iou  # already a dict
                        }, epoch=epoch, iter=batch_idx, to_tfb=True, tfx_tag="train"
                    )
                    # self.txt_logger.info(
                    #     f"train [{epoch+1}/{self.num_epochs} {batch_idx / len(self.train_dataloader):.2f}] " + \
                    #     f"loss: {loss:.3f} " + \
                    #     f"macc: {acc.mean():.3f} " + \
                    #     f"mrec: {rec.mean():.3f} " + \
                    #     f"miou: {iou.mean():.3f} "
                    # )
                    # self.tfx_logger.add_scalar(
                    #     tag="train/loss", scalar_value=loss,
                    #     global_step=epoch * len(self.train_dataloader) + batch_idx
                    # )
                    # self.tfx_logger.add_scalar(
                    #     tag="train/macc", scalar_value=acc.mean(),
                    #     global_step=epoch * len(self.train_dataloader) + batch_idx
                    # )
                    # self.tfx_logger.add_scalars(
                    #     main_tag="train/acc",
                    #     tag_scalar_dict={f"acc_{i}": acc[i] for i in range(acc.shape[0])},
                    #     global_step=epoch * len(self.train_dataloader) + batch_idx
                    # )

                    # visualize pred and gdth mask image
                    fmap_img = fmap[0][3].cpu().numpy()
                    pred_img = torch.argmax(pred[0], dim=0).cpu().numpy()
                    gdth_img = gdth[0].cpu().numpy()

                    # add color
                    pred_img = utils.visualize_fmap(pred_img)
                    gdth_img = utils.visualize_fmap(gdth_img)
                    self.tfx_logger.add_image(
                        tag="train/fmap_img", img_tensor=fmap_img, dataformats="HW",
                        global_step=epoch * len(self.train_dataloader) + batch_idx
                    )
                    self.tfx_logger.add_image(
                        tag="train/pred_img", img_tensor=pred_img, dataformats="HWC",
                        global_step=epoch * len(self.train_dataloader) + batch_idx
                    )
                    self.tfx_logger.add_image(
                        tag="train/gdth_img", img_tensor=gdth_img, dataformats="HWC",
                        global_step=epoch * len(self.train_dataloader) + batch_idx
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
                    "epoch_macc": epoch_macc,
                    "epoch_mrec": epoch_mrec,
                    "epoch_miou": epoch_miou,
                }, epoch=epoch + 1, iter=0, tfx_tag="train"
            )

            # self.valid(epoch)

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
            valid_macc = 0.0
            valid_mrec = 0.0
            valid_miou = 0.0

            for batch_idx, (fmap, gdth) in enumerate(self.valid_dataloader):
                fmap = fmap.to(device=self.device).float()
                gdth = gdth.to(device=self.device).long()

                pred = model(fmap)
                loss = criterion(pred, gdth)

                loss = loss.item()
                # accu = (torch.argmax(pred, dim=1) == gdth).float()
                # accu = accu[gdth != 0].mean().item()
                acc, rec, iou = self.compute_metrics(
                    pred, gdth,
                    self.train_dataloader.dataset.num_classes
                )
                valid_macc += acc
                valid_mrec += rec
                valid_miou += iou
                valid_loss += loss
                

                if (batch_idx + 1) % self.log_interv == 0:
                    self.log_metrics(
                        metric_dict={
                            "loss": loss,
                            "macc": acc.mean(),
                            "mrec": rec.mean(),
                            "miou": iou.mean()
                        }, epoch=epoch, iter=batch_idx, to_tfb=False, tag="valid"
                    )
            valid_loss /= len(self.valid_dataloader)
            valid_macc /= len(self.valid_dataloader)
            valid_mrec /= len(self.valid_dataloader)
            valid_miou /= len(self.valid_dataloader)
            self.log_metrics(
                metric_dict={
                    "loss": valid_loss / len(self.valid_dataloader),
                    "macc": valid_macc / len(self.valid_dataloader),
                }, epoch=epoch, iter=batch_idx, to_tfb=True, tag="valid"
            )
    

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

