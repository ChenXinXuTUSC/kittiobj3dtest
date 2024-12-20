import os
import os.path as osp
import sys
sys.path.append("..")

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

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
        ignore_cls: int, # should be the background class index
        num_epochs: int,
        log_alldir: str,
        log_interv: int = 10
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.cls_weight = cls_weight 
        if cls_weight is not None:
            self.cls_weight = torch.tensor(cls_weight)
        self.ignore_cls = ignore_cls

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        tfx_logdir = osp.join(log_alldir, timestamp, "tfx")
        if not osp.exists(tfx_logdir):
            os.makedirs(tfx_logdir, exist_ok=True)
        self.tfx_logger = SummaryWriter(log_dir=tfx_logdir)
        
        self.num_epochs = num_epochs
        self.log_interv = log_interv

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        model = self.model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss(
            weight=self.cls_weight,
            ignore_index=self.ignore_cls
        ).to(self.device)

        model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_accu = 0.0

            for batch_idx, (fmap, gdth) in enumerate(self.train_dataloader):
                fmap = fmap.to(self.device).float()
                gdth = gdth.to(self.device).long()

                pred = model(fmap)
                loss = criterion(pred, gdth)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                loss = loss.item()
                accu = (torch.argmax(pred, dim=1) == gdth).float()
                mask = gdth == self.ignore_cls
                accu = accu.sum().item() / mask.sum().item() # all correct pixel and all foreground pixel

                epoch_accu += accu
                epoch_loss += loss

                if (batch_idx + 1) % self.log_interv == 0:
                    print(
                        f"train [{epoch+1}/{self.num_epochs} {batch_idx / len(self.train_dataloader):.2f}] " + \
                        f"loss: {loss:.3f} " + \
                        f"accu: {accu:.3f} "
                    )
                    self.tfx_logger.add_scalar(
                        tag="train/loss", scalar_value=loss,
                        global_step=epoch * len(self.train_dataloader) + batch_idx
                    )
                    self.tfx_logger.add_scalar(
                        tag="train/accu", scalar_value=accu,
                        global_step=epoch * len(self.train_dataloader) + batch_idx
                    )

            epoch_loss /= len(self.train_dataloader)
            epoch_accu /= len(self.train_dataloader)
            print(f"train [{epoch+1}/{self.num_epochs}] avg_loss: {epoch_loss:.3f} avg_accu: {epoch_accu:.3f}")

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
            valid_accu = 0.0
            for batch_idx, (fmap, gdth) in enumerate(self.valid_dataloader):
                fmap = fmap.to(device=self.device)
                gdth = gdth.to(device=self.device)

                pred = model(fmap)
                loss = criterion(pred, gdth)

                loss = loss.item()
                accu = (torch.argmax(pred, dim=1) == gdth).float()
                mask = gdth == self.ignore_cls
                accu = accu.sum().item() / mask.sum().item() # all correct pixel and all foreground pixel

                valid_loss += loss
                valid_accu += accu

                if (batch_idx + 1) % self.log_interv == 0:
                    print(
                        f"valid [{epoch+1}/{self.num_epochs} {batch_idx / len(self.train_dataloader):.2f}] " + \
                        f"loss: {loss:.3f} " + \
                        f"accu: {accu:.3f} "
                    )
            self.tfx_logger.add_scalar(
                tag="valid/loss", scalar_value=valid_loss / len(self.valid_dataloader),
                global_step=(epoch + 1) * len(self.train_dataloader)
            )
            self.tfx_logger.add_scalar(
                tag="valid/accu", scalar_value=valid_accu / len(self.valid_dataloader),
                global_step=(epoch + 1) * len(self.train_dataloader)
            )
            
