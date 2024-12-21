import os
import os.path as osp
import argparse

import torch
import torch.utils.data

import core


parser = argparse.ArgumentParser("train.py")
args_group_conf = parser.add_argument_group("conf", "arguments related to conf read")
args_group_conf.add_argument("--conf_data_yaml", type=str, help="path to dataset conf file")
args_group_conf.add_argument("--conf_model_yaml", type=str, help="path to model conf file")
args_group_conf.add_argument("--conf_trainer_yaml", type=str, help="path to trainer conf file")


def main():
    args = parser.parse_args()
    conf_dataset = core.readconfyaml.read(args.conf_data_yaml)
    conf_model = core.readconfyaml.read(args.conf_model_yaml)
    conf_trainer = core.readconfyaml.read(args.conf_trainer_yaml)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=core.dataset.KITTISpherical(
            "data", "train", conf_dataset
        ),
        batch_size=conf_trainer.batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=core.dataset.KITTISpherical(
            "data", "valid", conf_dataset
        ),
        batch_size=conf_trainer.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = core.model.squeezeseg.SqueezeSeg(
        in_channels=conf_model.in_channels,
        out_channels=conf_model.num_classes,
        conf=conf_model
    )

    trainer = core.pipe.Trainer(
        model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        cls_weight=conf_trainer.cls_weight,
        ignore_cls=conf_trainer.ignore_cls,
        bkgcls_idx=conf_trainer.bkgcls_idx,
        num_epochs=conf_trainer.num_epochs,
        log_alldir=conf_trainer.log_alldir,
        log_exname=conf_trainer.exp_name,
        log_interv=conf_trainer.log_interv
    )

    trainer.train()


if __name__ == "__main__":
    main()
