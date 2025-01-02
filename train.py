import os
os.environ["TORCH_HOME"] = "."
import os.path as osp
import argparse

import torch
import torch.utils.data

import core


parser = argparse.ArgumentParser("train.py")
args_group_conf = parser.add_argument_group("conf", "arguments related to conf read")
args_group_conf.add_argument("--conf_datad_yaml", type=str, help="path to datad conf file")
args_group_conf.add_argument("--conf_model_yaml", type=str, help="path to model conf file")
args_group_conf.add_argument("--conf_train_yaml", type=str, help="path to train conf file")


def main():
    args = parser.parse_args()
    conf_datad = core.readconfyaml.read(args.conf_datad_yaml)
    conf_model = core.readconfyaml.read(args.conf_model_yaml)
    conf_train = core.readconfyaml.read(args.conf_train_yaml)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=core.dataset.DATASET[conf_train.dataset](
            conf_datad.root, "train", conf_datad
        ),
        batch_size=conf_train.batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=core.dataset.DATASET[conf_train.dataset](
            conf_datad.root, "valid", conf_datad
        ),
        batch_size=conf_train.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = core.model.MODEL[conf_train.model](
        in_channels=conf_model.in_channels,
        out_channels=conf_model.num_classes
    )

    trainer = core.pipe.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        cls_weight=conf_train.cls_weight,
        ignore_cls=conf_train.ignore_cls,
        bkgcls_idx=conf_train.bkgcls_idx,
        num_epochs=conf_train.num_epochs,
        log_exname=conf_train.exp_name,
        log_alldir=conf_train.log_alldir,
        log_interv=conf_train.log_interv
    )

    trainer.train()


if __name__ == "__main__":
    main()
