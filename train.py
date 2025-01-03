import os

import core.metric
os.environ["TORCH_HOME"] = "."
import os.path as osp
import argparse

import torch
import torch.utils.data

import core

parser = argparse.ArgumentParser("train.py")
parser.add_argument("--conf", type=str, help="main configuration yaml file path")


def main():
    args = parser.parse_args()
    conf_main = core.readconfyaml.read(args.conf)
    
    args_dataset = core.readconfyaml.read(conf_main.dataset.conf_yaml)
    args_model   = core.readconfyaml.read(conf_main.model.conf_yaml)
    args_loss    = core.readconfyaml.read(conf_main.loss.conf_yaml)
    args_metric  = core.readconfyaml.read(conf_main.metric.conf_yaml)

    # dataset is more important than model
    train_dataset=core.dataset.DATASET[conf_main.dataset.cls_name](**args_dataset, split="train")
    valid_dataset=core.dataset.DATASET[conf_main.dataset.cls_name](**args_dataset, split="valid")

    # create corresponding model
    model = core.model.MODEL[conf_main.model.cls_name](**args_model)

    criterion = core.loss.LOSS[conf_main.loss.cls_name](**args_loss)

    metriclog = core.metric.METRIC[conf_main.metric.cls_name](
        **args_metric,
        log_dir=conf_main.log_alldir,
        log_exname=conf_main.exp_name
    )

    trainer = core.pipe.Trainer(
        device_mod="cuda" if torch.cuda.is_available() else "cpu",
        world_size=torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count(),
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_epochs=conf_main.num_epochs,
        batch_size=conf_main.batch_size,
        lr=conf_main.lr,
        log_exname=conf_main.exp_name,
        log_alldir=conf_main.log_alldir,
        log_interv=conf_main.log_interv
    )

    trainer.run()


if __name__ == "__main__":
    main()
