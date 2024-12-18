import sys
import os
import os.path as osp

import torch
import torch.utils.data

import core

import argparse

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser("train.py")
args_group_conf = parser.add_argument_group("conf", "arguments related to conf read")
args_group_conf.add_argument("--yaml", type=str, help="path to yaml conf file")

def main():
    args = parser.parse_args()

    conf = core.readconfyaml.read(args.yaml)

    train_loader = torch.utils.data.DataLoader(
        core.dataset.KITTISpherical(
            "data", "train"
        ),
        batch_size=conf.dataloader.batch_size,
        shuffle=True,
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = core.model.squeezeseg.SqueezeSeg(conf.model)
    model.to(device)

    for iter, (fmap, gdth) in enumerate(train_loader):
        fmap = fmap.to(device)
        gdth = gdth.to(device)
        
        pred = model(fmap)
        break


if __name__ == "__main__":
    main()
