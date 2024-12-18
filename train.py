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
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = core.model.squeezeseg.SqueezeSeg(conf.model)

    for iter, (fmap, gdth) in enumerate(train_loader):
        pred = model(fmap)
        break


if __name__ == "__main__":
    main()
