import sys
import os
import os.path as osp

import torch
import torch.utils.data

import core

def main():
    train_loader = torch.utils.data.DataLoader(
        core.dataset.KITTISpherical(
            "data", "train"
        )
    )


if __name__ == "__main__":
    main()
