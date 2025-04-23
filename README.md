# Unified DDP Training Framework

This is a simple training framework utilizing pytorch DDP, automatically adapt to single GPU or multi GPU on a single machine.

# Module

The framework devides the training and testing pipeline into 4 modules and 1 entry point in the `core` dir as described below:

1. `conf` stores arguments for initializing the model and dataset that will be used in the following training pipeline.
2. `dataset` stores the implementation of loading datasets to pytorch `Dataset` , normally contains the logic of data

# Dataset

## KITTI Object 3D

## SemanticKITTI

SemanticKITTI only provides sementic lable for each frame in KITTI pointcloud(velodyne) dataset, should be used along with KITTI dataset sequences.
