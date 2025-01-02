#!/bin/bash

python train.py \
--conf_datad_yaml conf/data/kittiobj3d.yaml \
--conf_model_yaml conf/model/model.yaml \
--conf_train_yaml conf/pipe/deeplab+kittiobj3d.yaml \
