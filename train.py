import os
import os.path as osp
os.environ["TORCH_HOME"] = "."

import argparse


import torch

import core

parser = argparse.ArgumentParser("train.py")
parser.add_argument("--conf", type=str, help="main config file path")


def main():
	args = parser.parse_args()
	conf_main = core.conf.read(args.conf)
	
	args_dataset = core.conf.read(conf_main.dataset.conf)
	args_model   = core.conf.read(conf_main.model.conf)
	args_loss	 = core.conf.read(conf_main.loss.conf)
	args_metric  = core.conf.read(conf_main.metric.conf)

	# dataset is more important than model
	train_dataset = core.dataset.DATASET[conf_main.dataset.name](**args_dataset, split="train")
	valid_dataset = core.dataset.DATASET[conf_main.dataset.name](**args_dataset, split="valid")

	# create corresponding model
	model = core.model.MODEL[conf_main.model.name](**args_model)

	criterion = core.loss.LOSS[conf_main.loss.name](**args_loss)

	metriclog = core.metric.METRIC[conf_main.metric.name](**args_metric)

	trainer = core.pipe.Trainer(
		device_mod="cuda" if torch.cuda.is_available() else "cpu",
		world_size=torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count(),
		model=model,
		train_dataset=train_dataset,
		valid_dataset=valid_dataset,
		criterion=criterion,
		metriclog=metriclog,
		num_epochs=conf_main.train.num_epochs,
		batch_size=conf_main.train.batch_size,
		lr=conf_main.train.lr,
		log_exname=conf_main.exp_name,
		log_alldir=conf_main.log.log_alldir,
		log_interv=conf_main.log.log_interv
	)

	trainer.run()


if __name__ == "__main__":
	main()
