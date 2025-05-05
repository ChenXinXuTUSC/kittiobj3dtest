import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import random
import time

def train(rank, world_size, log_dir):
	"""
	Function to be run in each process. Simulates training and logs metrics.
	"""
	# Initialize the process group
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
	
	# Only rank 0 writes to TensorBoard
	writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None
	
	# Simulate a training loop
	num_epochs = 10
	for epoch in range(1, num_epochs + 1):
		# Simulate loss and accuracy values
		loss = random.uniform(0.1, 1.0) * (1 / epoch)  # Loss decreases over time
		accuracy = random.uniform(70.0, 90.0) + epoch  # Accuracy increases over time
		
		# Log only on rank 0
		if rank == 0:
			writer.add_scalar("Loss/train", loss, epoch)
			writer.add_scalar("Accuracy/train", accuracy, epoch)
			print(f"Rank {rank}, Epoch {epoch}: Logged loss={loss:.4f}, accuracy={accuracy:.2f}")
		
		# Simulate computation time
		time.sleep(0.5)
	
	# Close the writer in rank 0
	if rank == 0:
		writer.close()
	
	# Finalize the process group
	dist.destroy_process_group()

def main():
	"""
	Main function to spawn processes and start training.
	"""
	world_size = 2  # Number of processes
	log_dir = "./runs/mp_example"  # Log directory for TensorBoard
	
	# Set CUDA_VISIBLE_DEVICES to control GPU assignment
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use GPUs 0 and 1 (adjust as needed)
	
	# Spawn processes
	mp.spawn(train, args=(world_size, log_dir), nprocs=world_size, join=True)

if __name__ == "__main__":
	main()

	print("\nTraining complete! To visualize logs, run:")
	print("tensorboard --logdir=./runs/mp_example --port=6006")
