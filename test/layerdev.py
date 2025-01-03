import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# 封装模型类
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = torch.nn.Linear(10, 10)  # 10 个输入特征，10 个输出类别
        self.loss_fn = None  # 损失函数初始化为 None

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.layer(x)

# Trainer 类
class Trainer:
    def __init__(self, model, device_mod="cuda"):
        self.model = model
        self.device_mod = device_mod
        self.is_ddp = False

    def ddp_setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # Initialize the process group
        backend = "nccl" if self.device_mod == "cuda" else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        self.is_ddp = True

    def ddp_cleanup(self):
        dist.destroy_process_group()
        self.is_ddp = False

    def train(self, rank, world_size, batch_size):
        # Initialize process group
        self.ddp_setup(rank, world_size)

        # Set device
        device = torch.device(rank if self.device_mod == "cuda" else "cpu")
        self.model = self.model.to(device)

        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[rank] if self.device_mod == "cuda" else None)

        # Create dataset and DistributedSampler
        dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randint(0, 10, (100,)))  # 10 个类别
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        # Training logic
        for epoch in range(2):
            sampler.set_epoch(epoch)
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = self.model(data)

                # Compute loss
                if self.model.module.loss_fn is not None:
                    loss = self.model.module.loss_fn(output, target)
                    print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        # Cleanup process group
        self.ddp_cleanup()

    def train_ddp(self, world_size, batch_size):
        mp.spawn(
            fn=self.train,
            args=(world_size, batch_size),
            nprocs=world_size,
            join=True
        )

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = MyModel()

    # Set loss function with class weights
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 10 个类别的权重
    model.set_loss_fn(torch.nn.CrossEntropyLoss(weight=class_weights))

    # Set device mode (cuda or cpu)
    # device_mod = "cuda" if torch.cuda.is_available() else "cpu"
    device_mod = "cpu"
    world_size = torch.cuda.device_count() if device_mod == "cuda" else os.cpu_count()
    batch_size = 32  # Batch size per process

    # Create Trainer instance and run distributed training
    trainer = Trainer(model, device_mod)
    print(f"device_mod: {device_mod}, world_size: {world_size}, batch_size: {batch_size}")
    trainer.train_ddp(world_size, batch_size)
