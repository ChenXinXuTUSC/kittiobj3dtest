import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def ddp_setup(device_mod, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    backend = "nccl" if device_mod == "cuda" else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()


def train(rank, device_mod, world_size, batch_size):
    # 初始化进程组
    ddp_setup(device_mod, rank, world_size)

    device = torch.device(rank if device_mod == "cuda" else "cpu")

    # 创建模型并包装为 DDP
    model = torch.nn.Linear(10, 10).to(device)
    model = DDP(model, device_ids=[rank] if device_mod == "cuda" else None)

    # 创建数据集和 DistributedSampler
    dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 训练逻辑
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            print(f"Rank {rank}, Batch {batch_idx}, Data Shape: {data.shape}")

    # 清理进程组
    ddp_cleanup()

def run_demo(device_mod, world_size, batch_size):
    mp.spawn(
        fn=train,
        args=(device_mod, world_size, batch_size),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # device_mod = "cpu"
    device_mod = "cuda"
    world_size = torch.cuda.device_count() if device_mod == "cuda" else os.cpu_count()
    batch_size = 32  # 每个进程的 batch_size
    print(f"device_mod: {device_mod}, world_size: {world_size}, batch_size: {batch_size}")
    run_demo(device_mod, world_size, batch_size)
