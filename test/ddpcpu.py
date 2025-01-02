import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

# 自定义一个简单的数据集
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index], torch.tensor([index % 2])  # 二分类标签

    def __len__(self):
        return self.len

# 自定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x)).squeeze()

# 初始化分布式训练环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

# 清理分布式训练环境
def cleanup():
    dist.destroy_process_group()

# 训练函数
def train(rank, world_size, epochs=2):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # 创建模型并包装为 DDP
    model = SimpleModel().cpu()
    model = DDP(model)

    # 创建数据集和数据加载器
    dataset = RandomDataset(size=10, length=100)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 设置 epoch 以便 shuffle 数据
        for inputs, labels in dataloader:
            inputs, labels = inputs.cpu(), labels.float().cpu()
            labels = labels.squeeze()  # 修正 labels 的形状
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

# 主函数
def run_demo(world_size):
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    world_size = 2  # 使用 2 个进程
    run_demo(world_size)
