from torch.utils.data import DataLoader, TensorDataset
import torch

# 创建示例数据集
data = torch.arange(10)
dataset = TensorDataset(data)

# 创建 DataLoader，设置 shuffle=True
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 遍历 DataLoader
for epoch in range(3):
    print(f"Epoch {epoch+1}:")
    for batch in dataloader:
        print(batch)
    print()
