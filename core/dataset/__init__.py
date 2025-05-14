import utils.registry

# 定义一个全局的哈希表来存储模型类
DATASET = utils.registry.Registry("DATASET")


from .dataset_base import BaseDataset
from .dataset_kittiobj3d import KITTIObj3d
from .dataset_kittisem import KITTISemantic
