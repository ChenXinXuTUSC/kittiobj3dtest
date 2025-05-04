import utils.registry

# 定义一个全局的哈希表来存储模型类
DATASET = utils.registry.Registry("DATASET")


from .base import BaseDataset
from .kittiobj3d import KITTIObj3d
from .kittisem import KITTISemantic
