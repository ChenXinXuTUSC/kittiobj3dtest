# 定义一个全局的哈希表来存储模型类
DATASET = {}

# 定义装饰器
def register_dataset(cls):
    # 将类名作为键，类本身作为值注册到哈希表中
    DATASET[cls.__name__] = cls
    return cls # return original type after registration

from .kitti import *
