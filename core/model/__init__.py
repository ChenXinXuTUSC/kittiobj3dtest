from .squeezeseg import *

# 定义一个全局的哈希表来存储模型类
model_registry = {}

# 定义装饰器
def register_model(cls):
    # 将类名作为键，类本身作为值注册到哈希表中
    model_registry[cls.__name__] = cls
    return cls
