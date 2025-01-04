# 定义一个全局的哈希表来存储模型类
import utils.registry
MODEL = utils.registry.Registry("MODEL")

from .squeezeseg import SqueezeSeg
from .deeplabv3 import DeepLabV3
