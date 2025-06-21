import utils.registry
METRIC = utils.registry.Registry("METRIC")

# 通过 import 执行代码解释来注册类
from .metric_base import BaseMetricLog
from .metric_deeplabv3 import DeepLabV3Metric
from .metric_squeezeseg import SqueezeSegMetric
from .metric_transunetmini import TransUNetMiniMetric
