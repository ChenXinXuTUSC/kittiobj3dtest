import utils.registry
METRIC = utils.registry.Registry("METRIC")

from .metric_base import BaseMetricLog
from .metric_deeplabv3 import DeepLabV3Metric
