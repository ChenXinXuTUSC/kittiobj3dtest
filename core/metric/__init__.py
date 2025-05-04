import utils.registry
METRIC = utils.registry.Registry("METRIC")

from .base import BaseMetricLog
from .deeplabv3 import DeepLabV3Metric
