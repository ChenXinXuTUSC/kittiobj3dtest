import utils.registry
LOSS = utils.registry.Registry("LOSS")

from .loss_deeplabv3 import DeepLabV3Loss
from .loss_squeezeseg import SqueezeSegLoss
