import utils.registry
LOSS = utils.registry.Registry("LOSS")

from .DeepLabV3_KITTIObj3d_Loss import DeepLabV3_KITTIObj3d_Loss
from .DeepLabV3_KITTISemantic_Loss import DeepLabV3_KITTISemantic_Loss
