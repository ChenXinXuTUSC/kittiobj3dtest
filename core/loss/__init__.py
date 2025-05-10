import utils.registry
LOSS = utils.registry.Registry("LOSS")

from .DeepLabV3Loss import DeepLabV3Loss
