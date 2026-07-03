from .discriminator.registry import DISCRIMINATOR_REGISTRY
from .generator.registry import GENERATOR_REGISTRY
from .losses.registry import LOSS_REGISTRY

__all__ = ["GENERATOR_REGISTRY", "DISCRIMINATOR_REGISTRY", "LOSS_REGISTRY"]


def create_generator(cfg_G):
    name = cfg_G["name"]
    if name not in GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator: {name}")
    return GENERATOR_REGISTRY[name](cfg_G)


def create_discriminator(cfg_D):
    name = cfg_D["name"]
    if name not in DISCRIMINATOR_REGISTRY:
        raise KeyError(f"Unknown discriminator: {name}")
    return DISCRIMINATOR_REGISTRY[name](cfg_D)


def create_loss(cfg_losses):
    name = cfg_losses["name"]
    if name not in LOSS_REGISTRY:
        raise KeyError(f"Unknown loss: {name}")
    return LOSS_REGISTRY[name](cfg_losses)


def create_model(cfg):
    generator = create_generator(cfg["model"]["generator"])
    discriminator = create_discriminator(cfg["model"]["discriminator"])

    losses = {
        loss_key: create_loss(loss_cfg)
        for loss_key, loss_cfg in cfg["model"]["losses"].items()
    }
    return generator, discriminator, losses
