LOSS_REGISTRY = {}


def reg_losses(type_l):
    def wrapper(cls):
        LOSS_REGISTRY[type_l] = cls
        return cls

    return wrapper
