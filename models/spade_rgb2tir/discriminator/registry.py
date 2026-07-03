DISCRIMINATOR_REGISTRY = {}


def reg_discr(type_d):
    def wrapper(cls):
        DISCRIMINATOR_REGISTRY[type_d] = cls
        return cls

    return wrapper
