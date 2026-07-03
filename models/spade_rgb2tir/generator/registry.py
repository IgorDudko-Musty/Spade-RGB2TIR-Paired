GENERATOR_REGISTRY = {}


def reg_gen(type_g):
    def wrapper(cls):
        GENERATOR_REGISTRY[type_g] = cls
        return cls

    return wrapper
