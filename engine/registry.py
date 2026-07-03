import importlib

MODEL_REGISTRY = {"spade_rgb2tir"}


def create_trainer(model_name):
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}")
    module = importlib.import_module(f"engine.{model_name}.trainer")
    return module.Trainer
