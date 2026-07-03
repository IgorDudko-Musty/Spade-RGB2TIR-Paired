import os

from configs.spade_rgb2tir.config_factory import build_config
from engine.registry import create_trainer

os.environ["WANDB_BASE_URL"] = "http://127.0.0.1:5000"  # "http://192.168.65.99:5000"
os.environ["WANDB_API_KEY"] = "local-0a3cc2c7c227930d5d3e69ca34b74abb0e5366dd"
os.environ["WANDB_MODE"] = "online"


def main(model_name):
    cfg = build_config(model_name)

    trainer = create_trainer(model_name)(cfg)

    trainer.fit()


if __name__ == "__main__":

    main("spade_rgb2tir")
