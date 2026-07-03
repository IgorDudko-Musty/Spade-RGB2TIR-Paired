import torch


def create_optimizer(params, cfg):

    name = cfg["name"].lower()
    opt_params = cfg.get("opt_params", {})

    if name == "adam":
        return torch.optim.Adam(params, **opt_params)
    if name == "adamw":
        return torch.optim.AdamW(params, **opt_params)
    if name == "sgd":
        return torch.optim.SGD(params, **opt_params)

    raise KeyError(f"Unknown optimizer: {name}")
