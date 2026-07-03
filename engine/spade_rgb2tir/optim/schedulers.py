import torch


def build_linear_scheduler(optimizer, cfg):
    warmup = cfg["params"].get("warmup", 0)
    total_epochs = cfg["params"].get("total_epochs", 3000)

    def lr_lambda(epoch):
        # warmup
        if epoch + 1 <= warmup:
            return (epoch + 1) / (warmup + 1)

        # linear decay
        return 1.0 - max(0, epoch + 1 - warmup) / (total_epochs - warmup)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def create_scheduler(optimizer, cfg):

    name = cfg["name"].lower()
    kwargs = cfg.get("params", {})

    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, **kwargs)

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    if name == "custom":
        return build_linear_scheduler(optimizer, cfg)

    raise KeyError(f"Unknown scheduler: {name}")
