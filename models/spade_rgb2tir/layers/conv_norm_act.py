import torch.nn as nn


class ConvNormActLayer(nn.Module):
    NORM_REGISTRY = {
        "in": lambda channels: nn.InstanceNorm2d(channels),
        "bn": lambda channels: nn.BatchNorm2d(channels),
        "gn": lambda channels: nn.GroupNorm(32, channels),
    }
    ACT_REGISTRY = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
    ):
        super().__init__()
        conv = cfg.get("conv", {})
        norm = cfg.get("norm", {})
        act = cfg.get("act", {})

        block = [
            self._make_conv(
                in_channels=in_channels, out_channels=out_channels, cfg=conv
            )
        ]

        norm_type = norm.get("type", "ident")
        if norm_type != "ident":
            block.append(self.NORM_REGISTRY[norm_type](out_channels))

        act_type = act.get("type", "ident")
        if act_type != "ident":
            act_params = {k: v for k, v in act.items() if k != "type"}
            block.append(self.ACT_REGISTRY[act_type](**act_params))

        self.block = nn.Sequential(*block)

    def _make_conv(self, in_channels, out_channels, cfg):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=cfg.get("kernel_size", 3),
            stride=cfg.get("stride", 1),
            padding=cfg.get("padding", 1),
            padding_mode=cfg.get("padding_mode", "reflect"),
            bias=cfg.get("bias", False),
        )

    def forward(self, x):
        return self.block(x)
