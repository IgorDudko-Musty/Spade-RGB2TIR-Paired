import torch
import torch.nn as nn


class CBAMLayer(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.ca = ChannelAttention(in_channels=in_channels, cfg=cfg)
        self.sa = SpatialAttention(cfg=cfg)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()

        reduction = cfg.get("reduction", 16)
        self.pool_types = cfg.get("pool_types", ("avg", "max"))

        hidden = max(8, in_channels // reduction)

        if "avg" in self.pool_types:
            self.pool_avg = nn.AdaptiveAvgPool2d(1)
        if "max" in self.pool_types:
            self.pool_max = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden,
                out_channels=in_channels,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x):
        out = 0
        if "avg" in self.pool_types:
            out = out + self.mlp(self.pool_avg(x))
        if "max" in self.pool_types:
            out = out + self.mlp(self.pool_max(x))
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        kernel_size = cfg.get("kernel_size", 7)

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="zeros",
            bias=False,
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_concat = torch.cat([x_avg, x_max], dim=1)
        out = self.conv(x_concat)
        return torch.sigmoid(out)
