import torch
import torch.nn as nn

from ..layers.attention import AttentionLayer


class ChannelGMHA(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()

        self.head_dim = in_channels // cfg.get("num_heads", 4)

        self.heads = nn.ModuleList(
            [
                AttentionLayer(in_channels=self.head_dim, cfg=cfg)
                for _ in range(cfg.get("num_heads", 4))
            ]
        )

    def forward(self, x):

        x_splitted = torch.split(x, self.head_dim, dim=1)

        out = [head(x_sp) for head, x_sp in zip(self.heads, x_splitted)]
        out = torch.cat(out, dim=1)

        return out + x
