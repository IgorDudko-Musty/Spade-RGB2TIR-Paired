import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()

        # ch_reduction = cfg.get("reduction", 8)
        d_k = in_channels // cfg.get("reduction", 8)
        self.querry = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_k,
            kernel_size=1,
        )
        self.key = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_k,
            kernel_size=1,
        )
        self.value = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.drop = nn.Dropout(cfg.get("drop_prob", 0.1))

        self.scale = 1 / (d_k**0.5)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        querry = self.querry(x).view(B, -1, N)
        key = self.key(x).view(B, -1, N)
        value = self.value(x).view(B, C, N)

        querry = querry.transpose(1, 2)
        key = key
        value = value.transpose(1, 2)

        attn = F.softmax(torch.bmm(querry, key) * self.scale, dim=-1)
        attn = self.drop(attn)
        attn = torch.bmm(attn, value)

        attn = attn.transpose(1, 2).view(B, C, H, W)
        return x + self.gamma * attn
