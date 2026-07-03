import torch.nn as nn

from .registry import reg_losses


@reg_losses("l1")
class L1Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.weight = cfg["weight"]
        self.l1_loss = nn.L1Loss()

    def forward(self, G_tir, tir):
        return self.l1_loss(G_tir, tir) * self.weight
