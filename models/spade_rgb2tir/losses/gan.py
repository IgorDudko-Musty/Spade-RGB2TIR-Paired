import torch
import torch.nn as nn

from .registry import reg_losses


@reg_losses("gan")
class GANLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.real_label = cfg["real_label"]
        self.fake_label = cfg["fake_label"]
        self.weight = cfg["weight"]

        self.loss = nn.MSELoss()

    def D_loss(self, D_real, D_fake):
        loss_real = self.loss(D_real, torch.full_like(D_real, self.real_label))
        loss_fake = self.loss(D_fake, torch.full_like(D_fake, self.fake_label))
        return 0.5 * (loss_real + loss_fake) * self.weight

    def G_loss(self, D_fake):
        return self.loss(D_fake, torch.full_like(D_fake, self.real_label)) * self.weight
