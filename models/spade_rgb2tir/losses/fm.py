import torch.nn as nn

from .registry import reg_losses


@reg_losses("fm")
class FeatureMatchLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.weight = cfg["weight"]
        self.f1_loss = nn.L1Loss()

    def forward(self, feats_real, feats_fake):
        fm_loss = 0
        for feat_real, feat_fake in zip(feats_real, feats_fake):
            fm_loss += self.f1_loss(feat_real, feat_fake)
        return fm_loss / len(feats_real) * self.weight
