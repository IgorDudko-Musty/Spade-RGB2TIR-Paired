import torch.nn as nn
import torch.nn.functional as F

from .patchgan import PatchGAN
from .registry import reg_discr


@reg_discr("ms_patchgan")
class MultiScalePatchGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        amount = cfg.get("amount", 1)

        self.discriminators = nn.ModuleList(
            [PatchGAN(cfg.get("blocks", {})) for _ in range(amount)]
        )

    def forward(self, x):
        preds = []
        feats = []

        for D in self.discriminators:
            prob, feat = D(x)
            preds.append(prob)
            feats.append(feat)

            x = F.avg_pool2d(x, 2, 2)
        return preds, feats
