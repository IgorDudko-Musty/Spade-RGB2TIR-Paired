import torch.nn as nn

from ..layers.cbam import CBAMLayer
from ..layers.conv_norm_act import ConvNormActLayer


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
    ):
        super().__init__()

        self.conv = ConvNormActLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            cfg=cfg.get("down_cfg", {}),
        )

        cbam_cfg = cfg.get("cbam_cfg", {})
        if cbam_cfg.get("enable", False):
            self.cbam = CBAMLayer(in_channels=out_channels, cfg=cbam_cfg)
        else:
            self.cbam = None

    def forward(self, x):
        out = self.conv(x)
        if self.cbam is not None:
            out = self.cbam(out)
        return out
