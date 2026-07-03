import torch.nn as nn

from ..layers.cbam import CBAMLayer
from ..layers.conv_norm_act import ConvNormActLayer


class UpsampleSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()

        if cfg.get("up_scale", 2) != 1:
            self.upsample = nn.Upsample(
                scale_factor=cfg.get("up_scale", 2), mode="nearest"
            )
            self.up_conv = ConvNormActLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                cfg=cfg.get("up_conv_cfg", {}),
            )
        else:
            self.upsample = nn.Identity()
            self.up_conv = ConvNormActLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                cfg=cfg.get("up_conv_cfg", {}),
            )
        cbam_cfg = cfg.get("cbam_cfg", {})
        if cbam_cfg.get("enable", False):
            self.cbam = CBAMLayer(in_channels=out_channels, cfg=cbam_cfg)
        else:
            self.cbam = None

    def forward(self, x, skip):
        x = x + skip
        out = self.upsample(x)
        # out = out + skip
        out = self.up_conv(out)
        if self.cbam is not None:
            out = self.cbam(out)
        return out
