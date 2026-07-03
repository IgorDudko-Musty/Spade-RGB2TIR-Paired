import torch.nn as nn

from ..layers.attention import AttentionLayer
from ..layers.conv_norm_act import ConvNormActLayer


class SpadeResSkipBlock(nn.Module):
    def __init__(self, in_channels, cmap_ch, cfg):
        super().__init__()

        self.norm_2d = nn.InstanceNorm2d(in_channels, affine=False)
        layers = []
        for out_channel in cfg.get("cmap_chs_out", []):
            layers += [
                ConvNormActLayer(
                    in_channels=cmap_ch,
                    out_channels=out_channel,
                    cfg=cfg.get("condmap_cfg", {}),
                ),
            ]
            cmap_ch = out_channel
        self.cond_map = nn.Sequential(*layers)

        self.gamma_beta = ConvNormActLayer(
            in_channels=cmap_ch,
            out_channels=in_channels * 2,
            cfg=cfg.get("gamma_beta_cfg", {}),
        )

        atten_cfg = cfg.get("attention_cfg", {})
        if atten_cfg.get("enable", False):
            self.atten = AttentionLayer(in_channels=in_channels, cfg=atten_cfg)
        else:
            self.atten = None

    def forward(self, x, condmap):
        x_norm = self.norm_2d(x)
        condmap = self.cond_map(condmap)

        gamma_beta = self.gamma_beta(condmap)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        out = x_norm * (1 + gamma) + beta + x
        if self.atten is not None:
            out = self.atten(out)

        return out
