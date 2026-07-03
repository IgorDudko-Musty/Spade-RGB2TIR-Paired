import torch.nn as nn

from ..blocks.downsample import DownsampleBlock
from ..blocks.downsample_skip import DownsampleSkipBlock
from ..blocks.spade_skip import SpadeResSkipBlock
from ..blocks.upsample import UpsampleBlock
from ..blocks.upsample_skip import UpsampleSkipBlock
from .registry import reg_gen


@reg_gen("spade")
class Generator(nn.Module):
    BLOCK_REGISTRY = {
        "down": DownsampleBlock,
        "down_skip": DownsampleSkipBlock,
        "spade_skip": SpadeResSkipBlock,
        "up": UpsampleBlock,
        "up_skip": UpsampleSkipBlock,
    }

    def __init__(self, cfg):
        super().__init__()

        blocks = []
        for block in cfg.get("down_blocks", []):
            blocks.append(
                self.BLOCK_REGISTRY[block["type"]](
                    in_channels=block["in_ch"],
                    out_channels=block["out_ch"],
                    cfg=block["cfg"],
                )
            )
        self.encoder = nn.Sequential(*blocks)

        blocks = []
        for block in cfg.get("spade_blocks", []):
            blocks.append(
                self.BLOCK_REGISTRY[block.get("type", "spade_skip")](
                    in_channels=block["in_ch"],
                    cmap_ch=block["cmap_ch"],
                    cfg=block["cfg"],
                )
            )
        self.spade_bottleneck = nn.Sequential(*blocks)

        blocks = []
        for block in cfg.get("up_blocks", []):
            blocks.append(
                self.BLOCK_REGISTRY[block["type"]](
                    in_channels=block["in_ch"],
                    out_channels=block["out_ch"],
                    cfg=block["cfg"],
                )
            )
        self.decoder = nn.Sequential(*blocks)

    def forward(self, x):
        latent, feats = self._encode(x)
        latent = self._spade(latent, x)
        out = self._decode(latent, feats)
        return out

    def _encode(self, x):
        feats = []
        latent = x
        for block in self.encoder:
            latent = block(latent)
            if isinstance(latent, tuple):
                latent, skip = latent
                feats.append(skip)
        return latent, feats

    def _spade(self, latent, x):
        for block in self.spade_bottleneck:
            latent = block(latent, x)
        return latent

    def _decode(self, latent, feats):
        out = latent
        for layer in self.decoder:
            if isinstance(layer, UpsampleSkipBlock):
                out = layer(out, feats.pop(-1))
            else:
                out = layer(out)

        return out
