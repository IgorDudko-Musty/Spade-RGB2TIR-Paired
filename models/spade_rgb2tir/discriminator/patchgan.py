import torch.nn as nn

from ..blocks.downsample import DownsampleBlock
from ..blocks.downsample_skip import DownsampleSkipBlock
from ..blocks.spade_skip import SpadeResSkipBlock
from ..blocks.upsample import UpsampleBlock
from ..blocks.upsample_skip import UpsampleSkipBlock
from .registry import reg_discr


@reg_discr("patchgan")
class PatchGAN(nn.Module):
    BLOCK_REGISTRY = {
        "down": DownsampleBlock,
        "down_skip": DownsampleSkipBlock,
        "spade_skip": SpadeResSkipBlock,
        "up": UpsampleBlock,
        "up_skip": UpsampleSkipBlock,
    }

    def __init__(self, cfg):
        super().__init__()

        self.discriminator = nn.ModuleList()
        for block in cfg:
            block = self.BLOCK_REGISTRY[block["type"]](
                in_channels=block["in_ch"],
                out_channels=block["out_ch"],
                cfg=block["cfg"],
            )

            self.discriminator.append(block)

    def forward(self, x):
        feats = []
        last_idx = len(self.discriminator)

        for idx, block in enumerate(self.discriminator):
            x = block(x)
            if idx + 1 < last_idx:
                feats.append(x)
        prob_map = x
        return prob_map, feats
