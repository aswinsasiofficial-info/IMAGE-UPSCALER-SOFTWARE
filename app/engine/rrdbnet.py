"""Self-contained RRDBNet architecture used by the Real-ESRGAN x4plus model.

This is a faithful re-implementation of the network defined in
``basicsr/archs/rrdbnet_arch.py`` (the architecture that the official
``RealESRGAN_x4plus.pth`` checkpoint was trained with), written using only
public PyTorch APIs so we do NOT depend on ``basicsr`` / the ``realesrgan``
package.

Why not just ``pip install realesrgan``?  That package imports
``torchvision.transforms.functional_tensor`` (removed in modern torchvision) and
pins very old dependencies, so it breaks on recent torch/torchvision. Vendoring
the ~150 lines of network code here is far more robust and gives identical
output when paired with the official weights.

State-dict keys produced by this module line up 1:1 with the official
checkpoint, so ``load_state_dict`` works with zero key remapping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _make_layer(basic_block: type[nn.Module], num_basic_block: int, **kwarg) -> nn.Sequential:
    """Stack ``num_basic_block`` copies of ``basic_block`` (basicsr's make_layer)."""
    return nn.Sequential(*[basic_block(**kwarg) for _ in range(num_basic_block)])


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) with 5 conv layers and dense connections.

    Args:
        num_feat: Number of channels of the input / output feature maps.
        num_grow_ch: Growth channels for the dense connection.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperical scaling used during training to stabilise residuals.
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block: three RDBs with a residual shortcut."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet used by Real-ESRGAN (``scale=4``) — the x4plus generator.

    Defaults match the official ``RealESRGAN_x4plus.pth`` configuration:
    ``num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32``.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # Upsampling (two 2x nearest-neighbour stages => 4x total for scale=4).
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = torch.cat((x, F.pixel_unshuffle(x, downscale_factor=2)), dim=1)
        elif self.scale == 1:
            feat = torch.cat((x, F.pixel_unshuffle(x, downscale_factor=4)), dim=1)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
