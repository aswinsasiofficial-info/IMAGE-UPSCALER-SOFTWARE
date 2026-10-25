"""RRDBNet generator used by Real-ESRGAN x4plus (inference only).

Layer names follow the published checkpoint (``conv_first``,
``body.N.rdbM.convK``, ``conv_body``, ``conv_up1/2``, ``conv_hr``,
``conv_last``) so the official ``RealESRGAN_x4plus.pth`` loads with
``strict=True``. Only the 4x variant is implemented — it is the one the app
ships — and it always enlarges both axes by exactly 4.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _DenseBlock(nn.Module):
    """Five 3x3 convolutions, each fed every earlier feature map."""

    def __init__(self, channels: int, growth: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            features.append(self.act(conv(torch.cat(features, 1))))
        # Residual scaling of 0.2 is part of how the weights were trained.
        return x + 0.2 * self.conv5(torch.cat(features, 1))


class _ResidualInResidual(nn.Module):
    def __init__(self, channels: int, growth: int) -> None:
        super().__init__()
        self.rdb1 = _DenseBlock(channels, growth)
        self.rdb2 = _DenseBlock(channels, growth)
        self.rdb3 = _DenseBlock(channels, growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNetX4(nn.Module):
    def __init__(self, blocks: int = 23, channels: int = 64, growth: int = 32) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(3, channels, 3, 1, 1)
        self.body = nn.Sequential(*(_ResidualInResidual(channels, growth) for _ in range(blocks)))
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(channels, 3, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        # Two nearest-neighbour 2x steps in feature space; the convs learned the rest.
        for conv in (self.conv_up1, self.conv_up2):
            feat = self.act(conv(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.act(self.conv_hr(feat)))
