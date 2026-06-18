"""
Building blocks for U-Net architectures used in speech dereverberation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class DoubleConv2d(nn.Module):
    """Two successive (Conv2d -> BatchNorm -> PReLU) blocks."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None,
                 dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        pad = dilation if dilation > 1 else kernel_size // 2
        self.double_conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            weight_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, dilation=dilation)),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
            nn.ReflectionPad2d(kernel_size // 2),
            weight_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size)),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsample with MaxPool2d then DoubleConv2d."""

    def __init__(self, in_channels: int, out_channels: int, reduce_temporal: int = 2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2, reduce_temporal)),
            DoubleConv2d(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample then DoubleConv2d with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True,
                 reduce_temporal: int = 2):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=(2, reduce_temporal), mode="bilinear", align_corners=True)
            self.conv = DoubleConv2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv2d(nn.Module):
    """1x1 convolution with Tanh activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
