"""
Neural network architectures for the dereverberation pipeline.

The end-to-end model consists of two stages:
  1. Spec2Spec (UNetSpec2Spec): Magnitude spectrogram enhancement
  2. Spec2Wav  (UNetSpec2Wav):  Complex STFT (RI) refinement
"""

import torch
import torch.nn as nn

from .layers import DoubleConv2d, Down, Up, OutConv2d

EPSILON = 1e-6


class UNetSpec2Spec(nn.Module):
    """
    U-Net with Transformer bottleneck for magnitude spectrogram enhancement.

    Input:  batch[0] = magnitude spectrogram  (B, 1, F, T)
    Output: enhanced log-magnitude spectrogram (B, 1, F, T)
    """

    def __init__(self, bilinear: bool = True):
        super().__init__()
        self.inc = DoubleConv2d(1, 64)
        self.down0 = Down(64, 128, reduce_temporal=1)
        self.down1 = Down(128, 128, reduce_temporal=1)
        self.down2 = Down(128, 256, reduce_temporal=1)
        self.down3 = Down(256, 512, reduce_temporal=1)
        self.down4 = Down(512, 512, reduce_temporal=1)
        self.down5 = nn.Linear(4096, 512)

        self.encod1 = nn.TransformerEncoderLayer(
            d_model=512, nhead=16, batch_first=True, dropout=0.1, dim_feedforward=2048,
        )
        self.encod2 = nn.TransformerEncoderLayer(
            d_model=512, nhead=16, batch_first=True, dropout=0.1, dim_feedforward=2048,
        )

        self.up0 = nn.Linear(512, 4096)
        self.up1 = Up(1024, 256, bilinear, reduce_temporal=1)
        self.up2 = Up(512, 128, bilinear, reduce_temporal=1)
        self.up3 = Up(256, 128, bilinear, reduce_temporal=1)
        self.up4 = Up(256, 64, bilinear, reduce_temporal=1)
        self.up5 = Up(128, 32, bilinear, reduce_temporal=1)
        self.outc = OutConv2d(32, 1)

        self.mu = nn.Linear(512, 1)
        self.max = nn.Linear(512, 1)

    def forward(self, batch):
        x0 = self.inc(torch.log10(batch[0] + EPSILON))
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = x5.flatten(start_dim=1, end_dim=2).transpose(1, 2)
        x7 = self.down5(x6)

        x8 = self.encod1(x7)
        x9 = self.encod2(x8)

        x10 = self.up0(x9)
        x11 = x10.transpose(1, 2).view(-1, 512, 8, x10.shape[-2])

        x = self.up1(x11, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x0)
        logits = self.outc(x)

        emb = torch.mean(x8, dim=1)
        learned_mean = self.mu(emb).view(-1, 1, 1, 1)
        learned_max = self.max(emb).view(-1, 1, 1, 1)
        return logits * learned_max + learned_mean


class UNetSpec2Wav(nn.Module):
    """
    U-Net for complex STFT (Real-Imaginary) refinement.

    Input:  complex STFT as (B, 2, F, T)  — channel 0 = real, channel 1 = imag
    Output: refined complex STFT (B, 2, F, T)
    """

    def __init__(self, bilinear: bool = True):
        super().__init__()
        self.inc = DoubleConv2d(2, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv2d(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
