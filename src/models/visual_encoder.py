"""Spatially-preserving 3D visual encoder for VQA (spec §7.2).

Input: (B, C, D, H, W) exam tensor resampled to a fixed grid (default
32x48x48) with channels
    0: lung-windowed normalised CT
    1: lung mask
    2: (optional) normalised anomaly map from the flow detector
Output: (B, N_tokens, d) token sequence, one token per cell of the final
feature grid (default 4x6x6 = 144), plus learned 3D positional embeddings,
so localisation information is preserved (no global pooling).

Pretrained weights: no general-purpose pretrained 3D CT encoder could be
downloaded in the build environment (Hugging Face unreachable), so this
encoder is trained from scratch on the available (synthetic) data. The
pretrained medical model actually used in this project is the lungmask
U-Net R231 for lung segmentation (see preprocessing).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(cin, cout, stride):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(4, cout), nn.SiLU(),
                         nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(4, cout), nn.SiLU())


class VolumeEncoder(nn.Module):
    def __init__(self, in_channels=3, width=32, d_model=128, grid=(32, 48, 48)):
        super().__init__()
        self.net = nn.Sequential(conv_block(in_channels, width, 1),
                                 conv_block(width, width * 2, 2),
                                 conv_block(width * 2, width * 2, 2),
                                 conv_block(width * 2, width * 4, 2))
        self.proj = nn.Conv3d(width * 4, d_model, 1)
        g = tuple(s // 8 for s in grid)
        self.token_grid = g
        self.pos = nn.Parameter(torch.zeros(1, g[0] * g[1] * g[2], d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        f = self.proj(self.net(x))                  # (B,d,D/8,H/8,W/8)
        tokens = f.flatten(2).transpose(1, 2)       # (B,N,d)
        return tokens + self.pos
