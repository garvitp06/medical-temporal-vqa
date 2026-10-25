"""Feature-space flow anomaly detection using the PRETRAINED lungmask U-Net
R231 encoder as a frozen CT feature extractor (FastFlow/CFLOW-style).

Why: a GLOW trained directly on pixels ranks homogeneous dense lesions only
weakly above vessels (see reports/RESULTS.md, pixel GLOW row) — the well-known
tendency of pixel-space flows to reward low-complexity regions. Modelling the
density of pretrained features is the standard remedy in flow-based anomaly
detection.

Pipeline per axial slice (2.5 mm grid):
  HU -> clip[-1024, 600], (HU+1024)/1624 (lungmask's own input normalisation)
     -> bilinear x2 upsample (~1.25 mm/px, close to lungmask's training scale)
     -> down_path[0] -> avgpool -> down_path[1] (128 ch, 2.5 mm = input grid)
                     -> avgpool -> down_path[2] (256 ch, 5 mm) -> x2 upsample
     -> concat (384 ch) -> fixed PCA projection (fitted on TRAIN normal lung
        features) -> 64 ch -> single-scale GLOW (no squeeze) -> per-position NLL.
The extractor and PCA are frozen; only the flow is trained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LungmaskFeatures(nn.Module):
    def __init__(self, weights: str, n_components=64):
        super().__init__()
        from lungmask.mask import get_model
        self.unet = get_model("unet", weights if Path(weights).exists() else None).eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.register_buffer("pca_mean", torch.zeros(384))
        self.register_buffer("pca_proj", torch.eye(384)[:, :n_components])
        self.n_components = n_components
        self.fitted = False

    @torch.no_grad()
    def raw(self, hu_slices: torch.Tensor) -> torch.Tensor:
        """hu_slices: (B,H,W) HU -> (B,384,H,W)."""
        x = (hu_slices.clamp(-1024, 600) + 1024) / 1624.0
        x = F.interpolate(x[:, None], scale_factor=2, mode="bilinear", align_corners=False)
        d = self.unet.down_path
        f0 = d[0](x)
        f1 = d[1](F.avg_pool2d(f0, 2))                      # (B,128,H,W)
        f2 = d[2](F.avg_pool2d(f1, 2))                      # (B,256,H/2,W/2)
        f2 = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f1, f2], 1)

    @torch.no_grad()
    def fit_pca(self, feats: torch.Tensor):
        """feats: (N,384) feature vectors from TRAIN normal lung positions."""
        mu = feats.mean(0)
        U, S, Vh = torch.linalg.svd(feats - mu, full_matrices=False)
        self.pca_mean.copy_(mu)
        self.pca_proj.copy_(Vh[: self.n_components].T)
        self.fitted = True
        return float((S[: self.n_components] ** 2).sum() / (S ** 2).sum())

    @torch.no_grad()
    def forward(self, hu_slices):
        f = self.raw(hu_slices)
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, c)
        z = (f - self.pca_mean) @ self.pca_proj
        return z.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


def extract_volume_features(fx: LungmaskFeatures, hu: np.ndarray, z_list, batch=8):
    out = {}
    for i in range(0, len(z_list), batch):
        zz = z_list[i:i + batch]
        f = fx(torch.from_numpy(hu[zz].astype(np.float32)))
        for j, z in enumerate(zz):
            out[z] = f[j]
    return out
