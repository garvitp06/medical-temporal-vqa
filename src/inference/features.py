"""Build fixed-size VQA input tensors from processed exams.

The exam is cropped to the lung bounding box (+margin) and resized to
``grid`` (D,H,W). The crop box is returned so that VQA region tokens can be
mapped back to voxel indices and physical coordinates."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

GRID = (32, 48, 48)


def lung_crop_box(lung, margin=(2, 4, 4)):
    idx = np.argwhere(lung > 0)
    if idx.size == 0:
        return [0, 0, 0, *lung.shape]
    lo = np.maximum(idx.min(0) - margin, 0); hi = np.minimum(idx.max(0) + 1 + np.array(margin), lung.shape)
    return [int(v) for v in (*lo, *hi)]


def build_input(norm, lung, amap=None, box=None, grid=GRID, use_anomaly=True):
    box = box or lung_crop_box(lung)
    z0, y0, x0, z1, y1, x1 = box
    chans = [norm[z0:z1, y0:y1, x0:x1], lung[z0:z1, y0:y1, x0:x1].astype(np.float32)]
    if use_anomaly:
        a = np.zeros_like(norm) if amap is None else amap
        chans.append(np.clip(a[z0:z1, y0:y1, x0:x1] / 5.0, -1, 3))
    t = torch.from_numpy(np.stack(chans).astype(np.float32))[None]
    t = F.interpolate(t, size=grid, mode="trilinear", align_corners=False)[0]
    return t, box


def token_of_voxel(zyx, box, grid=GRID, stride=8):
    z0, y0, x0, z1, y1, x1 = box
    g = [s // stride for s in grid]
    rel = [(zyx[0] - z0) / max(1, z1 - z0), (zyx[1] - y0) / max(1, y1 - y0), (zyx[2] - x0) / max(1, x1 - x0)]
    cell = [int(np.clip(np.floor(r * n), 0, n - 1)) for r, n in zip(rel, g)]
    return (cell[0] * g[1] + cell[1]) * g[2] + cell[2]


def voxel_box_of_token(tok, box, grid=GRID, stride=8):
    z0, y0, x0, z1, y1, x1 = box
    g = [s // stride for s in grid]
    cz, rem = divmod(tok, g[1] * g[2]); cy, cx = divmod(rem, g[2])
    def rng(c, n, a, b):
        return a + (b - a) * c / n, a + (b - a) * (c + 1) / n
    zz, yy, xx = rng(cz, g[0], z0, z1), rng(cy, g[1], y0, y1), rng(cx, g[2], x0, x1)
    return [zz[0], yy[0], xx[0], zz[1], yy[1], xx[1]]
