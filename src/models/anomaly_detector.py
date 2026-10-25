"""Unified anomaly-detection interface (spec §6.5).

All detectors map a preprocessed exam to a voxel-wise anomaly map on the
exam's processed grid, restricted to the lung mask.

Score normalisation (documented procedure): raw map values inside the lung
are standardised with the mean/std of raw lung-voxel scores measured on the
VALIDATION split's normal exams (``calibrate``). A normalised value of 3 means
"3 standard deviations above typical normal lung tissue for this model".
These are anomaly scores, not probabilities and not clinical likelihoods.

Volume-level score: mean of the top 0.1% normalised lung-voxel scores.
Slice-level score: max normalised score in each axial slice's lung region.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage as ndi

from .glow import Glow, latent_anomaly_map
from .contextflow import ContextFlowPP


def stack_25d(norm: np.ndarray, k: int, ctx: int = 1) -> np.ndarray:
    """Channels = slices k-ctx .. k+ctx (edge-replicated)."""
    idx = np.clip(np.arange(k - ctx, k + ctx + 1), 0, norm.shape[0] - 1)
    return norm[idx]


def pad_to_multiple(x: np.ndarray, m: int, value: float):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    pads = [(0, 0)] * (x.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(x, pads, constant_values=value), (h, w)


@dataclass
class AnomalyOutput:
    raw_map: np.ndarray            # (z,y,x) raw score (model-specific units)
    map: np.ndarray                # normalised score (z-units vs normal lung)
    slice_scores: np.ndarray       # (z,)
    volume_score: float
    method: str


class BaseDetector:
    name = "base"
    norm_mean: float = 0.0
    norm_std: float = 1.0

    def raw_map(self, norm: np.ndarray, lung: np.ndarray, meta: dict | None = None) -> np.ndarray:
        raise NotImplementedError

    def calibrate(self, exams):
        """exams: iterable of (normalized, lung_mask, meta) for validation NORMAL exams."""
        vals = []
        for norm, lung, meta in exams:
            r = self.raw_map(norm, lung, meta)
            vals.append(r[lung > 0])
        v = np.concatenate(vals)
        self.norm_mean, self.norm_std = float(v.mean()), float(v.std() + 1e-8)
        return dict(mean=self.norm_mean, std=self.norm_std, n_voxels=int(v.size))

    def score(self, norm, lung, meta=None) -> AnomalyOutput:
        raw = self.raw_map(norm, lung, meta).astype(np.float32)
        nm = (raw - self.norm_mean) / self.norm_std
        nm = np.where(lung > 0, nm, 0.0).astype(np.float32)
        inside = nm[lung > 0]
        k = max(1, int(0.001 * inside.size)) if inside.size else 1
        vol_score = float(np.sort(inside)[-k:].mean()) if inside.size else float("nan")
        sl = np.array([nm[z][lung[z] > 0].max() if lung[z].any() else 0.0 for z in range(nm.shape[0])])
        return AnomalyOutput(raw, nm, sl, vol_score, self.name)

    def state(self):
        return dict(name=self.name, norm_mean=self.norm_mean, norm_std=self.norm_std)


class IntensityBaseline(BaseDetector):
    """Simple intensity baseline: local excess density above the lung
    parenchyma. raw = gaussian_smooth(HU-like normalised value) inside lung
    minus the lung median. Dense blobs (nodules) and vessels score high —
    which is exactly why a learned model is needed."""
    name = "intensity_baseline"

    def __init__(self, sigma_vox: float = 1.0):
        self.sigma = sigma_vox

    def raw_map(self, norm, lung, meta=None):
        sm = ndi.gaussian_filter(norm.astype(np.float32), self.sigma)
        med = float(np.median(norm[lung > 0])) if lung.any() else 0.0
        return sm - med


class FlowDetector(BaseDetector):
    """GLOW or ContextFlow++-style detector on 2.5D axial slices.

    The model is applied fully-convolutionally to whole slices (padded to a
    multiple of 2**levels); the per-scale latent log-density maps give a 2D map
    per slice, and slices are stacked into a 3D map. This is 2.5D reasoning
    (3 neighbouring slices of context), NOT true 3D volumetric modelling.
    """

    def __init__(self, model, name="glow", ctx_slices=1, device="cpu", batch=16, smooth_sigma=0.7):
        self.model = model.to(device).eval()
        self.name, self.ctx, self.device, self.batch = name, ctx_slices, device, batch
        self.smooth = smooth_sigma

    @torch.no_grad()
    def raw_map(self, norm, lung, meta=None):
        meta = meta or {}
        nz = norm.shape[0]
        zs = [z for z in range(nz) if lung[z].any()]
        out = np.zeros(norm.shape, np.float32)
        m = 2 ** (self.model.g.levels if isinstance(self.model, ContextFlowPP) else self.model.levels)
        for i in range(0, len(zs), self.batch):
            chunk = zs[i:i + self.batch]
            xs = []
            for z in chunk:
                x, (h, w) = pad_to_multiple(stack_25d(norm, z, self.ctx), m, value=-1.0)
                xs.append(x)
            x = torch.from_numpy(np.stack(xs)).float().to(self.device)
            if isinstance(self.model, ContextFlowPP):
                cont, disc = slice_context(chunk, lung, meta)
                _, o = self.model(x, cont.to(self.device), disc.to(self.device), return_maps=True)
            else:
                _, o = self.model(x, return_maps=True)
            amap = latent_anomaly_map(o, x.shape[-2:]).cpu().numpy()[:, :h, :w]
            for j, z in enumerate(chunk):
                out[z] = amap[j]
        if self.smooth:
            out = ndi.gaussian_filter(out, self.smooth)
        return out


THICKNESS_BINS = [2.75, 3.25]      # -> categories {<=2.75mm, (2.75,3.25], >3.25mm}


def thickness_category(mm: float) -> int:
    return int(np.searchsorted(THICKNESS_BINS, float(mm)))


def slice_context(z_list, lung, meta):
    """Continuous: slice position normalised to the lung's craniocaudal extent
    (0 = most inferior lung slice, 1 = most superior; LPS +z is superior).
    Discrete: original acquisition slice-thickness category."""
    zz = np.where(lung.reshape(lung.shape[0], -1).any(1))[0]
    z0, z1 = (zz.min(), zz.max()) if zz.size else (0, lung.shape[0] - 1)
    pos = [(z - z0) / max(1, z1 - z0) for z in z_list]
    cat = thickness_category(meta.get("original_slice_spacing", 3.0))
    cont = torch.tensor(pos, dtype=torch.float32)[:, None]
    disc = torch.full((len(z_list), 1), cat, dtype=torch.long)
    return cont, disc


def build_glow(cfg: dict) -> Glow:
    return Glow(in_channels=cfg.get("in_channels", 2 * cfg.get("ctx_slices", 1) + 1), levels=cfg.get("levels", 3),
                steps=cfg.get("steps", 6), hidden=cfg.get("hidden", 64),
                squeeze_input=cfg.get("squeeze_input", True), alt_kernels=cfg.get("alt_kernels", False))


class FeatureFlowDetector(BaseDetector):
    """Flow (GLOW or ContextFlow++-style) on frozen pretrained lungmask features.
    Map = per-position NLL (glow.position_nll_map) on the 2.5 mm grid."""

    def __init__(self, model, extractor, name, device="cpu", batch=8, smooth_sigma=0.7):
        self.model = model.to(device).eval(); self.fx = extractor.to(device).eval()
        self.name, self.device, self.batch, self.smooth = name, device, batch, smooth_sigma

    @torch.no_grad()
    def raw_map(self, norm, lung, meta=None):
        from .glow import position_nll_map
        meta = meta or {}
        hu = meta.get("hu")
        if hu is None:
            raise ValueError("FeatureFlowDetector needs the HU volume in meta['hu']")
        zs = [z for z in range(norm.shape[0]) if lung[z].any()]
        out = np.zeros(norm.shape, np.float32)
        for i in range(0, len(zs), self.batch):
            chunk = zs[i:i + self.batch]
            f = self.fx(torch.from_numpy(hu[chunk].astype(np.float32)).to(self.device))
            if isinstance(self.model, ContextFlowPP):
                cont, disc = slice_context(chunk, lung, meta)
                _, o = self.model(f, cont.to(self.device), disc.to(self.device), return_maps=True)
            else:
                _, o = self.model(f, return_maps=True)
            m = position_nll_map(o).cpu().numpy()
            for j, z in enumerate(chunk):
                out[z] = m[j]
        if self.smooth:
            out = ndi.gaussian_filter(out, self.smooth)
        return out


def load_feature_extractor(root="."):
    from pathlib import Path
    from .feature_flow import LungmaskFeatures
    root = Path(root)
    pca = torch.load(root / "experiments/feature_extractor/pca.pt", map_location="cpu", weights_only=False)
    fx = LungmaskFeatures(str(root / "weights/unet_r231-d5d2fc3d.pth"), pca["n_components"])
    fx.pca_mean.copy_(pca["pca_mean"]); fx.pca_proj.copy_(pca["pca_proj"]); fx.fitted = True
    return fx


def load_detector(kind: str, ckpt_path: str, device="cpu", root=".") -> BaseDetector:
    if kind == "intensity_baseline":
        d = IntensityBaseline()
        if ckpt_path:
            import json, pathlib
            s = json.loads(pathlib.Path(ckpt_path).read_text())
            d.norm_mean, d.norm_std = s["norm_mean"], s["norm_std"]
            d.threshold = s.get("threshold")
            d.candidate_min_peak_score = s.get("candidate_min_peak_score")
            d.candidate_min_volume_mm3 = s.get("candidate_min_volume_mm3")
        return d
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    g = build_glow(cfg["model"])
    feature = bool(cfg["model"].get("feature_space"))
    if kind in ("glow", "feature_glow"):
        g.load_state_dict(ck["state_dict"])
        det = (FeatureFlowDetector(g, load_feature_extractor(root), kind, device) if feature
               else FlowDetector(g, "glow", cfg["model"].get("ctx_slices", 1), device))
    elif kind == "contextflow":
        cf_cfg = cfg["contextflow"]
        model = ContextFlowPP(g, n_cont=1, discrete_cardinalities=cf_cfg.get("discrete_cardinalities", [3]),
                              emb_dim=cf_cfg.get("emb_dim", 8), cn_hidden=cf_cfg.get("cn_hidden", 64),
                              discrete_mode=cf_cfg.get("discrete_mode", "embedding"))
        model.load_state_dict(ck["state_dict"])
        det = (FeatureFlowDetector(model, load_feature_extractor(root), "contextflow", device) if feature
               else FlowDetector(model, "contextflow", cfg["model"].get("ctx_slices", 1), device))
    else:
        raise ValueError(kind)
    det.norm_mean = ck.get("norm_mean", 0.0); det.norm_std = ck.get("norm_std", 1.0)
    det.threshold = ck.get("threshold")
    det.candidate_min_peak_score = ck.get("candidate_min_peak_score")
    det.candidate_min_volume_mm3 = ck.get("candidate_min_volume_mm3")
    return det
