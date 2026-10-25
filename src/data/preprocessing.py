"""CT intensity + spatial preprocessing with a full operation log (spec §4.2-4.5).

The HU volume is never modified in place: we keep
  * ``hu``          resampled Hounsfield Units (for quantitative analysis)
  * ``normalized``  windowed + scaled image for the models
and every operation is appended to ``ops`` so it can be written next to the
processed volume.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

log = logging.getLogger(__name__)


@dataclass
class PreprocConfig:
    target_spacing: tuple = (2.5, 2.5, 2.5)     # mm (x, y, z)
    window_center: float = -600.0               # lung window
    window_width: float = 1500.0
    normalize: str = "window_minus1_1"          # or "window_0_1"
    lung_mask_method: str = "auto"              # auto | lungmask | threshold
    lungmask_weights: str = "weights/unet_r231-d5d2fc3d.pth"
    min_lung_volume_ml: float = 500.0           # QC plausibility bound
    max_lung_volume_ml: float = 9000.0

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        for k in ("target_spacing",):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProcessedExam:
    image: sitk.Image                 # resampled HU image (physical geometry attached)
    normalized: np.ndarray            # (z,y,x) float32 model input
    lung_mask: np.ndarray             # (z,y,x) uint8
    original_meta: dict
    qc: dict
    ops: list = field(default_factory=list)

    @property
    def hu(self) -> np.ndarray:
        return sitk.GetArrayFromImage(self.image)

    def save(self, out_dir: Path, exam_id: str):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(self.image, str(out_dir / f"{exam_id}_hu.nii.gz"))
        n = sitk.GetImageFromArray(self.normalized.astype(np.float32)); n.CopyInformation(self.image)
        sitk.WriteImage(n, str(out_dir / f"{exam_id}_norm.nii.gz"))
        m = sitk.GetImageFromArray(self.lung_mask.astype(np.uint8)); m.CopyInformation(self.image)
        sitk.WriteImage(m, str(out_dir / f"{exam_id}_lung.nii.gz"))
        (out_dir / f"{exam_id}_preproc.json").write_text(json.dumps(
            dict(original_meta=self.original_meta, qc=self.qc, ops=self.ops), indent=1, default=str))


def window_normalize(hu: np.ndarray, center: float, width: float, mode: str) -> np.ndarray:
    lo, hi = center - width / 2, center + width / 2
    x = (np.clip(hu, lo, hi) - lo) / (hi - lo)
    return (x * 2 - 1).astype(np.float32) if mode == "window_minus1_1" else x.astype(np.float32)


def resample_to_spacing(img: sitk.Image, spacing, is_mask=False, default=-1024.0) -> sitk.Image:
    """Resample in physical space. Linear for intensities, nearest for masks.
    The direction matrix is preserved; the output grid covers the same extent."""
    old_sp = np.array(img.GetSpacing()); old_sz = np.array(img.GetSize())
    new_sp = np.array(spacing, float)
    new_sz = np.maximum(1, np.round(old_sz * old_sp / new_sp)).astype(int)
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(tuple(new_sp)); f.SetSize([int(s) for s in new_sz])
    f.SetOutputOrigin(img.GetOrigin()); f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    f.SetDefaultPixelValue(0 if is_mask else default)
    return f.Execute(img)


def resample_like(img: sitk.Image, ref: sitk.Image, is_mask=False, transform=None, default=-1024.0):
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(ref)
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    f.SetDefaultPixelValue(0 if is_mask else default)
    if transform is not None:
        f.SetTransform(transform)
    return f.Execute(img)


def orient_lps(img: sitk.Image) -> sitk.Image:
    """Reorient to the canonical LPS axis order (identity direction)."""
    return sitk.DICOMOrient(img, "LPS")


# --------------------------------------------------------------------------- #
# Lung masks
# --------------------------------------------------------------------------- #
def threshold_lung_mask(hu: np.ndarray, spacing_xyz) -> np.ndarray:
    """Classical lung segmentation: air (<-400 HU) inside the body, excluding
    air connected to the image border, keep the two largest components, then
    per-slice hole filling + closing so that juxtavascular/solid nodules
    inside the lung are included (needed so lesions are not cropped away)."""
    air = hu < -400
    body = ndi.binary_fill_holes(ndi.binary_opening(hu > -500, iterations=2), structure=np.ones((1, 3, 3)))
    # remove outside air: anything in air connected to the slice border
    lab, n = ndi.label(air & body)
    if n == 0:
        return np.zeros_like(hu, np.uint8)
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.argsort(sizes)[::-1][:2] + 1
    voxel_ml = float(np.prod(spacing_xyz)) / 1000.0
    keep = [k for k in keep if sizes[k - 1] * voxel_ml > 50]
    mask = np.isin(lab, keep)
    r_mm = 6.0     # closing radius (mm): bridges small pleural indentations, not the mediastinum
    st = ndi.generate_binary_structure(3, 1)
    it = max(1, int(round(r_mm / max(spacing_xyz[0], 1e-3) / 2)))
    mask = ndi.binary_closing(mask, structure=st, iterations=it)
    mask = np.stack([ndi.binary_fill_holes(s) for s in mask])
    return mask.astype(np.uint8)


def lungmask_mask(img: sitk.Image, weights: str) -> np.ndarray:
    """Pretrained U-Net R231 (Hofmanninger et al., 2020) via the `lungmask`
    package. Returns 1 for any lung (labels 1/2 merged)."""
    from lungmask import LMInferer  # imported lazily: optional dependency
    inf = LMInferer(modelname="R231", modelpath=weights if Path(weights).exists() else None,
                    force_cpu=True, batch_size=8, tqdm_disable=True)
    seg = inf.apply(img)
    return (seg > 0).astype(np.uint8)


def compute_lung_mask(img: sitk.Image, cfg: PreprocConfig, ops: list, qc: dict) -> np.ndarray:
    hu = sitk.GetArrayFromImage(img)
    sp = img.GetSpacing()
    voxel_ml = float(np.prod(sp)) / 1000.0
    method = cfg.lung_mask_method
    mask = None
    if method in ("auto", "lungmask"):
        try:
            mask = lungmask_mask(img, cfg.lungmask_weights)
            vol = mask.sum() * voxel_ml
            if not (cfg.min_lung_volume_ml <= vol <= cfg.max_lung_volume_ml):
                qc.setdefault("warnings", []).append(
                    f"lungmask volume {vol:.0f} ml implausible; falling back to threshold method")
                mask = None
            else:
                # close small gaps so lesions at the pleura/hilum are inside the mask
                mask = np.stack([ndi.binary_fill_holes(s) for s in mask]).astype(np.uint8)
                ops.append(dict(op="lung_mask", method="lungmask_R231_pretrained",
                                weights=cfg.lungmask_weights))
        except Exception as exc:  # missing package/weights or inference error
            if method == "lungmask":
                raise
            qc.setdefault("warnings", []).append(f"lungmask unavailable ({type(exc).__name__}); threshold fallback")
            mask = None
    if mask is None:
        mask = threshold_lung_mask(hu, sp)
        ops.append(dict(op="lung_mask", method="threshold_-400HU_2largest_closing_fill"))
    qc["lung_volume_ml"] = float(mask.sum() * voxel_ml)
    qc["lung_mask_ok"] = bool(cfg.min_lung_volume_ml <= qc["lung_volume_ml"] <= cfg.max_lung_volume_ml)
    return mask


# --------------------------------------------------------------------------- #
def preprocess_series(loaded, cfg: PreprocConfig | None = None) -> ProcessedExam:
    """``loaded`` is a :class:`LoadedSeries` (or anything with .to_sitk())."""
    cfg = cfg or PreprocConfig()
    ops, qc = [], {"warnings": list(getattr(loaded, "warnings", []))}
    img = loaded.to_sitk() if hasattr(loaded, "to_sitk") else loaded
    meta = dict(getattr(loaded, "metadata", {}))
    meta.update(original_spacing=list(img.GetSpacing()), original_size=list(img.GetSize()),
                original_origin=list(img.GetOrigin()), original_direction=list(img.GetDirection()))
    ops.append(dict(op="hu_conversion", detail="RescaleSlope*stored+RescaleIntercept (dicom_loader)"))
    img = orient_lps(img)
    ops.append(dict(op="reorient", target="LPS"))
    img = resample_to_spacing(img, cfg.target_spacing)
    ops.append(dict(op="resample", spacing=list(cfg.target_spacing), interpolator="linear",
                    default_hu=-1024))
    hu = sitk.GetArrayFromImage(img)
    qc["hu_range"] = [float(hu.min()), float(hu.max())]
    qc["hu_plausible"] = bool(hu.min() >= -1100 and hu.max() <= 4000 and np.median(hu) < 200)
    if not qc["hu_plausible"]:
        qc["warnings"].append("HU distribution implausible for CT — check rescale parameters")
    mask = compute_lung_mask(img, cfg, ops, qc)
    norm = window_normalize(hu, cfg.window_center, cfg.window_width, cfg.normalize)
    ops.append(dict(op="window_normalize", center=cfg.window_center, width=cfg.window_width,
                    mode=cfg.normalize))
    qc["ok"] = bool(qc["hu_plausible"] and qc["lung_mask_ok"])
    return ProcessedExam(img, norm, mask, meta, qc, ops)
