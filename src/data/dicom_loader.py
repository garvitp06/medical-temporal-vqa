"""DICOM series discovery, validation, spatial sorting and HU conversion.

Design rules (see master spec §4.1):
* Series are grouped by (StudyInstanceUID, SeriesInstanceUID); unrelated
  series are never merged.
* Slices are sorted by projecting ImagePositionPatient onto the slice normal
  (cross product of the ImageOrientationPatient row/column cosines) — never by
  file name or InstanceNumber.
* Stored values are converted to HU with RescaleSlope/RescaleIntercept; we do
  not assume stored values are already HU.
* Spacing between slices is derived from positions (not SliceThickness).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

log = logging.getLogger(__name__)


class DicomValidationError(ValueError):
    """Raised when a series cannot be safely reconstructed into a volume."""


@dataclass
class SeriesInfo:
    study_uid: str
    series_uid: str
    files: list
    modality: str = ""
    description: str = ""
    n_files: int = 0


@dataclass
class LoadedSeries:
    hu: np.ndarray                 # float32, (z, y, x), Hounsfield Units
    spacing: tuple                 # (x, y, z) mm
    origin: tuple                  # (x, y, z) mm, position of voxel (0,0,0)
    direction: tuple               # 9-tuple row-major 3x3, columns = axis cosines
    slice_positions: list          # projected position of each slice along normal (mm)
    metadata: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_sitk(self) -> sitk.Image:
        img = sitk.GetImageFromArray(self.hu.astype(np.float32))
        img.SetSpacing(tuple(float(s) for s in self.spacing))
        img.SetOrigin(tuple(float(o) for o in self.origin))
        img.SetDirection(tuple(float(d) for d in self.direction))
        return img


# Metadata keys safe to keep (no direct identifiers).
_SAFE_KEYS = ["Modality", "StudyDate", "SeriesDescription", "KVP", "ConvolutionKernel",
              "Manufacturer", "ManufacturerModelName", "SliceThickness", "BitsStored",
              "PixelRepresentation", "RescaleType", "ImageComments"]


def discover_series(path: str | Path) -> list[SeriesInfo]:
    """Group every readable DICOM file under ``path`` by study/series UID."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    groups: dict = defaultdict(list)
    meta: dict = {}
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:  # not DICOM / corrupted header
            log.debug("skip non-DICOM file %s", f.name)
            continue
        if "SeriesInstanceUID" not in ds:
            continue
        key = (str(ds.get("StudyInstanceUID", "")), str(ds.SeriesInstanceUID))
        groups[key].append(f)
        meta.setdefault(key, (str(ds.get("Modality", "")), str(ds.get("SeriesDescription", ""))))
    out = [SeriesInfo(k[0], k[1], v, meta[k][0], meta[k][1], len(v)) for k, v in groups.items()]
    out.sort(key=lambda s: -s.n_files)
    return out


def select_series(series: list[SeriesInfo], series_uid: str | None = None,
                  min_slices: int = 10) -> SeriesInfo:
    """Pick a series explicitly. Without ``series_uid`` the single CT series
    with >= ``min_slices`` is selected; ambiguity raises an error."""
    if series_uid:
        for s in series:
            if s.series_uid == series_uid:
                return s
        raise DicomValidationError(f"series {series_uid} not found")
    cands = [s for s in series if s.modality in ("CT", "") and s.n_files >= min_slices]
    if not cands:
        raise DicomValidationError("no CT series with enough slices found")
    if len(cands) > 1:
        raise DicomValidationError(
            f"{len(cands)} candidate CT series found; pass series_uid explicitly: "
            + ", ".join(f"{c.series_uid} ({c.n_files} slices, '{c.description}')" for c in cands))
    return cands[0]


def _normal(iop):
    row = np.asarray(iop[:3], float)
    col = np.asarray(iop[3:], float)
    n = np.cross(row, col)
    return row, col, n / np.linalg.norm(n)


def to_hu(stored: np.ndarray, ds) -> np.ndarray:
    """Convert stored pixel values to HU using the modality LUT parameters."""
    arr = stored.astype(np.float32)
    if "RescaleSlope" in ds or "RescaleIntercept" in ds:
        slope = float(ds.get("RescaleSlope", 1.0))
        inter = float(ds.get("RescaleIntercept", 0.0))
        return arr * slope + inter
    raise DicomValidationError("RescaleSlope/RescaleIntercept missing; cannot convert to HU safely")


def load_series(series: SeriesInfo, spacing_tolerance: float = 0.05,
                allow_gaps: bool = False) -> LoadedSeries:
    """Read, validate, sort and convert one series to an HU volume."""
    slices, warnings = [], []
    for f in series.files:
        try:
            ds = pydicom.dcmread(str(f))
            _ = ds.pixel_array
        except Exception as exc:
            raise DicomValidationError(f"corrupted or unreadable slice {f.name}: {exc}") from exc
        for tag in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
            if tag not in ds:
                raise DicomValidationError(f"slice {f.name} is missing {tag}")
        slices.append(ds)
    if len(slices) < 2:
        raise DicomValidationError("series has fewer than 2 slices")
    iop0 = np.asarray(slices[0].ImageOrientationPatient, float)
    for ds in slices:
        if not np.allclose(np.asarray(ds.ImageOrientationPatient, float), iop0, atol=1e-4):
            raise DicomValidationError("inconsistent ImageOrientationPatient within series")
        if (ds.Rows, ds.Columns) != (slices[0].Rows, slices[0].Columns):
            raise DicomValidationError("inconsistent slice dimensions within series")
    row, col, n = _normal(iop0)
    pos = np.array([np.dot(np.asarray(ds.ImagePositionPatient, float), n) for ds in slices])
    order = np.argsort(pos)
    slices = [slices[i] for i in order]
    pos = pos[order]
    d = np.diff(pos)
    if np.any(d < 1e-3):
        raise DicomValidationError("duplicate slice positions detected")
    step = float(np.median(d))
    if np.max(np.abs(d - step)) > spacing_tolerance * step + 1e-3:
        msg = (f"non-uniform slice spacing (median {step:.3f} mm, range "
               f"{d.min():.3f}-{d.max():.3f} mm): possible missing slices")
        if not allow_gaps:
            raise DicomValidationError(msg)
        warnings.append(msg)
    ps = [float(v) for v in slices[0].PixelSpacing]   # [row spacing (y), col spacing (x)]
    vol = np.stack([to_hu(ds.pixel_array, ds) for ds in slices]).astype(np.float32)
    bits = int(slices[0].get("BitsAllocated", 16))
    if bits != 16:
        warnings.append(f"BitsAllocated={bits} (expected 16)")
    # direction matrix: columns are the x (row cosine), y (col cosine), z (normal) axes
    direction = np.stack([row, col, n], axis=1).reshape(-1)
    meta = {k: str(slices[0].get(k, "")) for k in _SAFE_KEYS}
    meta.update(study_uid=series.study_uid, series_uid=series.series_uid,
                n_slices=len(slices), rows=int(slices[0].Rows), cols=int(slices[0].Columns))
    return LoadedSeries(hu=vol, spacing=(ps[1], ps[0], step),
                        origin=tuple(float(v) for v in slices[0].ImagePositionPatient),
                        direction=tuple(direction.tolist()), slice_positions=pos.tolist(),
                        metadata=meta, warnings=warnings)


def load_dicom_dir(path, series_uid=None, **kw) -> LoadedSeries:
    series = discover_series(path)
    if not series:
        raise DicomValidationError(f"no DICOM files found under {path}")
    return load_series(select_series(series, series_uid), **kw)
