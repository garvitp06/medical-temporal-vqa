"""Anomaly map -> candidate regions (spec §6.6).

Procedure (all parameters configurable, threshold chosen on VALIDATION):
 1. binary mask = (normalised map >= threshold) AND lung mask
 2. 26-connected components (scipy.ndimage.label)
 3. drop components smaller than ``min_volume_mm3`` (documented rule:
    default 30 mm^3 ~ a sphere of 3.9 mm diameter; below typical nodule
    reporting size and dominated by noise at 2.5 mm voxels)
 4. per component: voxel count, volume (mm^3), voxel + physical bounding box,
    centroid (physical, via the image direction/origin/spacing), longest
    diameter (mm), max/mean anomaly score, laterality and lung zone.
Candidates are "suspicious regions requiring review", never confirmed tumours.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


@dataclass
class Candidate:
    candidate_id: str
    n_voxels: int
    volume_mm3: float                 # mask-derived volume of the candidate region
    bbox_voxel: list                  # [z0,y0,x0,z1,y1,x1] inclusive-exclusive
    bbox_mm: list                     # [x0,y0,z0,x1,y1,z1] physical (LPS)
    bbox_volume_mm3: float            # BOUNDING-BOX ESTIMATE, not lesion volume
    centroid_voxel: list              # (z,y,x)
    centroid_mm: list                 # (x,y,z)
    longest_diameter_mm: float
    max_score: float
    mean_score: float
    laterality: str
    zone: str
    key_slice: int
    feature: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def index_to_physical(img: sitk.Image, zyx) -> np.ndarray:
    """(z,y,x) continuous index -> (x,y,z) physical mm using full geometry."""
    z, y, x = [float(v) for v in zyx]
    return np.array(img.TransformContinuousIndexToPhysicalPoint((x, y, z)))


def physical_to_index(img: sitk.Image, xyz) -> np.ndarray:
    x, y, z = img.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in xyz))
    return np.array([z, y, x])


def longest_diameter_mm(coords_zyx: np.ndarray, spacing_xyz) -> float:
    if len(coords_zyx) < 2:
        return float(min(spacing_xyz))
    pts = coords_zyx[:, ::-1] * np.asarray(spacing_xyz)       # -> (x,y,z) mm
    if len(pts) > 4:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:
            pass
    if len(pts) > 3000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
    return float(pdist(pts).max() + min(spacing_xyz))           # + one voxel (extent, not centres)


def lung_geometry(lung: np.ndarray):
    zz = np.where(lung.reshape(lung.shape[0], -1).any(1))[0]
    return (int(zz.min()), int(zz.max())) if zz.size else (0, lung.shape[0] - 1)


def zone_of(z_index, lung, img: sitk.Image) -> str:
    """Upper/middle/lower thirds of the lung craniocaudal extent (in physical z)."""
    z0, z1 = lung_geometry(lung)
    p0 = index_to_physical(img, (z0, 0, 0))[2]; p1 = index_to_physical(img, (z1, 0, 0))[2]
    pz = index_to_physical(img, (z_index, 0, 0))[2]
    lo, hi = min(p0, p1), max(p0, p1)
    rel = (pz - lo) / max(1e-6, hi - lo)
    return "lower" if rel < 1 / 3 else ("middle" if rel < 2 / 3 else "upper")


def extract_candidates(score_map: np.ndarray, lung: np.ndarray, img: sitk.Image, threshold: float,
                       min_volume_mm3: float = 30.0, max_candidates: int = 20, prefix="C",
                       body_midline_x_mm: float | None = None,
                       min_peak_score: float | None = None) -> tuple[list[Candidate], np.ndarray]:
    """``min_peak_score``: optional second, candidate-level rule (component
    peak score must reach it); selected on validation by lesion-level F1."""
    sp = img.GetSpacing()
    vox_mm3 = float(np.prod(sp))
    binary = (score_map >= threshold) & (lung > 0)
    lab, n = ndi.label(binary, structure=np.ones((3, 3, 3)))
    cands, keep_mask = [], np.zeros_like(lab, dtype=np.int32)
    if n == 0:
        return cands, keep_mask
    objs = ndi.find_objects(lab)
    if body_midline_x_mm is None:
        # midline = centre of the lung mask in physical x (robust to FOV shifts)
        c = np.array(ndi.center_of_mass(lung > 0))
        body_midline_x_mm = float(index_to_physical(img, c)[0])
    rows = []
    for i, sl in enumerate(objs, start=1):
        comp = lab[sl] == i
        nvox = int(comp.sum())
        vol = nvox * vox_mm3
        if vol < min_volume_mm3:
            continue
        coords = np.argwhere(comp) + np.array([s.start for s in sl])
        scores = score_map[tuple(coords.T)]
        if min_peak_score is not None and scores.max() < min_peak_score:
            continue
        rows.append((float(scores.max()), i, sl, coords, scores, nvox, vol))
    rows.sort(key=lambda r: -r[0])
    for rank, (mx, i, sl, coords, scores, nvox, vol) in enumerate(rows[:max_candidates]):
        z0, y0, x0 = [s.start for s in sl]; z1, y1, x1 = [s.stop for s in sl]
        cz, cy, cx = coords.mean(0)
        # physical box from the outer voxel faces
        corners = [index_to_physical(img, (z - 0.5, y - 0.5, x - 0.5))
                   for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
        corners = np.array(corners)
        bmin, bmax = corners.min(0), corners.max(0)
        cmm = index_to_physical(img, (cz, cy, cx))
        key = int(coords[np.argmax(scores)][0])
        cands.append(Candidate(
            candidate_id=f"{prefix}{rank+1}", n_voxels=nvox, volume_mm3=float(vol),
            bbox_voxel=[z0, y0, x0, z1, y1, x1], bbox_mm=bmin.tolist() + bmax.tolist(),
            bbox_volume_mm3=float(np.prod(bmax - bmin)), centroid_voxel=[float(cz), float(cy), float(cx)],
            centroid_mm=cmm.tolist(), longest_diameter_mm=longest_diameter_mm(coords, sp),
            max_score=float(scores.max()), mean_score=float(scores.mean()),
            laterality="left" if cmm[0] > body_midline_x_mm else "right",
            zone=zone_of(cz, lung, img), key_slice=key))
        keep_mask[tuple(coords.T)] = rank + 1
    return cands, keep_mask


def bbox2d_for_slice(cand: Candidate, z: int):
    z0, y0, x0, z1, y1, x1 = cand.bbox_voxel
    if not (z0 <= z < z1):
        return None
    return [x0, y0, x1, y1]
