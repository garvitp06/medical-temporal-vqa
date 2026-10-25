"""Numerical measurement module (spec §8.3). The VQA model never produces
numbers; every number in an answer comes from here.

Conventions
  * V = N_voxels * dx * dy * dz  (mm^3), on the grid where the mask lives.
  * %dV = (V2 - V1) / V1 * 100; undefined (None) when V1 == 0 -> 'new' case.
  * Bounding-box volumes are reported separately and labelled as estimates.
"""
from __future__ import annotations

import numpy as np


def mask_volume_mm3(mask: np.ndarray, spacing_xyz) -> float:
    return float(np.count_nonzero(mask)) * float(np.prod(spacing_xyz))


def box_volume_mm3(bbox_mm) -> float:
    x0, y0, z0, x1, y1, z1 = bbox_mm
    return float(max(0, x1 - x0) * max(0, y1 - y0) * max(0, z1 - z0))


def change(v1: float | None, v2: float | None) -> dict:
    """Absolute and percentage change with explicit edge cases."""
    if v1 is None and v2 is None:
        return dict(absolute=None, percent=None, case="no_measurement")
    v1 = 0.0 if v1 is None else float(v1)
    v2 = 0.0 if v2 is None else float(v2)
    absolute = v2 - v1
    if v1 == 0.0 and v2 == 0.0:
        return dict(absolute=0.0, percent=None, case="absent_both")
    if v1 == 0.0:
        return dict(absolute=absolute, percent=None, case="new_candidate_percent_undefined")
    pct = absolute / v1 * 100.0
    case = "resolved" if v2 == 0.0 else "measured"
    return dict(absolute=absolute, percent=pct, case=case)


def centroid_displacement_mm(c1_xyz, c2_xyz) -> float:
    return float(np.linalg.norm(np.asarray(c2_xyz, float) - np.asarray(c1_xyz, float)))


def categorize_change(pct: float | None, case: str, stable_band_pct: float = 15.0) -> str:
    """Map a measured volume change to a category. ``stable_band_pct`` is a
    research default (volumetric repeatability of small nodules is commonly
    cited around +/-15-25%); it is configurable and NOT a clinical criterion."""
    if case == "new_candidate_percent_undefined":
        return "new"
    if case == "resolved":
        return "resolved"
    if pct is None:
        return "undetermined"
    if pct > stable_band_pct:
        return "increased"
    if pct < -stable_band_pct:
        return "decreased"
    return "stable"


def pair_measurements(c1: dict | None, c2: dict | None) -> dict:
    """c1/c2: candidate dicts (from localization) in the SAME (later-exam) space."""
    out = {}
    v1 = c1["volume_mm3"] if c1 else None
    v2 = c2["volume_mm3"] if c2 else None
    out["volume_mm3"] = dict(earlier=v1, later=v2, **change(v1, v2), source="candidate_mask")
    b1 = c1["bbox_volume_mm3"] if c1 else None
    b2 = c2["bbox_volume_mm3"] if c2 else None
    out["bbox_volume_mm3"] = dict(earlier=b1, later=b2, **change(b1, b2),
                                  note="bounding-box estimate, not lesion volume")
    d1 = c1["longest_diameter_mm"] if c1 else None
    d2 = c2["longest_diameter_mm"] if c2 else None
    out["longest_diameter_mm"] = dict(earlier=d1, later=d2, **change(d1, d2))
    s1 = c1["max_score"] if c1 else None
    s2 = c2["max_score"] if c2 else None
    out["anomaly_score_max"] = dict(earlier=s1, later=s2,
                                    absolute=(s2 - s1) if (s1 is not None and s2 is not None) else None,
                                    note="score change is not evidence of biological growth")
    out["centroid_displacement_mm"] = (centroid_displacement_mm(c1["centroid_mm"], c2["centroid_mm"])
                                       if (c1 and c2) else None)
    return out
