"""Cross-time candidate association (spec §8.2).

Both candidate sets must already be expressed in the SAME physical space
(earlier exam registered onto the later exam). Matching cost for a pair (i,j):

  cost = w_iou * (1 - IoU3D(box_i, box_j))
       + w_dist * min(1, d_ij / max_distance_mm)
       + w_vol  * |log(V_i / V_j)| / log(4)        (clipped to 1)
       + w_feat * (1 - cos(f_i, f_j)) / 2           (if features present)

Pairs with centroid distance > ``max_distance_mm`` are gated out. The
assignment is solved globally with the Hungarian algorithm. ``mode='iou'``
reproduces the IoU-only ablation (IoU > iou_threshold required).

Unmatched candidates: later-only -> "potentially_new", earlier-only ->
"potentially_resolved"; either becomes "uncertain" when registration is
unreliable or the candidate lies within ``edge_margin_mm`` of the field of
view boundary (it may simply be outside the other scan).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou_3d(a, b) -> float:
    """a, b: [x0,y0,z0,x1,y1,z1]."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    lo = np.maximum(a[:3], b[:3]); hi = np.minimum(a[3:], b[3:])
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    va = float(np.prod(np.clip(a[3:] - a[:3], 0, None)))
    vb = float(np.prod(np.clip(b[3:] - b[:3], 0, None)))
    union = va + vb - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a, b) -> float:
    a = np.asarray(a) > 0; b = np.asarray(b) > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


@dataclass
class MatchConfig:
    mode: str = "combined"           # combined | iou
    max_distance_mm: float = 25.0
    max_cost: float = 0.85
    iou_threshold: float = 0.1
    w_iou: float = 0.35
    w_dist: float = 0.45
    w_vol: float = 0.1
    w_feat: float = 0.1
    edge_margin_mm: float = 10.0

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class MatchResult:
    matches: list = field(default_factory=list)      # dicts(earlier, later, cost, iou, distance_mm)
    unmatched_earlier: list = field(default_factory=list)
    unmatched_later: list = field(default_factory=list)


def _cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return None
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else None


def pair_cost(c1, c2, cfg: MatchConfig):
    iou = box_iou_3d(c1["bbox_mm"], c2["bbox_mm"])
    d = float(np.linalg.norm(np.asarray(c1["centroid_mm"]) - np.asarray(c2["centroid_mm"])))
    if cfg.mode == "iou":
        return (1 - iou if iou >= cfg.iou_threshold else np.inf), iou, d
    if d > cfg.max_distance_mm:
        return np.inf, iou, d
    v1, v2 = max(c1["volume_mm3"], 1e-6), max(c2["volume_mm3"], 1e-6)
    vol_term = min(1.0, abs(np.log(v1 / v2)) / np.log(4))
    cos = _cos(c1.get("feature", []), c2.get("feature", []))
    w_feat = cfg.w_feat if cos is not None else 0.0
    feat_term = (1 - cos) / 2 if cos is not None else 0.0
    wsum = cfg.w_iou + cfg.w_dist + cfg.w_vol + w_feat
    cost = (cfg.w_iou * (1 - iou) + cfg.w_dist * min(1.0, d / cfg.max_distance_mm)
            + cfg.w_vol * vol_term + w_feat * feat_term) / wsum
    return cost, iou, d


def match_candidates(earlier: list[dict], later: list[dict], cfg: MatchConfig | None = None) -> MatchResult:
    cfg = cfg or MatchConfig()
    res = MatchResult()
    if not earlier or not later:
        res.unmatched_earlier = list(range(len(earlier)))
        res.unmatched_later = list(range(len(later)))
        return res
    C = np.full((len(earlier), len(later)), 1e6)
    info = {}
    for i, a in enumerate(earlier):
        for j, b in enumerate(later):
            c, iou, d = pair_cost(a, b, cfg)
            info[(i, j)] = (c, iou, d)
            if np.isfinite(c):
                C[i, j] = c
    rows, cols = linear_sum_assignment(C)
    used_i, used_j = set(), set()
    limit = 1.0 if cfg.mode == "iou" else cfg.max_cost
    for i, j in zip(rows, cols):
        c, iou, d = info[(i, j)]
        if np.isfinite(c) and c <= limit:
            res.matches.append(dict(earlier=int(i), later=int(j), cost=float(c), iou=float(iou),
                                    distance_mm=float(d)))
            used_i.add(i); used_j.add(j)
    res.unmatched_earlier = [i for i in range(len(earlier)) if i not in used_i]
    res.unmatched_later = [j for j in range(len(later)) if j not in used_j]
    return res


def near_fov_edge(cand: dict, fov_bbox_mm, margin_mm: float) -> bool:
    x0, y0, z0, x1, y1, z1 = fov_bbox_mm
    c = cand["centroid_mm"]
    return bool(c[2] - z0 < margin_mm or z1 - c[2] < margin_mm
                or c[0] - x0 < margin_mm or x1 - c[0] < margin_mm
                or c[1] - y0 < margin_mm or y1 - c[1] < margin_mm)


def classify_unmatched(res: MatchResult, earlier, later, registration_reliable: bool,
                       fov_earlier_mm=None, fov_later_mm=None, cfg: MatchConfig | None = None):
    cfg = cfg or MatchConfig()
    out = []
    for i in res.unmatched_earlier:
        st = "potentially_resolved"
        if not registration_reliable or (fov_later_mm and near_fov_edge(earlier[i], fov_later_mm, cfg.edge_margin_mm)):
            st = "uncertain"
        out.append(dict(side="earlier", index=i, status=st))
    for j in res.unmatched_later:
        st = "potentially_new"
        if not registration_reliable or (fov_earlier_mm and near_fov_edge(later[j], fov_earlier_mm, cfg.edge_margin_mm)):
            st = "uncertain"
        out.append(dict(side="later", index=j, status=st))
    return out
