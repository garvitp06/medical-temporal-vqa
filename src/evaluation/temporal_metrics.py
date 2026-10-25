"""Temporal measurement metrics (spec §10.5).

Single-measurement error (e.g. later-exam volume) and change error (error in
V2 - V1 or in %change) are computed by separate functions and must be
reported separately.
MAPE is only computed over references with |y| >= ``min_abs_ref``; the
number of excluded items is returned.
"""
from __future__ import annotations

import numpy as np


def _clean(pred, ref):
    p = np.asarray(pred, float); r = np.asarray(ref, float)
    ok = np.isfinite(p) & np.isfinite(r)
    return p[ok], r[ok]


def mae(pred, ref):
    p, r = _clean(pred, ref)
    return float(np.mean(np.abs(p - r))) if p.size else float("nan")


def rmse(pred, ref):
    p, r = _clean(pred, ref)
    return float(np.sqrt(np.mean((p - r) ** 2))) if p.size else float("nan")


def mape(pred, ref, min_abs_ref=1e-6):
    p, r = _clean(pred, ref)
    keep = np.abs(r) >= min_abs_ref
    n_excl = int((~keep).sum())
    if not keep.any():
        return float("nan"), n_excl
    return float(100.0 * np.mean(np.abs((p[keep] - r[keep]) / r[keep]))), n_excl


def measurement_errors(pred, ref, min_abs_ref):
    m, excl = mape(pred, ref, min_abs_ref)
    return dict(n=int(len(_clean(pred, ref)[0])), mae=mae(pred, ref), rmse=rmse(pred, ref),
                mape=m, mape_excluded=excl)


def change_errors(pred_v1, pred_v2, ref_v1, ref_v2):
    """Error in the absolute change (V2 - V1), distinct from single errors."""
    pd_ = np.asarray(pred_v2, float) - np.asarray(pred_v1, float)
    rd = np.asarray(ref_v2, float) - np.asarray(ref_v1, float)
    return dict(n=int(np.isfinite(pd_ - rd).sum()), mae=mae(pd_, rd), rmse=rmse(pd_, rd))


def category_agreement(pred_cats, ref_cats):
    p = list(pred_cats); r = list(ref_cats)
    n = len(r)
    acc = float(np.mean([a == b for a, b in zip(p, r)])) if n else float("nan")
    labels = sorted(set(r) | set(p))
    conf = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(r, p):
        conf[a][b] += 1
    return dict(n=n, accuracy=acc, confusion_ref_rows_pred_cols=conf)
