"""Anomaly-detection / segmentation metrics (spec §10.1, §10.3).

Zero-denominator convention (used everywhere in this package):
  * precision = TP/(TP+FP); if nothing is predicted (TP+FP == 0) -> 1.0 when
    the reference is also empty (nothing to find), else 0.0.
  * recall = TP/(TP+FN); with no reference positives recall is UNDEFINED ->
    NaN, excluded from means (counts are reported).
  * Dice/IoU with both masks empty -> NaN (undefined) and excluded; with only
    one empty -> 0.0.
NaN-producing cases are counted and reported so exclusions are transparent.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def confusion(pred, ref):
    p = np.asarray(pred).astype(bool); r = np.asarray(ref).astype(bool)
    tp = int(np.sum(p & r)); fp = int(np.sum(p & ~r)); fn = int(np.sum(~p & r)); tn = int(np.sum(~p & ~r))
    return tp, fp, fn, tn


def precision(tp, fp, fn=0):
    if tp + fp == 0:
        return 1.0 if fn == 0 else 0.0
    return tp / (tp + fp)


def recall(tp, fn):
    return float("nan") if tp + fn == 0 else tp / (tp + fn)


def f1(p, r):
    if np.isnan(p) or np.isnan(r):
        return float("nan")
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def dice(pred, ref):
    p = np.asarray(pred).astype(bool); r = np.asarray(ref).astype(bool)
    s = p.sum() + r.sum()
    return float("nan") if s == 0 else float(2 * (p & r).sum() / s)


def iou(pred, ref):
    p = np.asarray(pred).astype(bool); r = np.asarray(ref).astype(bool)
    u = (p | r).sum()
    return float("nan") if u == 0 else float((p & r).sum() / u)


def segmentation_report(pred, ref):
    tp, fp, fn, tn = confusion(pred, ref)
    p, r = precision(tp, fp, fn), recall(tp, fn)
    return dict(tp=tp, fp=fp, fn=fn, dice=dice(pred, ref), iou=iou(pred, ref),
                precision=p, sensitivity=r, f1=f1(p, r))


def auroc_ap(scores, labels):
    """Continuous scores vs binary labels. Returns NaN when only one class."""
    s = np.asarray(scores, float).ravel(); y = np.asarray(labels).astype(int).ravel()
    if y.min() == y.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def best_threshold(scores_list, refs_list, lungs_list, grid):
    """Pick the voxel threshold maximising mean per-exam Dice on VALIDATION
    abnormal exams (exams with empty reference are skipped for Dice)."""
    best = (None, -1.0)
    for t in grid:
        ds = []
        for s, r, l in zip(scores_list, refs_list, lungs_list):
            if r.sum() == 0:
                continue
            ds.append(dice((s >= t) & (l > 0), r > 0))
        m = float(np.nanmean(ds)) if ds else float("nan")
        if np.isfinite(m) and m > best[1]:
            best = (float(t), m)
    return best


def subsample_voxels(score, ref, lung, n=20000, seed=0):
    """Stratified voxel subsample inside lung (all positives + random negatives)
    for AUROC/AP so one huge exam does not dominate memory."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(lung.ravel() > 0)
    pos = idx[ref.ravel()[idx] > 0]; neg = idx[ref.ravel()[idx] == 0]
    if len(neg) > n:
        neg = rng.choice(neg, n, replace=False)
    sel = np.concatenate([pos, neg])
    return score.ravel()[sel], (ref.ravel()[sel] > 0).astype(int)
