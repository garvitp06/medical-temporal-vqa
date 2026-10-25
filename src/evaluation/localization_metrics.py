"""Lesion-level 3D detection metrics (spec §10.2).

A predicted 3D box is a true positive if its 3D IoU with an unmatched
reference box is >= ``iou_threshold`` (greedy matching in descending score
order, the standard detection protocol). Precision/recall/F1 at that
threshold, mean IoU of matched pairs, and AP (all-point interpolated area
under the precision-recall curve over the score ranking) are reported.
A ``hit_criterion='centroid'`` option counts a hit when the predicted
centroid falls inside the reference box — useful for small lesions where
box IoU is very sensitive to a single voxel.
"""
from __future__ import annotations

import numpy as np

from ..temporal.candidate_matching import box_iou_3d


def _inside(c, box):
    return all(box[k] <= c[k] <= box[k + 3] for k in range(3))


def match_detections(preds, refs, iou_threshold=0.1, hit_criterion="iou"):
    """preds: list of dict(bbox_mm, centroid_mm, score); refs: list of dict(bbox_mm).
    Returns list of (score, is_tp, iou) in score order and n_refs."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    used = set()
    out = []
    for i in order:
        p = preds[i]
        best, best_j = -1.0, None
        for j, r in enumerate(refs):
            if j in used:
                continue
            v = box_iou_3d(p["bbox_mm"], r["bbox_mm"])
            ok = v >= iou_threshold if hit_criterion == "iou" else _inside(p["centroid_mm"], r["bbox_mm"])
            if ok and v > best:
                best, best_j = v, j
        if best_j is not None:
            used.add(best_j)
            out.append((p["score"], True, best))
        else:
            out.append((p["score"], False, 0.0))
    return out, len(refs)


def average_precision(records, n_refs):
    if n_refs == 0:
        return float("nan")
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records]); fp = np.cumsum([not r[1] for r in records])
    rec = tp / n_refs; prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_report(per_exam, iou_threshold=0.1, hit_criterion="iou"):
    """per_exam: list of (preds, refs). Pooled lesion-level metrics."""
    allrec, nref, tp_ious = [], 0, []
    for preds, refs in per_exam:
        rec, n = match_detections(preds, refs, iou_threshold, hit_criterion)
        allrec += rec; nref += n
        tp_ious += [r[2] for r in rec if r[1]]
    tp = sum(r[1] for r in allrec); fp = len(allrec) - tp; fn = nref - tp
    p = tp / (tp + fp) if tp + fp else (1.0 if nref == 0 else 0.0)
    r = tp / nref if nref else float("nan")
    if not np.isfinite(r):
        f = float("nan")
    else:
        f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return dict(iou_threshold=iou_threshold, hit_criterion=hit_criterion, tp=int(tp), fp=int(fp),
                fn=int(fn), n_reference_lesions=int(nref), precision=p, recall=r, f1=f,
                mean_iou_matched=float(np.mean(tp_ious)) if tp_ious else float("nan"),
                average_precision=average_precision(allrec, nref))
