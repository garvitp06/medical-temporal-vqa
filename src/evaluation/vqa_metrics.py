"""VQA metrics (spec §10.6-10.8): exact match, accuracy, macro-F1, ROUGE-L
(supplementary only), calibration (Brier, ECE, reliability bins) and
patient-level bootstrap confidence intervals."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def exact_match(pred, ref):
    p = [str(a).strip().lower() for a in pred]; r = [str(b).strip().lower() for b in ref]
    return float(np.mean([a == b for a, b in zip(p, r)])) if r else float("nan")


def macro_f1(pred, ref):
    labels = sorted(set(ref))
    f1s = []
    for c in labels:
        tp = sum(p == c and r == c for p, r in zip(pred, ref))
        fp = sum(p == c and r != c for p, r in zip(pred, ref))
        fn = sum(p != c and r == c for p, r in zip(pred, ref))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f1s)) if f1s else float("nan")


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F-measure on whitespace tokens. Supplementary only: text
    overlap does not establish medical correctness."""
    a, b = pred.lower().split(), ref.lower().split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[-1][-1]
    p, r = l / len(a), l / len(b)
    return 0.0 if l == 0 else 2 * p * r / (p + r)


def brier_multiclass(probs, ref_idx):
    P = np.asarray(probs, float); y = np.zeros_like(P); y[np.arange(len(P)), ref_idx] = 1
    return float(np.mean(np.sum((P - y) ** 2, axis=1)))


def ece(probs, ref_idx, n_bins=10):
    P = np.asarray(probs, float)
    conf = P.max(1); pred = P.argmax(1); correct = (pred == np.asarray(ref_idx)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e, table = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
            table.append(dict(bin=[float(lo), float(hi)], n=int(m.sum()),
                              accuracy=float(correct[m].mean()), confidence=float(conf[m].mean())))
    return float(e), table


def patient_bootstrap_ci(values_by_patient: dict, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """values_by_patient: {patient_id: [per-item values]}. Resamples PATIENTS
    with replacement and recomputes the pooled statistic."""
    rng = np.random.default_rng(seed)
    pids = list(values_by_patient)
    if not pids:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat(np.concatenate([np.asarray(values_by_patient[p], float) for p in pids])))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(pids, len(pids), replace=True)
        v = np.concatenate([np.asarray(values_by_patient[p], float) for p in samp])
        v = v[np.isfinite(v)]
        if v.size:
            boots.append(stat(v))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if boots else (np.nan, np.nan)
    return point, float(lo), float(hi)


def group_by_patient(items, key="patient_id", value="correct"):
    g = defaultdict(list)
    for it in items:
        g[it[key]].append(float(it[value]))
    return dict(g)
