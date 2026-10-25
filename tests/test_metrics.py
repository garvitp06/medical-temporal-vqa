import math

import numpy as np
import pytest

from src.evaluation import anomaly_metrics as am
from src.evaluation.localization_metrics import average_precision, detection_report
from src.evaluation.temporal_metrics import change_errors, mae, mape, measurement_errors, rmse
from src.evaluation.vqa_metrics import ece, exact_match, macro_f1, patient_bootstrap_ci, rouge_l, brier_multiclass
from src.temporal.candidate_matching import box_iou_3d, mask_iou


def test_dice_iou_known_values():
    a = np.zeros((4, 4, 4), bool); b = np.zeros_like(a)
    a[:2] = True; b[1:3] = True            # |A|=32 |B|=32 |A∩B|=16
    assert am.dice(a, b) == pytest.approx(0.5)
    assert am.iou(a, b) == pytest.approx(16 / 48)
    assert mask_iou(a, b) == pytest.approx(16 / 48)


def test_empty_mask_conventions():
    z = np.zeros((3, 3), bool); o = z.copy(); o[0, 0] = True
    assert math.isnan(am.dice(z, z)) and math.isnan(am.iou(z, z))
    assert am.dice(o, z) == 0.0
    assert math.isnan(am.recall(0, 0))
    assert am.precision(0, 0, 0) == 1.0 and am.precision(0, 0, 3) == 0.0


def test_precision_recall_f1():
    tp, fp, fn, _ = am.confusion([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (tp, fp, fn) == (2, 1, 1)
    p, r = am.precision(tp, fp, fn), am.recall(tp, fn)
    assert am.f1(p, r) == pytest.approx(2 / 3)


def test_box_iou_3d():
    a = [0, 0, 0, 2, 2, 2]; b = [1, 1, 1, 3, 3, 3]
    assert box_iou_3d(a, b) == pytest.approx(1 / 15)
    assert box_iou_3d(a, a) == 1.0
    assert box_iou_3d(a, [5, 5, 5, 6, 6, 6]) == 0.0


def test_auroc_ap():
    au, ap = am.auroc_ap([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])
    assert au == pytest.approx(0.75)
    assert math.isnan(am.auroc_ap([1, 2], [1, 1])[0])


def test_detection_report_and_ap():
    refs = [dict(bbox_mm=[0, 0, 0, 10, 10, 10])]
    preds = [dict(bbox_mm=[1, 1, 1, 10, 10, 10], centroid_mm=[5, 5, 5], score=0.9),
             dict(bbox_mm=[50, 50, 50, 60, 60, 60], centroid_mm=[55, 55, 55], score=0.5)]
    r = detection_report([(preds, refs)], iou_threshold=0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)
    assert r["precision"] == 0.5 and r["recall"] == 1.0
    assert r["average_precision"] == pytest.approx(1.0)
    assert average_precision([], 0) != average_precision([], 0)        # NaN when no references


def test_regression_errors():
    assert mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(math.sqrt(4 / 3))
    m, excl = mape([110, 5], [100, 0], min_abs_ref=1.0)
    assert m == pytest.approx(10.0) and excl == 1
    ce = change_errors([10, 20], [15, 20], [10, 20], [12, 30])     # pred Δ=(5,0) ref Δ=(2,10)
    assert ce["mae"] == pytest.approx((3 + 10) / 2)
    assert measurement_errors([1], [2], 0.5)["mae"] == 1.0


def test_vqa_metrics():
    assert exact_match(["Yes", "no"], ["yes", "yes"]) == 0.5
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert rouge_l("the lesion grew", "the lesion grew") == 1.0
    P = np.array([[0.9, 0.1], [0.6, 0.4]]); y = [0, 1]
    e, table = ece(P, y, n_bins=10)
    assert e == pytest.approx(0.5 * 0.1 + 0.5 * 0.6)
    assert brier_multiclass(P, y) == pytest.approx(((0.01 + 0.01) + (0.36 + 0.36)) / 2)


def test_patient_bootstrap_ci_contains_point():
    v = {f"p{i}": [float(i % 2)] * 3 for i in range(20)}
    p, lo, hi = patient_bootstrap_ci(v, n_boot=300)
    assert lo <= p <= hi and p == pytest.approx(0.5)


def test_best_threshold_selects_separating_value():
    s = np.zeros((4, 4, 4)); s[1:3, 1:3, 1:3] = 5.0
    ref = s > 1; lung = np.ones_like(s)
    t, d = am.best_threshold([s], [ref], [lung], [1.0, 4.0, 6.0])
    assert t in (1.0, 4.0) and d == pytest.approx(1.0)
