import numpy as np
import pytest

from src.temporal.measurements import (box_volume_mm3, categorize_change, centroid_displacement_mm, change,
                                       mask_volume_mm3, pair_measurements)


def test_mask_volume_uses_physical_spacing():
    m = np.zeros((4, 4, 4), bool); m[:2, :2, :2] = True
    assert mask_volume_mm3(m, (0.5, 0.5, 2.0)) == pytest.approx(8 * 0.5)


def test_percent_change_and_edge_cases():
    assert change(100, 150) == dict(absolute=50, percent=50.0, case="measured")
    c = change(0, 80)
    assert c["percent"] is None and c["case"] == "new_candidate_percent_undefined" and c["absolute"] == 80
    assert change(None, 80)["case"] == "new_candidate_percent_undefined"
    r = change(80, 0)
    assert r["percent"] == -100.0 and r["case"] == "resolved"
    assert change(0, 0)["case"] == "absent_both" and change(None, None)["case"] == "no_measurement"


def test_categories():
    assert categorize_change(40, "measured") == "increased"
    assert categorize_change(-40, "measured") == "decreased"
    assert categorize_change(5, "measured") == "stable"
    assert categorize_change(None, "new_candidate_percent_undefined") == "new"
    assert categorize_change(-100, "resolved") == "resolved"


def test_box_volume_and_displacement():
    assert box_volume_mm3([0, 0, 0, 2, 3, 4]) == 24
    assert centroid_displacement_mm([0, 0, 0], [3, 4, 0]) == 5


def test_pair_measurements_labels_bbox_estimate():
    c1 = dict(volume_mm3=100.0, bbox_volume_mm3=200.0, longest_diameter_mm=6.0, max_score=4.0, centroid_mm=[0, 0, 0])
    c2 = dict(volume_mm3=150.0, bbox_volume_mm3=330.0, longest_diameter_mm=7.0, max_score=5.0, centroid_mm=[0, 0, 2])
    m = pair_measurements(c1, c2)
    assert m["volume_mm3"]["percent"] == pytest.approx(50.0)
    assert "estimate" in m["bbox_volume_mm3"]["note"]
    assert "not evidence" in m["anomaly_score_max"]["note"]
    assert m["centroid_displacement_mm"] == pytest.approx(2.0)
    new = pair_measurements(None, c2)
    assert new["volume_mm3"]["percent"] is None and new["centroid_displacement_mm"] is None
