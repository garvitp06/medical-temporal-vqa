import numpy as np
import pytest
import SimpleITK as sitk

from src.inference.localization import extract_candidates
from src.temporal.candidate_matching import MatchConfig, classify_unmatched, match_candidates


def _img(shape, spacing=(2.0, 2.0, 2.0)):
    im = sitk.GetImageFromArray(np.zeros(shape, np.float32)); im.SetSpacing(spacing)
    im.SetOrigin((-shape[2] * spacing[0] / 2, -shape[1] * spacing[1] / 2, 0.0)); return im


def _blob(m, c, r, val=6.0):
    zz, yy, xx = np.indices(m.shape)
    m[((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r] = val


def test_extract_candidates_geometry_and_small_component_rule():
    shape = (30, 40, 40)
    s = np.zeros(shape, np.float32); lung = np.ones(shape, np.uint8)
    _blob(s, (10, 20, 8), 3)          # right side (low x -> patient right in LPS)
    _blob(s, (20, 20, 32), 2)         # left side
    s[2, 2, 2] = 9.0                  # single voxel: 8 mm^3 < 30 mm^3 -> removed
    img = _img(shape)
    cands, mask = extract_candidates(s, lung, img, threshold=3.0, min_volume_mm3=30.0)
    assert len(cands) == 2
    big = max(cands, key=lambda c: c.n_voxels)
    assert big.laterality == "right"
    assert big.volume_mm3 == pytest.approx(big.n_voxels * 8.0)
    z0, y0, x0, z1, y1, x1 = big.bbox_voxel
    assert (z1 - z0, y1 - y0, x1 - x0) == (7, 7, 7)
    assert big.bbox_volume_mm3 == pytest.approx(14.0 ** 3)
    assert big.longest_diameter_mm == pytest.approx(14.0, abs=0.5)
    assert big.centroid_mm == pytest.approx(list(img.TransformIndexToPhysicalPoint((8, 20, 10))))
    assert mask.max() == 2


def _cand(c, v=500.0, half=5.0):
    return dict(centroid_mm=list(c), volume_mm3=v, bbox_mm=[c[0] - half, c[1] - half, c[2] - half,
                                                            c[0] + half, c[1] + half, c[2] + half])


def test_combined_matching_survives_displacement_iou_only_does_not():
    e = [_cand((0, 0, 0)), _cand((60, 0, 0))]
    l = [_cand((12, 0, 0), v=900), _cand((61, 1, 0))]           # first lesion moved 12 mm (no box overlap)
    comb = match_candidates(e, l, MatchConfig(mode="combined"))
    assert {(m["earlier"], m["later"]) for m in comb.matches} == {(0, 0), (1, 1)}
    iou = match_candidates(e, l, MatchConfig(mode="iou"))
    assert {(m["earlier"], m["later"]) for m in iou.matches} == {(1, 1)}


def test_unmatched_classification():
    e = [_cand((0, 0, 0))]; l = [_cand((0, 0, 0)), _cand((80, 0, 0))]
    res = match_candidates(e, l)
    st = classify_unmatched(res, e, l, registration_reliable=True)
    assert st == [dict(side="later", index=1, status="potentially_new")]
    st2 = classify_unmatched(res, e, l, registration_reliable=False)
    assert st2[0]["status"] == "uncertain"
    st3 = classify_unmatched(match_candidates(l, e), l, e, True)
    assert st3[0]["status"] == "potentially_resolved"


def test_candidate_peak_rule_filters_weak_components():
    shape = (20, 30, 30)
    s = np.zeros(shape, np.float32); lung = np.ones(shape, np.uint8)
    _blob(s, (6, 15, 8), 3, val=10.0)
    _blob(s, (14, 15, 22), 3, val=4.0)       # above voxel threshold, below the peak rule
    img = _img(shape)
    assert len(extract_candidates(s, lung, img, 3.0, 30.0)[0]) == 2
    kept, _ = extract_candidates(s, lung, img, 3.0, 30.0, min_peak_score=6.0)
    assert len(kept) == 1 and kept[0].max_score == pytest.approx(10.0)
