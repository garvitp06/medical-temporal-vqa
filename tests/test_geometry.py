import numpy as np
import pytest
import SimpleITK as sitk

from src.data.preprocessing import resample_to_spacing, window_normalize, threshold_lung_mask
from src.inference.localization import index_to_physical, physical_to_index


def _img(arr, spacing=(0.7, 0.7, 3.0), origin=(-50, -40, 10)):
    im = sitk.GetImageFromArray(arr.astype(np.float32)); im.SetSpacing(spacing); im.SetOrigin(origin); return im


def test_index_physical_roundtrip():
    im = _img(np.zeros((10, 20, 30)))
    p = index_to_physical(im, (4, 5, 6))
    assert p == pytest.approx((-50 + 6 * 0.7, -40 + 5 * 0.7, 10 + 4 * 3.0))
    assert physical_to_index(im, p) == pytest.approx((4, 5, 6))


def test_resampling_preserves_physical_position():
    arr = np.zeros((10, 40, 40)); arr[5, 20, 20] = 1000.0
    im = _img(arr)
    r = resample_to_spacing(im, (1.4, 1.4, 1.5))
    a = sitk.GetArrayFromImage(r)
    peak = np.unravel_index(a.argmax(), a.shape)
    assert np.allclose(index_to_physical(r, peak), index_to_physical(im, (5, 20, 20)), atol=1.5)
    assert r.GetSpacing() == pytest.approx((1.4, 1.4, 1.5))


def test_mask_resampling_is_nearest():
    m = np.zeros((10, 20, 20), np.uint8); m[3:7, 5:15, 5:15] = 2
    im = sitk.GetImageFromArray(m); im.SetSpacing((1, 1, 2))
    r = sitk.GetArrayFromImage(resample_to_spacing(im, (0.5, 0.5, 1.0), is_mask=True))
    assert set(np.unique(r)) <= {0, 2}


def test_window_normalize_range():
    x = window_normalize(np.array([-2000.0, -600.0, 2000.0]), -600, 1500, "window_minus1_1")
    assert x.tolist() == pytest.approx([-1.0, 0.0, 1.0])


def test_threshold_lung_mask_includes_nodule():
    hu = np.full((20, 64, 64), -1000.0)
    yy, xx = np.mgrid[:64, :64]
    body = ((yy - 32) / 28) ** 2 + ((xx - 32) / 30) ** 2 <= 1
    hu[:, body] = 40
    for cx in (20, 44):
        lung = ((yy - 32) / 16) ** 2 + ((xx - cx) / 8) ** 2 <= 1
        hu[2:18, lung] = -850
    hu[9:12, 30:33, 18:21] = 30                # nodule inside right lung
    m = threshold_lung_mask(hu, (2.0, 2.0, 2.0))
    assert m[10, 31, 19] == 1                  # solid nodule kept inside lung mask
    assert m[10, 32, 32] == 0                  # mediastinum excluded
