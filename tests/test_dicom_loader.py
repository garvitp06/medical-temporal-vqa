import numpy as np
import pydicom
import pytest

from src.data.dicom_loader import DicomValidationError, discover_series, load_dicom_dir, select_series
from src.data.synthetic import write_dicom_series
from pydicom.uid import generate_uid


def _series(tmp, nz=12, spacing=(0.8, 0.7, 2.0), origin=(-10.0, 5.0, -30.0), slope=1.0, intercept=-1024.0, sub="s"):
    hu = np.stack([np.full((6, 5), -1000.0 + 100 * k) for k in range(nz)]).astype(np.float32)
    hu[:, 0, 0] = 40.0
    d = write_dicom_series(hu, spacing, origin, tmp / sub, patient_id="T-1", study_uid=generate_uid(),
                           series_uid=generate_uid(), study_date="20240101", slope=slope, intercept=intercept,
                           shuffle_seed=5)
    return hu, d


def test_sorting_by_position_not_filename(tmp_path):
    hu, d = _series(tmp_path)
    s = load_dicom_dir(d)
    # slices were written with shuffled filenames and InstanceNumbers
    assert np.allclose(s.hu[:, 1, 1], hu[:, 1, 1])
    assert np.allclose(np.diff(s.slice_positions), 2.0)


def test_hu_conversion_uses_rescale(tmp_path):
    hu, d = _series(tmp_path, slope=2.0, intercept=-2048.0)
    s = load_dicom_dir(d)
    assert np.allclose(s.hu, hu, atol=2.0)          # stored = (HU+2048)/2 rounded
    ds = pydicom.dcmread(str(next(d.iterdir())))
    assert ds.BitsAllocated == 16 and ds.pixel_array.dtype == np.uint16


def test_spacing_origin_preserved(tmp_path):
    _, d = _series(tmp_path)
    s = load_dicom_dir(d)
    assert s.spacing == pytest.approx((0.8, 0.7, 2.0))   # (x, y, z)
    assert s.origin == pytest.approx((-10.0, 5.0, -30.0))
    img = s.to_sitk()
    assert img.TransformIndexToPhysicalPoint((1, 2, 3)) == pytest.approx((-10 + 0.8, 5 + 1.4, -30 + 6.0))


def test_missing_slice_detected(tmp_path):
    _, d = _series(tmp_path)
    files = sorted(d.iterdir(), key=lambda f: float(pydicom.dcmread(str(f)).ImagePositionPatient[2]))
    files[3].unlink()
    with pytest.raises(DicomValidationError, match="missing slices"):
        load_dicom_dir(d)
    s = load_dicom_dir(d, allow_gaps=True)
    assert s.warnings


def test_missing_rescale_rejected(tmp_path):
    _, d = _series(tmp_path)
    for f in d.iterdir():
        ds = pydicom.dcmread(str(f)); del ds.RescaleSlope; del ds.RescaleIntercept; ds.save_as(str(f))
    with pytest.raises(DicomValidationError, match="Rescale"):
        load_dicom_dir(d)


def test_corrupted_file_rejected(tmp_path):
    _, d = _series(tmp_path)
    f = next(d.iterdir())
    raw = f.read_bytes(); f.write_bytes(raw[:-40])        # truncated pixel data
    with pytest.raises(DicomValidationError):
        load_dicom_dir(d)


def test_multi_series_not_merged(tmp_path):
    _series(tmp_path, sub="a"); _series(tmp_path, nz=14, sub="b")
    series = discover_series(tmp_path)
    assert len(series) == 2 and sorted(s.n_files for s in series) == [12, 14]
    with pytest.raises(DicomValidationError, match="candidate CT series"):
        select_series(series)
    s = select_series(series, series[0].series_uid)
    assert s.n_files == 14
