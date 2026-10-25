import numpy as np
import pytest
import SimpleITK as sitk

from src.data.preprocessing import PreprocConfig, preprocess_series
from src.data.registration import RegistrationConfig, identity_pair, register_pair
from src.data.dicom_loader import load_dicom_dir


def test_rigid_registration_recovers_simulated_motion(small_patient):
    root, man = small_patient
    cfg = PreprocConfig(lung_mask_method="threshold", target_spacing=(3.0, 3.0, 3.0))
    e1, e2 = [preprocess_series(load_dicom_dir(root / r["dicom_dir"]), cfg) for r in man]
    res = register_pair(e2.image, e1.image, e2.lung_mask, e1.lung_mask, RegistrationConfig(iterations=150))
    rep = res.report
    assert rep["lung_dice_after"] > rep["lung_dice_before"]
    assert rep["reliable"] and rep["quality"] == "acceptable", rep["failure_reasons"]
    # target registration error on the simulated anatomy: map a later-exam point back to the earlier exam
    from src.data.synthetic import rigid_matrix
    R, t = rigid_matrix(man[1]["sim_rotation_deg"], man[1]["sim_translation_mm"])
    a = np.array([30.0, 10.0, 20.0])                # anatomy-frame point
    p_later = R @ a + t; p_earlier = a              # exam 1 has identity pose
    mapped = np.array(res.transform.TransformPoint(tuple(p_later)))
    assert np.linalg.norm(mapped - p_earlier) < 4.0          # mm, at 3 mm voxels
    assert res.moving_resampled.GetSize() == e2.image.GetSize()
    ident = identity_pair(e2.image, e1.image, e2.lung_mask, e1.lung_mask)
    assert not ident.reliable and ident.report["quality"] == "not_registered"
