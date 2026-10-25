"""End-to-end on a SYNTHETIC patient: DICOM -> preprocessing -> anomaly map ->
candidates -> registration -> matching -> measurements -> answers.
Uses the intensity baseline so it runs without trained weights."""
import numpy as np

from src.inference.pipeline import Orchestrator
from src.models.anomaly_detector import IntensityBaseline


def test_pipeline_end_to_end(small_patient):
    root, man = small_patient
    det = IntensityBaseline()
    cfg = dict(preprocessing=dict(lung_mask_method="threshold", target_spacing=[3.0, 3.0, 3.0]),
               localization=dict(threshold=0.9, min_volume_mm3=60.0), registration=dict(iterations=150))
    o = Orchestrator(cfg, root=root, load_vqa=False, detector=det)
    det.norm_mean, det.norm_std = 0.0, 0.1         # fixed normalisation for the test
    e1 = o.analyze_dicom(root / man[0]["dicom_dir"], "T1")
    e2 = o.analyze_dicom(root / man[1]["dicom_dir"], "T2")
    assert e1.processed.qc["lung_mask_ok"] and e2.processed.qc["lung_mask_ok"]
    tc = o.temporal(e1, e2)
    assert tc["registration"]["reliable"]
    assert set(tc["counts"]) == {"n_matched", "n_new", "n_resolved", "n_uncertain"}
    a = o.ask("Has the suspected lesion increased or decreased in size since the previous scan?", e2, e1, tc)
    for key in ("question", "earlier_exam", "later_exam", "matched_candidate_id", "measurements",
                "registration_quality", "coordinates", "answer_text", "uncertainty"):
        assert key in a
    assert "clinician review" in a["answer_text"]
    if a["measurements"]:
        vm = a["measurements"]["volume_mm3"]
        if vm["earlier"] and vm["later"]:
            assert np.isclose(vm["percent"], (vm["later"] - vm["earlier"]) / vm["earlier"] * 100)
    w = o.ask("Where is the suspicious region?", e2)
    assert w["task"] == "where" and "answer_text" in w
    u = o.ask("What medication should be prescribed?", e2)
    assert u["status"] == "unsupported"
    t = o.ask("Did the lesion grow?", e2)          # temporal question without a prior exam
    assert t["status"] == "insufficient_evidence"
