"""Select the candidate-level filter (minimum component peak score and minimum
volume) on VALIDATION exams (normal + abnormal) by maximising lesion-level F1
(centroid-in-box criterion), keeping the voxel threshold from score_exams.py.
Writes the choice into experiments/<kind>/state.json (and best.pt for flows).

Usage: python scripts/select_candidate_filter.py --kind contextflow
"""
import argparse
import json

import numpy as np
import torch

from _common import ROOT, data_context, load_yaml
from src.evaluation.localization_metrics import detection_report
from src.inference.localization import extract_candidates
from evaluate import gt_lesion_boxes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True)
    a = ap.parse_args()
    _, _, split, store, by = data_context()
    d = ROOT / "experiments" / a.kind
    st = json.loads((d / "state.json").read_text())
    thr = st["threshold"]
    exams = []
    for r in by["val"]:
        eid = r["exam_id"]
        m = np.load(d / "maps" / f"{eid}.npz")["map"].astype(np.float32)
        exams.append((m, store.array(eid, "lung"), store.image(eid), gt_lesion_boxes(store.array(eid, "gtles") > 0, store.image(eid))))
    best, table = None, []
    for mult in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        for vmin in (30.0, 60.0, 100.0, 200.0, 400.0, 600.0, 900.0, 1200.0, 1600.0, 2400.0):
            per = []
            for m, lung, img, refs in exams:
                c, _ = extract_candidates(m, lung, img, thr, vmin, min_peak_score=thr * mult)
                per.append(([dict(bbox_mm=x.bbox_mm, centroid_mm=x.centroid_mm, score=x.max_score) for x in c], refs))
            rep = detection_report(per, 0.0, "centroid")
            table.append(dict(peak_multiplier=mult, min_volume_mm3=vmin, f1=rep["f1"], precision=rep["precision"],
                              recall=rep["recall"]))
            if best is None or rep["f1"] > best["f1"]:
                best = table[-1]
    st.update(candidate_min_peak_score=thr * best["peak_multiplier"], candidate_min_volume_mm3=best["min_volume_mm3"],
              candidate_filter_selection=dict(criterion="max lesion-level F1 (centroid) on validation exams",
                                              chosen=best, grid=table))
    (d / "state.json").write_text(json.dumps(st, indent=1))
    if (d / "best.pt").exists():
        ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
        ck.update(candidate_min_peak_score=st["candidate_min_peak_score"], candidate_min_volume_mm3=best["min_volume_mm3"])
        torch.save(ck, d / "best.pt")
    print(a.kind, "chosen", best)


if __name__ == "__main__":
    main()
