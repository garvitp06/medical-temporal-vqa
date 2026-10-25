"""Calibrate a detector on VALIDATION patients and score every exam.

1. norm stats  <- raw lung-voxel scores of validation NORMAL exams
2. threshold   <- grid value maximising mean Dice on validation ABNORMAL exams
3. writes normalised maps for all exams to experiments/<kind>/maps/<exam>.npz
   (float16) and the calibration into the checkpoint / state file.

Usage: python scripts/score_exams.py --kind {intensity_baseline,glow,contextflow}
"""
import argparse
import json
import time

import numpy as np
import torch

from _common import ROOT, data_context, load_yaml
from src.evaluation.anomaly_metrics import best_threshold
from src.models.anomaly_detector import IntensityBaseline, load_detector


def meta_of(store, eid):
    m = store.meta(eid)
    return {"original_slice_spacing": float(m["original_meta"]["original_spacing"][2]),
            "hu": store.array(eid, "hu")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--eval-config", default="configs/evaluation.yaml")
    ap.add_argument("--resume", action="store_true", help="reuse maps written by an interrupted run with identical calibration")
    ap.add_argument("--splits", default="train,val,test", help="which patient splits to score (val is always scored)")
    a = ap.parse_args()
    ev = load_yaml(ROOT / a.eval_config)
    _, manifest, split, store, by = data_context()
    out = ROOT / "experiments" / a.kind
    (out / "maps").mkdir(parents=True, exist_ok=True)
    if a.kind == "intensity_baseline":
        det = IntensityBaseline()
    else:
        det = load_detector(a.kind, str(out / "best.pt"), root=ROOT)
    val_norm = [(store.array(e["exam_id"], "norm"), store.array(e["exam_id"], "lung"), meta_of(store, e["exam_id"]))
                for e in by["val"] if not e["abnormal"]]
    t0 = time.time()
    cal = det.calibrate(val_norm)
    cal_f = out / "calibration_cache.json"
    prev_cal = json.loads(cal_f.read_text()) if cal_f.exists() else None
    cal_f.write_text(json.dumps(cal))
    print("calibration", cal, f"{time.time() - t0:.1f}s", flush=True)
    timings = {}
    val_abn = []
    want = set(a.splits.split(",")) | {"val"}
    for r in manifest:
        if split[r["patient_id"]] not in want:
            continue
        eid = r["exam_id"]
        norm = store.array(eid, "norm"); lung = store.array(eid, "lung")
        f = out / "maps" / f"{eid}.npz"
        if a.resume and f.exists() and prev_cal == cal:
            d = np.load(f)
            o = type("O", (), dict(map=d["map"].astype(np.float32), volume_score=float(d["volume_score"])))
            print(f"{eid} reused", flush=True)
            if split[r["patient_id"]] == "val" and r["abnormal"]:
                val_abn.append((o.map, store.array(eid, "gtles") > 0, lung))
            continue
        t = time.time()
        o = det.score(norm, lung, meta_of(store, eid))
        timings[eid] = round(time.time() - t, 2)
        np.savez_compressed(f, map=o.map.astype(np.float16),
                            slice_scores=o.slice_scores.astype(np.float32), volume_score=o.volume_score)
        if split[r["patient_id"]] == "val" and r["abnormal"]:
            gt = store.array(eid, "gtles")
            val_abn.append((o.map, gt > 0, lung))
        print(f"{eid} {timings[eid]}s vol_score={o.volume_score:.2f}", flush=True)
    thr, val_dice = best_threshold([v[0] for v in val_abn], [v[1] for v in val_abn], [v[2] for v in val_abn],
                                   ev["threshold_grid"])
    print(f"selected threshold {thr} (val mean Dice {val_dice:.3f})")
    state = dict(det.state(), threshold=thr, val_mean_dice_at_threshold=val_dice, calibration=cal,
                 threshold_selection="max mean per-exam voxel Dice on validation abnormal exams",
                 mean_seconds_per_exam=float(np.mean(list(timings.values()))) if timings else None, n_scored=len(timings))
    (out / "state.json").write_text(json.dumps(state, indent=1))
    if a.kind != "intensity_baseline":
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        ck.update(norm_mean=det.norm_mean, norm_std=det.norm_std, threshold=thr)
        torch.save(ck, out / "best.pt")


if __name__ == "__main__":
    main()
