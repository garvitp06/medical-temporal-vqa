"""Register every temporal pair (earlier -> later) once and cache the result.

Writes experiments/registration/<patient>.tfm (transform), <patient>.json
(quality report incl. simulated ground-truth motion for synthetic data) and a
summary with failure rates.
Usage: python scripts/register_pairs.py [--config configs/registration.yaml]
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import SimpleITK as sitk

from _common import ROOT, data_context, load_yaml
from src.data.registration import RegistrationConfig, register_pair


def pairs_of(manifest):
    byp = defaultdict(dict)
    for r in manifest:
        byp[r["patient_id"]][r["timepoint"]] = r
    return {p: (d[1], d[2]) for p, d in byp.items() if 1 in d and 2 in d}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/registration.yaml")
    a = ap.parse_args()
    cfg = RegistrationConfig.from_dict(load_yaml(ROOT / a.config))
    _, manifest, split, store, _ = data_context()
    out = ROOT / "experiments/registration"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, (e1, e2) in sorted(pairs_of(manifest).items()):
        t0 = time.time()
        fixed, moving = store.image(e2["exam_id"]), store.image(e1["exam_id"])
        res = register_pair(fixed, moving, store.array(e2["exam_id"], "lung"), store.array(e1["exam_id"], "lung"), cfg)
        sitk.WriteTransform(res.transform, str(out / f"{pid}.tfm"))
        rep = dict(res.report, patient_id=pid, split=split[pid], earlier=e1["exam_id"], later=e2["exam_id"],
                   seconds=round(time.time() - t0, 1),
                   simulated_motion=dict(rotation_deg=e2.get("sim_rotation_deg"), translation_mm=e2.get("sim_translation_mm")))
        # Ground-truth check (synthetic only): map lesion centres of the earlier exam into later space
        (out / f"{pid}.json").write_text(json.dumps(rep, indent=1))
        rows.append(rep)
        print(pid, rep["quality"], f"dice {rep['lung_dice_before']:.3f}->{rep['lung_dice_after']:.3f}",
              f"ncc {rep['ncc_lung_before']:.3f}->{rep['ncc_lung_after']:.3f}", f"{rep['seconds']}s", flush=True)
    summ = dict(n_pairs=len(rows), n_reliable=sum(r["reliable"] for r in rows),
                failure_rate=float(np.mean([not r["reliable"] for r in rows])) if rows else None,
                lung_dice_after_mean=float(np.mean([r["lung_dice_after"] for r in rows])),
                lung_dice_before_mean=float(np.mean([r["lung_dice_before"] for r in rows])),
                config=cfg.__dict__)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
