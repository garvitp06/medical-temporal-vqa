"""Generate (optional) synthetic cohort, split patients, preprocess every exam.

Usage: python scripts/prepare_data.py --config configs/data.yaml [--workers 2] [--limit N]
Outputs: data/processed/<exam>_{hu,norm,lung,gtles}.nii.gz + _preproc.json,
         data/splits/patient_split.json, reports/data_report.json
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402

from src.data.dataset import assert_no_patient_leakage, load_manifest, patient_level_split, split_summary  # noqa: E402
from src.data.dicom_loader import load_dicom_dir  # noqa: E402
from src.data.preprocessing import PreprocConfig, preprocess_series, resample_like  # noqa: E402
from src.training.utils import load_yaml  # noqa: E402


def process_one(args):
    import torch
    torch.set_num_threads(1)
    rec, raw_root, out_root, pre_cfg = args
    t0 = time.time()
    eid = rec["exam_id"]
    if (out_root / f"{eid}_preproc.json").exists():
        return eid, "cached", 0.0, None
    try:
        loaded = load_dicom_dir(raw_root / rec["dicom_dir"])
        p = preprocess_series(loaded, PreprocConfig.from_dict(pre_cfg))
        p.save(out_root, eid)
        if rec.get("lesion_mask"):
            gt = sitk.ReadImage(str(raw_root / rec["lesion_mask"]))
            gt.SetDirection(loaded.to_sitk().GetDirection())
            g2 = resample_like(gt, p.image, is_mask=True)
            sitk.WriteImage(sitk.Cast(g2, sitk.sitkUInt8), str(out_root / f"{eid}_gtles.nii.gz"))
            ga = sitk.GetArrayFromImage(g2) > 0
            p.qc["gt_lesion_voxels"] = int(ga.sum())
            p.qc["gt_lesion_fraction_inside_lung_mask"] = float((ga & (p.lung_mask > 0)).sum() / ga.sum()) if ga.any() else None
            p.save(out_root, eid)
        return eid, "ok", time.time() - t0, p.qc
    except Exception as exc:
        return eid, f"failed: {type(exc).__name__}: {exc}", time.time() - t0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--preproc", default="configs/preprocessing.yaml")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / a.config)
    pre = load_yaml(ROOT / a.preproc)
    pre["lungmask_weights"] = str(ROOT / pre["lungmask_weights"])
    raw = ROOT / cfg["raw_root"]
    if cfg.get("synthetic", {}).get("generate") and not (ROOT / cfg["manifest"]).exists():
        from src.data.synthetic import generate_cohort
        s = cfg["synthetic"]
        print(f"generating SYNTHETIC cohort: {s['n_patients']} patients -> {raw}")
        generate_cohort(raw, s["n_patients"], s["frac_abnormal"], s["seed"], s["image_size_xy"])
    manifest = load_manifest(ROOT / cfg["manifest"])
    if a.limit:
        manifest = manifest[: a.limit]
    sp = cfg["split"]
    split = patient_level_split(manifest, tuple(sp["fractions"]), sp["seed"], sp["stratify_key"],
                                sp.get("held_out_key"), sp.get("held_out_values"))
    assert_no_patient_leakage(split, manifest)
    sf = ROOT / cfg["split_file"]; sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(split, indent=1))
    out = ROOT / cfg["processed_root"]; out.mkdir(parents=True, exist_ok=True)
    jobs = [(r, raw, out, pre) for r in manifest]
    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        for eid, status, dt, qc in ex.map(process_one, jobs):
            results.append(dict(exam_id=eid, status=status, seconds=round(dt, 1), qc=qc))
            print(f"{eid}: {status} ({dt:.1f}s)" + (f" lung={qc['lung_volume_ml']:.0f}ml" if qc else ""), flush=True)
    qa = json.loads((ROOT / cfg["qa_pairs"]).read_text()) if (ROOT / cfg["qa_pairs"]).exists() else None
    les = json.loads((ROOT / cfg["lesions"]).read_text()) if (ROOT / cfg["lesions"]).exists() else None
    failed = [r for r in results if not (r["status"] in ("ok", "cached"))]
    report = dict(dataset=cfg["dataset_name"], synthetic=bool(cfg.get("synthetic", {}).get("generate")),
                  n_exams=len(manifest), split_summary=split_summary(manifest, split, qa, les),
                  failed_or_excluded=failed, preprocessing=pre, per_exam=results)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/data_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["split_summary"], indent=1))
    print(f"failed/excluded: {len(failed)}")


if __name__ == "__main__":
    main()
