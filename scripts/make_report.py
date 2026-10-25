"""Render reports/results/results.json into reports/RESULTS.md (tables are
generated from the JSON; nothing is hand-entered)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f(v, d=3):
    if v is None:
        return "—"
    try:
        if v != v:
            return "n/a"
        return f"{v:.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def fci(c, d=3):
    if not c:
        return "—"
    return f"{f(c['point'], d)} [{f(c['ci95'][0], d)}, {f(c['ci95'][1], d)}]"


def write_markdown(r, out: Path):
    L = []
    w = L.append
    w("# Results — SYNTHETIC evaluation\n")
    w("> All numbers below were computed by `scripts/evaluate.py` from saved predictions on the **held-out test "
      "patients of a SYNTHETIC phantom cohort**. They demonstrate that the pipeline and metrics work; they say "
      "nothing about clinical performance.\n")
    w(f"Evaluation set: {r['evaluation_set']} — {r['n_test_patients']} patients, {r['n_test_exams']} exams, "
      f"{r['n_test_pairs']} temporal pairs. 95% CIs are patient-level bootstrap (2000 resamples).\n")
    w("## 1. Anomaly detection and localisation\n")
    w("| Detector | Voxel AUROC | Voxel AP | Exam AUROC | Exam AP | Dice mean [95% CI] | Dice median | Sensitivity | Precision | Voxel thr / cand. rule (val) | s/exam |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, a in r["anomaly"].items():
        s = a["segmentation"]
        w(f"| {k} | {f(a['voxel_level']['auroc'])} | {f(a['voxel_level']['average_precision'])} | "
          f"{f(a['exam_level']['auroc'])} | {f(a['exam_level']['average_precision'])} | {fci(s['dice_ci'])} | "
          f"{f(s['dice_median'])} | {f(s['sensitivity_mean'])} | {f(s['precision_mean'])} | {a['threshold']} / "
          f"≥{f(a['candidate_rule']['min_volume_mm3'], 0)} mm³, peak ≥{f(a['candidate_rule']['min_peak_score'], 1)} | "
          f"{f(a['runtime_s_per_exam'], 1)} |")
    w("\nVoxel-level: lung voxels (all lesion voxels + ≤15k random lung voxels per test exam). Exam-level: volume score "
      "vs. abnormal label. Segmentation: voxel-level per abnormal test exam at the validation-selected threshold.\n")
    w("**Lesion-level 3D detection** (greedy matching by score)\n")
    w("| Detector | Criterion | Ref lesions | TP | FP | FN | Precision | Recall | F1 | AP | Mean IoU (TP) | FP / normal exam |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, a in r["anomaly"].items():
        for key, name in (("lesion_detection_iou", "3D IoU ≥ 0.1"), ("lesion_detection_centroid", "centroid in box")):
            d = a[key]
            w(f"| {k} | {name} | {d['n_reference_lesions']} | {d['tp']} | {d['fp']} | {d['fn']} | {f(d['precision'])} | "
              f"{f(d['recall'])} | {f(d['f1'])} | {f(d['average_precision'])} | {f(d['mean_iou_matched'])} | "
              f"{f(a['false_positives_per_exam'], 2)} |")
    g = r["registration"]
    w("\n## 2. Registration (test pairs)\n")
    w("| Pairs | Failure rate | Lung Dice before → after | Lung NCC before → after | TRE mean / median (mm) | Displacement without registration (mm) |")
    w("|---|---|---|---|---|---|")
    w(f"| {g['n_pairs']} | {f(g['failure_rate'])} | {f(g['lung_dice_before'])} → {f(g['lung_dice_after'])} | "
      f"{f(g['ncc_before'])} → {f(g['ncc_after'])} | {f(g['tre_mm_mean'], 2)} / {f(g['tre_mm_median'], 2)} | "
      f"{f(g['displacement_without_registration_mm_mean'], 1)} |")
    w(f"\nTRE uses {g['n_landmarks']} synthetic lesion-centre landmarks whose true positions are known by construction.\n")
    w("## 3. Temporal comparison (lesion level, test pairs)\n")
    w("| Configuration | Ref lesions | Associated | Association acc. | Change-category acc. | Category acc. as reported* | Later-vol MAE (mm³) | Later-vol MAPE (%) | ΔV MAE (mm³) | %Δ MAE / median (pp) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for k, t in r["temporal"].items():
        w(f"| {k} | {t['n_reference_lesions']} | {t['n_associated']} | {f(t['association_accuracy'])} | "
          f"{f(t['change_category']['accuracy'])} | {f(t['change_category_as_reported']['accuracy'])} | "
          f"{f(t['volume_later_errors_mm3']['mae'], 0)} | {f(t['volume_later_errors_mm3']['mape'], 1)} | "
          f"{f(t['volume_change_errors_mm3']['mae'], 0)} | {f(t['percent_change_errors']['mae_percentage_points'], 1)} / "
          f"{f(t['percent_change_errors'].get('median_abs_error_percentage_points'), 1)} |")
    w("\n*As reported: pairs with unreliable registration are reported as 'undetermined' (counted wrong). "
      "Volume errors compare candidate-mask volumes with reference lesion-mask volumes for lesions detected at both "
      "timepoints; single-measurement errors and change errors are listed separately.\n")
    if r["temporal"]:
        k0 = next(iter(r["temporal"]))
        conf = r["temporal"][k0]["change_category"]["confusion_ref_rows_pred_cols"]
        labs = list(conf)
        w(f"Confusion matrix ({k0}; rows = reference, columns = predicted):\n")
        w("| ref \\ pred | " + " | ".join(labs) + " |"); w("|---" * (len(labs) + 1) + "|")
        for a_ in labs:
            w(f"| {a_} | " + " | ".join(str(conf[a_][b]) for b in labs) + " |")
    w("\n## 4. Visual question answering (test questions)\n")
    tasks = sorted({t for v in r["vqa"].values() for t in v["per_task"]})
    w("| Task | " + " | ".join(f"{k} acc [95% CI]" for k in r["vqa"]) + " | candidate-derived acc | n |")
    w("|---" * (len(r["vqa"]) + 3) + "|")
    for t in tasks:
        cells = [fci(v["per_task"].get(t, {}).get("accuracy_ci")) for v in r["vqa"].values()]
        cd = r["candidate_derived_answers"].get(t, {}).get("accuracy")
        n = next((v["per_task"][t]["n"] for v in r["vqa"].values() if t in v["per_task"]), "—")
        w(f"| {t} | " + " | ".join(cells) + f" | {f(cd)} | {n} |")
    w("| **overall** | " + " | ".join(fci(v["overall_accuracy"]) for v in r["vqa"].values()) + " | | |")
    w("\nMacro-F1 and region-token hit rate per task:\n")
    w("| Model | Task | Macro-F1 | Region hit |"); w("|---|---|---|---|")
    for k, v in r["vqa"].items():
        for t, d in v["per_task"].items():
            w(f"| {k} | {t} | {f(d['macro_f1'])} | {f(d.get('region_token_hit_rate'))} |")
    w("\n**Calibration** (temperature fitted on validation):\n")
    w("| Model | Head | ECE raw → scaled | Brier raw → scaled | T | n |"); w("|---|---|---|---|---|---|")
    for k, v in r["vqa"].items():
        for h in ("single", "temporal"):
            c = v[f"calibration_{h}"]
            w(f"| {k} | {h} | {f(c['ece_raw'])} → {f(c['ece_temperature_scaled'])} | {f(c['brier_raw'])} → "
              f"{f(c['brier_temperature_scaled'])} | {f(c['temperature'], 2)} | {c['n']} |")
    w("\n**Question router on held-out paraphrases:** " + ", ".join(
        f"{k}: {f(v['router_heldout_paraphrases']['accuracy'])} (n={v['router_heldout_paraphrases']['n']})"
        for k, v in r["vqa"].items()) + "\n")
    w(f"Evaluation runtime {r['eval_runtime_s']} s; peak RSS {f(r['peak_rss_mb'], 0)} MB.\n")
    out.write_text("\n".join(L))


if __name__ == "__main__":
    res = json.loads((ROOT / "reports/results/results.json").read_text())
    write_markdown(res, ROOT / "reports/RESULTS.md")
