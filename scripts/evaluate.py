"""Evaluate every supported capability on the held-out TEST patients and write
reports/results/*.json plus reports/RESULTS.md. All numbers are computed from
saved predictions and reference labels; nothing is typed in by hand.

Usage: python scripts/evaluate.py [--detectors intensity_baseline glow contextflow]
"""
import argparse
import json
import resource
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage as ndi

from _common import ROOT, data_context, load_yaml
from src.data.preprocessing import ProcessedExam
from src.data.registration import dice as reg_dice
from src.evaluation import anomaly_metrics as am
from src.evaluation.localization_metrics import detection_report
from src.evaluation.temporal_metrics import category_agreement, change_errors, measurement_errors
from src.evaluation.vqa_metrics import (brier_multiclass, ece, exact_match, group_by_patient, macro_f1,
                                        patient_bootstrap_ci)
from src.inference.localization import extract_candidates, index_to_physical
from src.inference.pipeline import ExamAnalysis, Orchestrator
from src.models.anomaly_detector import AnomalyOutput, IntensityBaseline
from src.temporal.candidate_matching import MatchConfig
from src.temporal.measurements import categorize_change

RES = ROOT / "reports/results"


def ci(vals_by_patient, cfg):
    p, lo, hi = patient_bootstrap_ci(vals_by_patient, np.mean, cfg["n_boot"], cfg["alpha"], cfg["seed"])
    return dict(point=p, ci95=[lo, hi])


def finite(v):
    return [x for x in v if x is not None and np.isfinite(x)]


def gt_lesion_boxes(gt, img):
    lab, n = ndi.label(gt > 0, structure=np.ones((3, 3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        z0, y0, x0 = [s.start for s in sl]; z1, y1, x1 = [s.stop for s in sl]
        cs = np.array([index_to_physical(img, (z - .5, y - .5, x - .5)) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)])
        out.append(dict(bbox_mm=cs.min(0).tolist() + cs.max(0).tolist(), n_voxels=int((lab[sl] == i).sum())))
    return out


def cand_rule(kind):
    st = json.loads((ROOT / "experiments" / kind / "state.json").read_text())
    return st["threshold"], st.get("candidate_min_volume_mm3", 30.0), st.get("candidate_min_peak_score")


def load_analysis(store, kind, eid, threshold, min_vol, min_peak=None):
    img = store.image(eid); lung = store.array(eid, "lung"); norm = store.array(eid, "norm")
    d = np.load(ROOT / "experiments" / kind / "maps" / f"{eid}.npz")
    amap = d["map"].astype(np.float32)
    meta = store.meta(eid)
    p = ProcessedExam(img, norm, lung, meta["original_meta"], meta["qc"], meta["ops"])
    out = AnomalyOutput(None, amap, d["slice_scores"], float(d["volume_score"]), kind)
    cands, cmask = extract_candidates(amap, lung, img, threshold, min_vol, min_peak_score=min_peak)
    hu = sitk.GetArrayFromImage(img)
    cd = []
    for k, c in enumerate(cands, start=1):
        dd = c.to_dict(); dd["mean_hu"] = float(hu[cmask == k].mean()); cd.append(dd)
    return ExamAnalysis(eid, p, out, cd, cmask)


# --------------------------------------------------------------------------- anomaly
def eval_anomaly(kind, test, store, ev):
    st = json.loads((ROOT / "experiments" / kind / "state.json").read_text())
    thr, cvol, cpeak = cand_rule(kind)
    seg_by_pat, vox_s, vox_y, ex_s, ex_y, det = defaultdict(list), [], [], [], [], []
    seg_rows = []
    for r in test:
        eid = r["exam_id"]
        lung = store.array(eid, "lung"); gt = store.array(eid, "gtles") > 0
        amap = np.load(ROOT / "experiments" / kind / "maps" / f"{eid}.npz")
        m = amap["map"].astype(np.float32)
        s, y = am.subsample_voxels(m, gt, lung, 15000, seed=zlib.crc32(eid.encode()) % 1000)
        vox_s.append(s); vox_y.append(y)
        ex_s.append(float(amap["volume_score"])); ex_y.append(int(gt.any()))
        img = store.image(eid)
        cands, _ = extract_candidates(m, lung, img, thr, cvol, min_peak_score=cpeak)
        preds = [dict(bbox_mm=c.bbox_mm, centroid_mm=c.centroid_mm, score=c.max_score) for c in cands]
        det.append((preds, gt_lesion_boxes(gt, img)))
        if gt.any():
            rep = am.segmentation_report((m >= thr) & (lung > 0), gt)
            seg_by_pat[r["patient_id"]].append(rep["dice"])
            seg_rows.append(dict(exam_id=eid, **rep))
    au, ap = am.auroc_ap(np.concatenate(vox_s), np.concatenate(vox_y))
    eau, eap = am.auroc_ap(ex_s, ex_y)
    dices = [x["dice"] for x in seg_rows]
    bs = load_yaml(ROOT / "configs/evaluation.yaml")["bootstrap"]
    return dict(
        detector=kind, threshold=thr, threshold_selected_on="validation", n_test_exams=len(test),
        candidate_rule=dict(min_volume_mm3=cvol, min_peak_score=cpeak, selected_on="validation lesion-level F1"),
        n_test_abnormal_exams=sum(ex_y),
        voxel_level=dict(unit="lung voxels (all lesion voxels + <=15k random lung voxels per exam)",
                         auroc=au, average_precision=ap, n_voxels=int(sum(len(v) for v in vox_y)),
                         positive_fraction=float(np.mean(np.concatenate(vox_y)))),
        exam_level=dict(unit="examination (volume score = mean of top 0.1% voxel scores)", auroc=eau,
                        average_precision=eap),
        segmentation=dict(unit="per abnormal exam, voxel-level at val threshold; patient-level bootstrap CI",
                          dice_mean=float(np.mean(dices)) if dices else None,
                          dice_median=float(np.median(dices)) if dices else None,
                          dice_ci=ci(seg_by_pat, bs) if seg_by_pat else None,
                          iou_mean=float(np.mean([x["iou"] for x in seg_rows])) if seg_rows else None,
                          sensitivity_mean=float(np.nanmean([x["sensitivity"] for x in seg_rows])) if seg_rows else None,
                          precision_mean=float(np.mean([x["precision"] for x in seg_rows])) if seg_rows else None,
                          per_exam=seg_rows),
        lesion_detection_iou=detection_report(det, ev["detection_iou_threshold"], "iou"),
        lesion_detection_centroid=detection_report(det, 0.0, "centroid"),
        false_positives_per_exam=float(np.mean([sum(1 for p in preds) for preds, refs in det if not refs]))
        if any(not refs for _, refs in det) else None,
        runtime_s_per_exam=st.get("mean_seconds_per_exam"))


# --------------------------------------------------------------------------- registration
def eval_registration(pairs, lesions_by_exam):
    rows, tres = [], []
    for pid, (e1, e2) in pairs.items():
        rep = json.loads((ROOT / "experiments/registration" / f"{pid}.json").read_text())
        tfm = sitk.ReadTransform(str(ROOT / "experiments/registration" / f"{pid}.tfm"))
        # TRE on synthetic anatomical points (lesion centres are defined for every lesion at both times)
        l1 = {l["lesion_id"]: l["center_world_mm"] for l in lesions_by_exam[e1["exam_id"]]}
        for l in lesions_by_exam[e2["exam_id"]]:
            if l["lesion_id"] in l1:
                mapped = np.array(tfm.TransformPoint(tuple(l["center_world_mm"])))
                tres.append(float(np.linalg.norm(mapped - np.array(l1[l["lesion_id"]]))))
                unreg = float(np.linalg.norm(np.array(l["center_world_mm"]) - np.array(l1[l["lesion_id"]])))
                rows.append(dict(patient_id=pid, tre_mm=tres[-1], displacement_without_registration_mm=unreg))
    reps = [json.loads((ROOT / "experiments/registration" / f"{p}.json").read_text()) for p in pairs]
    return dict(n_pairs=len(reps), failure_rate=float(np.mean([not r["reliable"] for r in reps])),
                lung_dice_before=float(np.mean([r["lung_dice_before"] for r in reps])),
                lung_dice_after=float(np.mean([r["lung_dice_after"] for r in reps])),
                ncc_before=float(np.mean([r["ncc_lung_before"] for r in reps])),
                ncc_after=float(np.mean([r["ncc_lung_after"] for r in reps])),
                tre_mm_mean=float(np.mean(tres)) if tres else None,
                tre_mm_median=float(np.median(tres)) if tres else None,
                displacement_without_registration_mm_mean=float(np.mean([r["displacement_without_registration_mm"] for r in rows])) if rows else None,
                n_landmarks=len(tres), unit="synthetic lesion-centre landmarks (pair level)")


# --------------------------------------------------------------------------- temporal
def eval_temporal(kind, pairs, store, lesions_by_exam, ev, register=True, match_mode="combined"):
    thr, cvol, cpeak = cand_rule(kind)
    orch = Orchestrator(dict(localization=dict(threshold=thr, min_volume_mm3=cvol, min_peak_score=cpeak),
                             matching=dict(mode=match_mode), registration=load_yaml(ROOT / "configs/registration.yaml")),
                        root=ROOT, load_vqa=False, detector=IntensityBaseline())
    orch.threshold = thr
    rows = []
    for pid, (e1, e2) in pairs.items():
        a1 = load_analysis(store, kind, e1["exam_id"], thr, cvol, cpeak)
        a2 = load_analysis(store, kind, e2["exam_id"], thr, cvol, cpeak)
        tc = orch.temporal(a1, a2, register=register)
        img2 = a2.processed.image; sp = img2.GetSpacing()
        gt2 = store.array(e2["exam_id"], "gtles")
        # reference volumes are measured on each exam's own processed grid (no interpolation)
        g1_native = store.array(e1["exam_id"], "gtles")
        vox1 = float(np.prod(store.image(e1["exam_id"]).GetSpacing())); vox2 = float(np.prod(sp))
        for l2 in lesions_by_exam[e2["exam_id"]]:
            lid = l2["lesion_id"]; k = l2["label"]
            ref_v1 = float((g1_native == k).sum()) * vox1
            ref_v2 = float((gt2 == k).sum()) * vox2
            c_later = np.array(l2["center_world_mm"])
            best, bd = None, 1e9
            for ent in tc["entities"]:
                for side in ("later", "earlier"):
                    c = ent[side]
                    if c is None:
                        continue
                    d = float(np.linalg.norm(np.array(c["centroid_mm"]) - c_later))
                    if d < bd:
                        bd, best = d, ent
            assoc = best if bd <= 15.0 else None
            if assoc is not None and assoc["measurements"]:
                vm = assoc["measurements"]["volume_mm3"]
                pred_cat = categorize_change(vm["percent"], vm["case"], ev["stable_band_pct"])
                pv1, pv2 = vm["earlier"] or 0.0, vm["later"] or 0.0
            else:
                pred_cat, pv1, pv2 = "none", 0.0, 0.0
            if not tc["registration"].get("reliable"):
                pred_cat_reported = "undetermined"
            else:
                pred_cat_reported = pred_cat
            both = ref_v1 > 0 and ref_v2 > 0
            correct_assoc = (assoc is not None and
                             ((both and assoc["match_status"] == "matched") or
                              (ref_v1 == 0 and assoc["match_status"] in ("potentially_new",)) or
                              (ref_v2 == 0 and assoc["match_status"] in ("potentially_resolved",))))
            rows.append(dict(patient_id=pid, lesion_id=lid, ref_change=l2["change"], pred_change=pred_cat,
                             pred_change_reported=pred_cat_reported, registration_reliable=tc["registration"].get("reliable"),
                             ref_v1=ref_v1, ref_v2=ref_v2, pred_v1=pv1, pred_v2=pv2, associated=assoc is not None,
                             association_distance_mm=bd, match_status=assoc["match_status"] if assoc else None,
                             association_correct=bool(correct_assoc),
                             ref_pct=((ref_v2 - ref_v1) / ref_v1 * 100) if ref_v1 > 0 else None,
                             pred_pct=assoc["measurements"]["volume_mm3"]["percent"] if assoc and assoc["measurements"] else None))
        rows.append(dict(patient_id=pid, pair_summary=True, counts=tc["counts"],
                         registration_quality=tc["registration"].get("quality")))
    les = [r for r in rows if not r.get("pair_summary")]
    det = [r for r in les if r["associated"]]
    both = [r for r in det if r["ref_v1"] > 0 and r["ref_v2"] > 0 and r["pred_v1"] > 0 and r["pred_v2"] > 0]
    pct = [r for r in both if r["pred_pct"] is not None and r["ref_pct"] is not None]
    mape_min = ev["mape_min_abs_ref_mm3"]
    out = dict(detector=kind, registration=register, matching_mode=match_mode, n_pairs=len(pairs),
               n_reference_lesions=len(les), n_associated=len(det),
               change_category=category_agreement([r["pred_change"] for r in les], [r["ref_change"] for r in les]),
               change_category_as_reported=category_agreement([r["pred_change_reported"] for r in les],
                                                              [r["ref_change"] for r in les]),
               association_accuracy=float(np.mean([r["association_correct"] for r in les])) if les else None,
               volume_later_errors_mm3=measurement_errors([r["pred_v2"] for r in both], [r["ref_v2"] for r in both], mape_min),
               volume_earlier_errors_mm3=measurement_errors([r["pred_v1"] for r in both], [r["ref_v1"] for r in both], mape_min),
               volume_change_errors_mm3=change_errors([r["pred_v1"] for r in both], [r["pred_v2"] for r in both],
                                                      [r["ref_v1"] for r in both], [r["ref_v2"] for r in both]),
               percent_change_errors=dict(n=len(pct), mae_percentage_points=float(np.mean([abs(r["pred_pct"] - r["ref_pct"]) for r in pct])) if pct else None,
                                          median_abs_error_percentage_points=float(np.median([abs(r["pred_pct"] - r["ref_pct"]) for r in pct])) if pct else None),
               rows=rows)
    return out


# --------------------------------------------------------------------------- VQA
def eval_vqa(tag, qa, split, store, manifest, lesions, bs):
    from train_vqa import norm_answer, predict
    from src.models.vqa import (ANSWERS, HELDOUT_PARAPHRASES, TASKS, QuestionRouter, Tokenizer, VQAModel)
    from src.training.vqa_data import VQATensorCache, build_items
    ck = torch.load(ROOT / "experiments/vqa" / f"{tag}.pt", map_location="cpu", weights_only=False)
    tok = Tokenizer(vocab=ck["vocab"])
    cache = VQATensorCache(ROOT, store, manifest, lesions, ck["detector"], use_anomaly=ck["use_anomaly"])
    items = build_items(qa, split, "test", cache, manifest)
    res = dict(model=tag, use_anomaly=ck["use_anomaly"], detector=ck["detector"], n_test_items=len(items))
    per_task, all_rows = {}, []
    inch = 3 if ck["use_anomaly"] else 2
    for temporal in (False, True):
        name = "temporal" if temporal else "single"
        m = VQAModel(tok, inch, temporal=temporal, width=ck.get("width", 32)); m.load_state_dict(ck[name])
        its = [i for i in items if (i["scope"] == "temporal") == temporal]
        lg, rg = predict(m, tok, its, cache, temporal)
        T = ck[f"temperature_{name}"]
        p_raw = torch.softmax(lg, 1).numpy(); p_cal = torch.softmax(lg / T, 1).numpy()
        y = np.array([ANSWERS.index(norm_answer(i)) for i in its])
        pred = p_raw.argmax(1)
        for k, it in enumerate(its):
            row = dict(patient_id=it["patient_id"], task=it["task"], ref=norm_answer(it), pred=ANSWERS[pred[k]],
                       correct=float(pred[k] == y[k]), conf_raw=float(p_raw[k].max()), conf_cal=float(p_cal[k].max()))
            if it["region"] >= 0 and rg is not None:
                row["region_hit"] = float(int(rg[k].argmax()) == it["region"])
            all_rows.append(row)
        e_raw, _ = ece(p_raw, y); e_cal, table = ece(p_cal, y)
        res[f"calibration_{name}"] = dict(ece_raw=e_raw, ece_temperature_scaled=e_cal, temperature=T,
                                          brier_raw=brier_multiclass(p_raw, y), brier_temperature_scaled=brier_multiclass(p_cal, y),
                                          reliability_bins=table, n=len(y))
    for t in sorted({r["task"] for r in all_rows}):
        rr = [r for r in all_rows if r["task"] == t]
        per_task[t] = dict(n=len(rr), exact_match=exact_match([r["pred"] for r in rr], [r["ref"] for r in rr]),
                           macro_f1=macro_f1([r["pred"] for r in rr], [r["ref"] for r in rr]),
                           accuracy_ci=ci(group_by_patient(rr), bs))
        hits = [r["region_hit"] for r in rr if "region_hit" in r]
        if hits:
            per_task[t]["region_token_hit_rate"] = float(np.mean(hits))
    res["per_task"] = per_task
    res["overall_accuracy"] = ci(group_by_patient(all_rows), bs)
    router = QuestionRouter(tok); router.load_state_dict(ck["router"])
    rr = [(t, q, router.route(q)) for t, qs in HELDOUT_PARAPHRASES.items() for q in qs]
    res["router_heldout_paraphrases"] = dict(n=len(rr), accuracy=float(np.mean([r[2][0] == r[0] for r in rr])),
                                             rows=[dict(task=t, question=q, routed=r[0], method=r[2]) for t, q, r in rr])
    return res


def eval_pipeline_answers(kind, qa, split, store, manifest, lesions, ev, temporal_rows):
    """Candidate-derived (deterministic) answers for single-exam questions."""
    from src.models.temporal_vqa import candidate_derived_answer
    from src.models.vqa import qa_task
    thr, cvol, cpeak = cand_rule(kind)
    rows = []
    cache = {}
    for q in qa:
        pid = q["exam_id"].rsplit("-T", 1)[0]
        if split.get(pid) != "test" or q["scope"] != "single":
            continue
        task = qa_task(q)
        if q["exam_id"] not in cache:
            cache[q["exam_id"]] = load_analysis(store, kind, q["exam_id"], thr, cvol, cpeak)
        a = cache[q["exam_id"]]
        pred = candidate_derived_answer(task, a.candidates)
        ref = ("region" if q["answer"] != "none" else "none") if task == "where" else q["answer"]
        rows.append(dict(patient_id=pid, task=task, pred=pred, ref=ref, correct=float(pred == ref)))
    out = {}
    for t in sorted({r["task"] for r in rows}):
        rr = [r for r in rows if r["task"] == t]
        out[t] = dict(n=len(rr), accuracy=float(np.mean([r["correct"] for r in rr])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=["intensity_baseline", "glow", "feature_glow", "contextflow"])
    ap.add_argument("--temporal-detector", default="contextflow")
    a = ap.parse_args()
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    ev = load_yaml(ROOT / "configs/evaluation.yaml")
    dcfg, manifest, split, store, by = data_context()
    test = by["test"]
    lesions = json.loads((ROOT / dcfg["lesions"]).read_text())
    qa = json.loads((ROOT / dcfg["qa_pairs"]).read_text())
    lbe = defaultdict(list)
    for l in lesions:
        lbe[l["exam_id"]].append(l)
    from register_pairs import pairs_of
    pairs = {p: v for p, v in pairs_of(manifest).items() if split[p] == "test"}
    results = dict(evaluation_set="test split (patient-level, held out)", dataset="SYNTHETIC chest CT phantoms",
                   n_test_patients=len({r["patient_id"] for r in test}), n_test_exams=len(test), n_test_pairs=len(pairs))
    results["anomaly"] = {}
    for k in a.detectors:
        if (ROOT / "experiments" / k / "state.json").exists():
            print("anomaly eval", k, flush=True)
            results["anomaly"][k] = eval_anomaly(k, test, store, ev)
    print("registration eval", flush=True)
    results["registration"] = eval_registration(pairs, lbe)
    results["temporal"] = {}
    for k in a.detectors:
        if (ROOT / "experiments" / k / "state.json").exists():
            print("temporal eval", k, flush=True)
            results["temporal"][f"{k}|registered|combined"] = eval_temporal(k, pairs, store, lbe, ev, True, "combined")
    k = a.temporal_detector
    print("temporal ablations", flush=True)
    results["temporal"][f"{k}|unregistered|combined"] = eval_temporal(k, pairs, store, lbe, ev, False, "combined")
    results["temporal"][f"{k}|registered|iou_only"] = eval_temporal(k, pairs, store, lbe, ev, True, "iou")
    results["vqa"] = {}
    for tag in ("vqa_with_anomaly", "vqa_without_anomaly"):
        if (ROOT / "experiments/vqa" / f"{tag}.pt").exists():
            print("vqa eval", tag, flush=True)
            results["vqa"][tag] = eval_vqa(tag, qa, split, store, manifest, lesions, ev["bootstrap"])
    results["candidate_derived_answers"] = eval_pipeline_answers(k, qa, split, store, manifest, lesions, ev, None)
    results["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    results["eval_runtime_s"] = round(time.time() - t0, 1)
    (RES / "results.json").write_text(json.dumps(results, indent=1, default=float))
    from make_report import write_markdown
    write_markdown(results, ROOT / "reports/RESULTS.md")
    print("wrote", RES / "results.json", "and reports/RESULTS.md")


if __name__ == "__main__":
    main()
