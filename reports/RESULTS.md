# Results — SYNTHETIC evaluation

> All numbers below were computed by `scripts/evaluate.py` from saved predictions on the **held-out test patients of a SYNTHETIC phantom cohort**. They demonstrate that the pipeline and metrics work; they say nothing about clinical performance.

Evaluation set: test split (patient-level, held out) — 10 patients, 20 exams, 10 temporal pairs. 95% CIs are patient-level bootstrap (2000 resamples).

## 1. Anomaly detection and localisation

| Detector | Voxel AUROC | Voxel AP | Exam AUROC | Exam AP | Dice mean [95% CI] | Dice median | Sensitivity | Precision | Voxel thr / cand. rule (val) | s/exam |
|---|---|---|---|---|---|---|---|---|---|---|
| intensity_baseline | 0.993 | 0.791 | 0.978 | 0.989 | 0.583 [0.453, 0.700] | 0.625 | 0.519 | 0.683 | 5.0 / ≥400 mm³, peak ≥5.0 | 0.1 |
| glow | 0.926 | 0.072 | 0.615 | 0.788 | 0.023 [0.018, 0.027] | 0.021 | 0.394 | 0.012 | 2.0 / ≥600 mm³, peak ≥2.5 | 19.1 |
| feature_glow | 0.999 | 0.895 | 0.890 | 0.956 | 0.534 [0.432, 0.616] | 0.610 | 0.583 | 0.538 | 4.5 / ≥1600 mm³, peak ≥4.5 | 103.1 |
| contextflow | 0.999 | 0.900 | 0.846 | 0.931 | 0.552 [0.463, 0.639] | 0.599 | 0.577 | 0.606 | 4.0 / ≥1600 mm³, peak ≥4.0 | 44.2 |

Voxel-level: lung voxels (all lesion voxels + ≤15k random lung voxels per test exam). Exam-level: volume score vs. abnormal label. Segmentation: voxel-level per abnormal test exam at the validation-selected threshold.

**Lesion-level 3D detection** (greedy matching by score)

| Detector | Criterion | Ref lesions | TP | FP | FN | Precision | Recall | F1 | AP | Mean IoU (TP) | FP / normal exam |
|---|---|---|---|---|---|---|---|---|---|---|---|
| intensity_baseline | 3D IoU ≥ 0.1 | 20 | 15 | 0 | 5 | 1.000 | 0.750 | 0.857 | 0.750 | 0.595 | 0.00 |
| intensity_baseline | centroid in box | 20 | 15 | 0 | 5 | 1.000 | 0.750 | 0.857 | 0.750 | 0.595 | 0.00 |
| glow | 3D IoU ≥ 0.1 | 20 | 14 | 165 | 6 | 0.078 | 0.700 | 0.141 | 0.057 | 0.303 | 6.43 |
| glow | centroid in box | 20 | 14 | 165 | 6 | 0.078 | 0.700 | 0.141 | 0.057 | 0.291 | 6.43 |
| feature_glow | 3D IoU ≥ 0.1 | 20 | 11 | 3 | 9 | 0.786 | 0.550 | 0.647 | 0.432 | 0.603 | 0.00 |
| feature_glow | centroid in box | 20 | 11 | 3 | 9 | 0.786 | 0.550 | 0.647 | 0.432 | 0.603 | 0.00 |
| contextflow | 3D IoU ≥ 0.1 | 20 | 10 | 2 | 10 | 0.833 | 0.500 | 0.625 | 0.417 | 0.613 | 0.00 |
| contextflow | centroid in box | 20 | 10 | 2 | 10 | 0.833 | 0.500 | 0.625 | 0.417 | 0.613 | 0.00 |

## 2. Registration (test pairs)

| Pairs | Failure rate | Lung Dice before → after | Lung NCC before → after | TRE mean / median (mm) | Displacement without registration (mm) |
|---|---|---|---|---|---|
| 10 | 0.000 | 0.890 → 0.971 | 0.306 → 0.852 | 0.09 / 0.09 | 7.5 |

TRE uses 12 synthetic lesion-centre landmarks whose true positions are known by construction.

## 3. Temporal comparison (lesion level, test pairs)

| Configuration | Ref lesions | Associated | Association acc. | Change-category acc. | Category acc. as reported* | Later-vol MAE (mm³) | Later-vol MAPE (%) | ΔV MAE (mm³) | %Δ MAE / median (pp) |
|---|---|---|---|---|---|---|---|---|---|
| intensity_baseline|registered|combined | 12 | 11 | 0.667 | 0.583 | 0.583 | 1402 | 35.8 | 527 | 134.0 / 38.4 |
| glow|registered|combined | 12 | 12 | 0.750 | 0.333 | 0.333 | 5747 | 164.1 | 1349 | 99.2 / 106.7 |
| feature_glow|registered|combined | 12 | 9 | 0.417 | 0.333 | 0.333 | 578 | 14.0 | 789 | 20.3 / 20.3 |
| contextflow|registered|combined | 12 | 8 | 0.333 | 0.250 | 0.250 | 461 | 11.8 | 461 | 10.9 / 10.9 |
| contextflow|unregistered|combined | 12 | 8 | 0.167 | 0.250 | 0.000 | 461 | 11.8 | 414 | 9.7 / 9.7 |
| contextflow|registered|iou_only | 12 | 8 | 0.333 | 0.250 | 0.250 | 461 | 11.8 | 461 | 10.9 / 10.9 |

*As reported: pairs with unreliable registration are reported as 'undetermined' (counted wrong). Volume errors compare candidate-mask volumes with reference lesion-mask volumes for lesions detected at both timepoints; single-measurement errors and change errors are listed separately.

Confusion matrix (intensity_baseline|registered|combined; rows = reference, columns = predicted):

| ref \ pred | decreased | increased | new | none | resolved | stable |
|---|---|---|---|---|---|---|
| decreased | 0 | 0 | 0 | 0 | 2 | 0 |
| increased | 0 | 2 | 1 | 0 | 0 | 0 |
| new | 0 | 0 | 1 | 0 | 0 | 0 |
| none | 0 | 0 | 0 | 0 | 0 | 0 |
| resolved | 0 | 0 | 0 | 0 | 3 | 0 |
| stable | 0 | 1 | 0 | 1 | 0 | 1 |

## 4. Visual question answering (test questions)

| Task | vqa_with_anomaly acc [95% CI] | vqa_without_anomaly acc [95% CI] | candidate-derived acc | n |
|---|---|---|---|---|
| existence | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] | 0.800 | 20 |
| laterality | 0.750 [0.550, 0.950] | 0.600 [0.350, 0.850] | 0.700 | 20 |
| new_finding | 0.900 [0.700, 1.000] | 0.800 [0.500, 1.000] | — | 10 |
| progression | 0.600 [0.300, 0.900] | 0.700 [0.400, 1.000] | — | 10 |
| where | 0.900 [0.750, 1.000] | 1.000 [1.000, 1.000] | 0.800 | 20 |
| zone | 0.750 [0.500, 1.000] | 0.750 [0.550, 0.950] | 0.800 | 20 |
| **overall** | 0.810 [0.680, 0.920] | 0.820 [0.710, 0.930] | | |

Macro-F1 and region-token hit rate per task:

| Model | Task | Macro-F1 | Region hit |
|---|---|---|---|
| vqa_with_anomaly | existence | 0.896 | — |
| vqa_with_anomaly | laterality | 0.689 | — |
| vqa_with_anomaly | new_finding | 0.474 | — |
| vqa_with_anomaly | progression | 0.542 | 0.800 |
| vqa_with_anomaly | where | 0.896 | 0.692 |
| vqa_with_anomaly | zone | 0.546 | — |
| vqa_without_anomaly | existence | 1.000 | — |
| vqa_without_anomaly | laterality | 0.475 | — |
| vqa_without_anomaly | new_finding | 0.444 | — |
| vqa_without_anomaly | progression | 0.700 | 0.800 |
| vqa_without_anomaly | where | 1.000 | 0.846 |
| vqa_without_anomaly | zone | 0.587 | — |

**Calibration** (temperature fitted on validation):

| Model | Head | ECE raw → scaled | Brier raw → scaled | T | n |
|---|---|---|---|---|---|
| vqa_with_anomaly | single | 0.119 → 0.079 | 0.270 → 0.248 | 1.60 | 80 |
| vqa_with_anomaly | temporal | 0.189 → 0.173 | 0.371 → 0.363 | 3.00 | 20 |
| vqa_without_anomaly | single | 0.040 → 0.081 | 0.194 → 0.201 | 1.60 | 80 |
| vqa_without_anomaly | temporal | 0.180 → 0.168 | 0.376 → 0.372 | 2.50 | 20 |

**Question router on held-out paraphrases:** vqa_with_anomaly: 1.000 (n=9), vqa_without_anomaly: 1.000 (n=9)

Evaluation runtime 356.3 s; peak RSS 1448 MB.
