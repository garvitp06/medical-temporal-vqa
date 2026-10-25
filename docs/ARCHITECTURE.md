# Architecture, assumptions and limitations

## Data flow

```
DICOM series ──► dicom_loader ──► preprocessing ──► anomaly detector ──► localisation ──► candidates
 (16-bit)       group by UID,     LPS reorient,     2.5D GLOW /           threshold (val),
                sort by IPP·n,    2.5 mm resample,  ContextFlow++-style   26-conn. components,
                HU = slope·v+b,   lung mask         latent-density map,   ≥30 mm³ rule, boxes,
                validate gaps     (lungmask R231),  z-scored vs normal    centroid (mm), diameter
                                  lung window       val lung voxels
                                                                │
earlier exam ──(same steps)──► registration (rigid MI, QC) ──► warp earlier map into later grid
                                                                │
                              candidate matching (Hungarian; IoU + distance + volume + feature cost)
                                                                │
                              measurements module (mask volume, bbox estimate, diameter, %Δ, Δscore)
                                                                │
question ──► router (GRU + keyword fallback) ──► VQA model (cross-attention over 3D spatial tokens
                                                  of [earlier, later, later−earlier]) ──► categorical answer
                                                                │
                              answer composer: VQA answer + measurement-derived answer + evidence
                              + registration QC + uncertainty; numbers only from the measurement module
```

## Modules

| Path | Responsibility |
|---|---|
| `src/data/dicom_loader.py` | Series discovery by (Study, Series) UID; explicit series selection; spatial sorting by projection of ImagePositionPatient on the slice normal; HU conversion; spacing from positions; gap/duplicate/orientation/dimension/corruption validation. |
| `src/data/preprocessing.py` | LPS reorientation, physical resampling (linear for images, nearest for masks), lung window normalisation, lung masks (pretrained lungmask R231 with QC, threshold fallback), operation log. HU is retained. |
| `src/data/registration.py` | Earlier→later rigid (Euler3D, Mattes MI, 3-level pyramid), optional B-spline; QC report (lung Dice, NCC before/after; reliability flag). |
| `src/data/dataset.py` | Manifest, patient-level stratified split, leakage check, split counts, 2.5D patch dataset. |
| `src/data/synthetic.py` | Synthetic phantom cohort with exact ground truth (SYNTHETIC). |
| `src/models/glow.py` | GLOW with additive-context hooks. |
| `src/models/contextflow.py` | ContextFlow++-inspired generalist–specialist flow. |
| `src/models/anomaly_detector.py` | Detector interface, intensity baseline, flow detector, score normalisation, volume/slice scores. |
| `src/inference/localization.py` | Map → candidates, physical geometry. |
| `src/temporal/*` | Matching and measurements. |
| `src/models/visual_encoder.py`, `vqa.py`, `temporal_vqa.py` | VQA encoder, router, fusion model, answer composer. |
| `src/inference/pipeline.py` | Orchestrator. |
| `src/evaluation/*` | Metrics (unit tested). |
| `src/api`, `frontend` | FastAPI + React research UI. |

## Detectors

| Name | Input | Density model | Status |
|---|---|---|---|
| `intensity_baseline` | lung-window image | none (smoothed density minus lung median) | baseline |
| `glow` | 3 neighbouring axial slices (pixels) | multi-scale GLOW | implemented; **poor localisation** on this data (see RESULTS) |
| `feature_glow` | frozen **pretrained lungmask R231** encoder features (down_path[1] ⊕ upsampled down_path[2], PCA→64, PCA fitted on train normal exams) | single-scale GLOW (no squeeze, couplings alternate 3×3/1×1) | implemented |
| `contextflow` | same features | ContextFlow++-style specialist on frozen `feature_glow` generalist | **default** (`configs/inference.yaml`) |

Feature-space flows (FastFlow/CFLOW-style) were added after the pixel-space GLOW, trained exactly as specified,
ranked dense lesions only weakly above vessels (validation voxel AP ≈ 0.02–0.03). A Gaussian on the pretrained
features already reached validation AP ≈ 0.77, so the flows were moved to that feature space. The pixel GLOW is kept
as an ablation. For feature flows the map is the per-position NLL
`−(Σ_c log N(z_c) + Σ_k log s_k)` (coupling log-scales at that position; ActNorm/1×1-conv log-dets are
position-independent constants). With 3×3 couplings positions interact, so this is the standard per-position
approximation.

## Anomaly map definition — pixel GLOW (exact)

For each lung-containing axial slice *k* the input is `x = [I(k−1), I(k), I(k+1)]` (lung-window normalised to
[−1, 1], zero-padded with −1 to a multiple of 8). The flow produces latents `z_l` at three scales. The raw map is

`r(x, y) = (1/L) Σ_l upsample_bilinear( −mean_c log N(z_l[c, ·, ·]; μ_l, σ_l) )(x, y)`

followed by a 3D Gaussian (σ = 0.7 voxel). The Jacobian log-determinant is a per-sample term and is not
localised. The normalised map is `(r − μ_val) / σ_val`, where μ/σ are computed over lung voxels of the validation
normal exams (same normalisation for every detector). Candidates use `map ≥ τ` with τ chosen on validation abnormal
exams by maximising mean Dice, followed by a candidate-level rule (component peak score ≥ p and candidate volume ≥ v)
chosen on all validation exams by maximising lesion-level F1 (`scripts/select_candidate_filter.py`).

## ContextFlow++ adaptation

Context `c = (p, k)`: `p ∈ [0, 1]` slice position within the lung's craniocaudal extent (continuous), and `k ∈ {0,1,2}`
original slice-thickness category (discrete, learned embedding). The conditional network outputs, for every flow
step, additive deltas to ActNorm log-scale/bias, the coupling network's output channels, and the 1×1 conv weight
(`W_g + 0.1·W_CN(c)`, log-det via per-sample `slogdet`). The generalist (GLOW) is frozen; the CN is
zero-initialised so the specialist starts exactly at the generalist likelihood.

## Assumptions

* Training protocol is normal-only (configurable); the original project protocol is unverified.
* Laterality uses the LPS convention (+x = patient left) relative to the lung-mask centroid.
* Lung zones are craniocaudal thirds of the lung extent — an engineering convention, not anatomical lobes.
* "Stable" volume band ±15% — configurable research default, not a clinical criterion.

## Limitations

* All quantitative results are on synthetic phantoms; there is no clinical validation of any kind.
* In these phantoms lesions are, by construction, homogeneous soft-tissue-density ellipsoids, which favours the
  simple intensity baseline; it outperforms every flow on most lesion-level metrics here. This should not be
  extrapolated to real CT, where vessels, scarring, atelectasis and noise patterns break density thresholding.
* The validation-selected candidate rule trades recall for precision (flows: ≈0.5 lesion recall on test at zero
  false positives on normal exams); a different operating point is a config change.
* 2.5D modelling, not volumetric 3D.
* Candidate volumes come from thresholded anomaly maps and are biased by the threshold and smoothing; they are
  not expert segmentations.
* The VQA model is small and trained on templated questions; the router generalises only to paraphrases close to
  the training bank; unknown questions fall back to keyword routing or are refused.
* Confidence calibration (temperature scaling) is fitted on synthetic validation data only.
* Rigid registration cannot model breathing/deformation; B-spline is provided but not evaluated.
