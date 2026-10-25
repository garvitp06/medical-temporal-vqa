# Phase 1 — Data, model and feasibility report

Date of assessment: 2026-10-24. Build environment: 2-core CPU, 7 GB RAM, no GPU, outbound network limited to
PyPI, npm and GitHub (Hugging Face, download.pytorch.org and most other hosts were blocked).

## Datasets

| Item | What was verified | Outcome |
|---|---|---|
| "3D Rad" | Identified as **3D-RAD** (Gai et al., NeurIPS 2025 Datasets & Benchmarks; arXiv:2506.11147): 3D chest-CT Med-VQA built on CT-RATE (16,188 scans / 11,255 patients); six tasks incl. Static and Longitudinal Temporal Diagnosis; volumes in NIfTI (`.nii.gz`), not DICOM; hosted on Hugging Face as `Tang-xiaoxiao/3D-RAD`; source data CT-RATE is CC BY-NC-SA; no separate 3D-RAD licence stated in the paper. | **Not used.** Hugging Face was unreachable from the build environment, and its schema/annotations could not be inspected. No 3D-RAD-specific loader was written, to avoid guessing its schema. The generic manifest loader (`src/data/dataset.py`) accepts any dataset converted to the documented manifest format; NIfTI input would need a small adapter (DICOM is the implemented input path). |
| Real DICOM chest CT with lesion masks + serial scans | Not available in the environment. | Not used. |
| Synthetic cohort | Implemented in `src/data/synthetic.py`. 48 synthetic patients, 2 timepoints each, 16-bit DICOM with shuffled filenames/InstanceNumbers, varying pixel spacing (2.6–2.9 mm), slice spacing (2.5/3.0/3.5 mm), FOV offsets and rigid repositioning between timepoints; analytic lung/vessel/spine phantoms; ellipsoidal lesions that increase, decrease, stay stable, appear or resolve. Exact masks, volumes, laterality, zones and QA pairs are derived from the generator's ground truth. | **Used for all training and evaluation.** Every output is labelled SYNTHETIC. |

## Pretrained models

| Model | Source | Outcome |
|---|---|---|
| lungmask U-Net R231 (Hofmanninger et al., Eur Radiol Exp 2020) — pretrained CT lung segmentation | GitHub release `JoHof/lungmask` v0.0, `unet_r231-d5d2fc3d.pth` (≈124 MB) | **Downloaded and used** (a) for lung masks (`lung_mask_method: auto`; all 96 synthetic exams segmented, 100% of reference lesion voxels inside the mask, Dice 0.975 vs. the threshold method on a spot check) and (b) as the frozen feature extractor of the feature-space flows. |
| MONAI `lung_nodule_ct_detection` bundle (RetinaNet, LUNA16) | GitHub release `Project-MONAI/model-zoo` (v0.5.9 reachable) | Reachable but **not integrated**: it is a supervised nodule detector, outside the flow-based protocol, and its 3D RetinaNet inference is impractical on the 2-core CPU. Documented as a future reference detector. |
| GLOW / ContextFlow++ CT weights | None published for CT. | Trained from scratch on synthetic data. |
| Pretrained medical VQA (e.g. M3D-RAD, 4–7B parameters) | Weights location not stated in the 3D-RAD paper; Hugging Face blocked; far beyond CPU budget. | Not used. A compact cross-attention VQA model is trained from scratch. |
| Pretrained 3D CT image encoder | Hugging Face blocked. | Encoder trained from scratch (documented in `src/models/visual_encoder.py`). |

## Method verification

* **GLOW**: implemented from Kingma & Dhariwal (2018): ActNorm with data-dependent init, LU-parameterised
  invertible 1×1 convolutions, affine coupling with `sigmoid(raw+2)` scales, squeeze, multi-scale split priors,
  exact change-of-variables log-likelihood. Layer log-determinants are unit-tested against brute-force Jacobians.
* **ContextFlow++**: verified against Gudovskiy, Okuno & Nakata, UAI 2024 (PMLR v244, arXiv:2406.00578; code
  `github.com/gudovskiy/contextflow`). Reproduced: frozen generalist, additive specialist conditioning of coupling,
  ActNorm and 1×1-conv weights (`W_g + W_CN(c)`), mixed-variable context encoders (learned embedding,
  dequantisation). **Not** reproduced: FlowGMM 8-component mixture base and the semi-supervised classifier term; the
  paper reports no CT anomaly-detection setup, so CT context variables and the anomaly map are our adaptation.
  The module is therefore labelled "ContextFlow++-inspired".
* **Original IIT Jodhpur project code/paper**: not available. Training protocol (normal-only vs. mixed) could not be
  verified; `protocol` is a config switch, default `normal_only`.

## Compute-driven design decisions

* 2.5D flows (3 neighbouring axial slices as channels), 64×64 training patches, fully-convolutional whole-slice
  inference. This is **not** true 3D volumetric modelling.
* Isotropic 2.5 mm resampling; VQA encoder input 32×48×48 lung-cropped grid.
* Rigid registration by default (B-spline available via `method: rigid+bspline`).

## Consequence

The system is a complete, runnable research pipeline whose reported metrics are valid only for the synthetic
cohort. To use real data: convert it to the manifest format, re-run `prepare_data.py`, retrain, and re-evaluate.
