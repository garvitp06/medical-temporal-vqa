"""Longitudinal registration (spec §4.4).

Convention: the EARLIER exam (moving) is registered into the LATER exam's
(fixed) coordinate system and resampled onto the later exam's grid. All
temporal measurements are reported in the later exam's physical space.

Stages: centre-of-mass initialisation -> rigid (Euler3D, Mattes MI,
multi-resolution) -> optional B-spline deformable refinement.
Quality report: NCC and lung-mask Dice before/after, final metric value,
and a pass/fail flag against configurable thresholds. Growth must not be
reported for pairs whose registration is flagged unreliable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import SimpleITK as sitk

from .preprocessing import resample_like


@dataclass
class RegistrationConfig:
    method: str = "rigid"            # rigid | rigid+bspline
    sampling_percentage: float = 0.2
    iterations: int = 200
    bspline_mesh: int = 6
    bspline_iterations: int = 60
    min_lung_dice: float = 0.85      # QC thresholds
    min_ncc_lung: float = 0.30
    seed: int = 42

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class RegistrationResult:
    transform: sitk.Transform
    moving_resampled: sitk.Image          # earlier exam on later grid (HU)
    moving_mask_resampled: np.ndarray     # earlier lung mask on later grid
    report: dict = field(default_factory=dict)

    @property
    def reliable(self) -> bool:
        return bool(self.report.get("reliable", False))


def ncc(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    if mask is not None:
        a, b = a[mask > 0], b[mask > 0]
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    if a.size < 2:
        return float("nan")
    a -= a.mean(); b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def dice(a, b) -> float:
    a = a > 0; b = b > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def _clip_lung(img):
    # register on a lung-windowed image so bone/air extremes do not dominate MI
    return sitk.Clamp(sitk.Cast(img, sitk.sitkFloat32), sitk.sitkFloat32, -1000.0, 300.0)


def register_pair(fixed: sitk.Image, moving: sitk.Image, fixed_mask: np.ndarray,
                  moving_mask: np.ndarray, cfg: RegistrationConfig | None = None) -> RegistrationResult:
    cfg = cfg or RegistrationConfig()
    f, m = _clip_lung(fixed), _clip_lung(moving)
    init = sitk.CenteredTransformInitializer(f, m, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(cfg.sampling_percentage, cfg.seed)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4,
                                                numberOfIterations=cfg.iterations,
                                                relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1]); R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    R.SetInitialTransform(init, inPlace=False)
    rigid = R.Execute(f, m)
    final_metric = R.GetMetricValue()
    stop = R.GetOptimizerStopConditionDescription()
    transform = rigid
    if cfg.method == "rigid+bspline":
        mesh = [cfg.bspline_mesh] * 3
        bs = sitk.BSplineTransformInitializer(f, mesh)
        B = sitk.ImageRegistrationMethod()
        B.SetMetricAsMattesMutualInformation(48)
        B.SetMetricSamplingStrategy(B.RANDOM); B.SetMetricSamplingPercentage(cfg.sampling_percentage, cfg.seed)
        B.SetMovingInitialTransform(rigid)
        B.SetInitialTransform(bs, inPlace=True)
        B.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=cfg.bspline_iterations)
        B.SetInterpolator(sitk.sitkLinear)
        B.Execute(f, m)
        transform = sitk.CompositeTransform([rigid, bs])
        final_metric = B.GetMetricValue()
    moved = resample_like(moving, fixed, transform=transform)
    mmask_img = sitk.GetImageFromArray(moving_mask.astype(np.uint8)); mmask_img.CopyInformation(moving)
    moved_mask = sitk.GetArrayFromImage(resample_like(mmask_img, fixed, is_mask=True, transform=transform))
    # "before": earlier exam resampled with identity (physical overlay, no alignment)
    before = sitk.GetArrayFromImage(resample_like(moving, fixed))
    before_mask = sitk.GetArrayFromImage(resample_like(mmask_img, fixed, is_mask=True))
    fa = sitk.GetArrayFromImage(fixed)
    union = ((fixed_mask > 0) | (moved_mask > 0)).astype(np.uint8)
    rep = dict(method=cfg.method, final_metric_mattes_mi=float(final_metric), stop_condition=stop,
               ncc_lung_before=ncc(fa, before, fixed_mask), ncc_lung_after=ncc(fa, sitk.GetArrayFromImage(moved), union),
               lung_dice_before=dice(fixed_mask, before_mask), lung_dice_after=dice(fixed_mask, moved_mask))
    if isinstance(rigid, sitk.CompositeTransform) or hasattr(rigid, "GetNthTransform"):
        try:
            eul = sitk.Euler3DTransform(rigid.GetNthTransform(0)) if rigid.GetNumberOfTransforms() else None
        except Exception:
            eul = None
    else:
        eul = None
    try:
        e = sitk.Euler3DTransform(rigid.GetNthTransform(rigid.GetNumberOfTransforms() - 1)) \
            if hasattr(rigid, "GetNumberOfTransforms") else sitk.Euler3DTransform(rigid)
        rep["rigid_params"] = dict(rotation_deg=np.rad2deg(e.GetParameters()[:3]).tolist(),
                                   translation_mm=list(e.GetParameters()[3:]),
                                   center_mm=list(e.GetCenter()))
    except Exception:
        pass
    reasons = []
    if rep["lung_dice_after"] < cfg.min_lung_dice:
        reasons.append(f"lung Dice after registration {rep['lung_dice_after']:.3f} < {cfg.min_lung_dice}")
    if not np.isfinite(rep["ncc_lung_after"]) or rep["ncc_lung_after"] < cfg.min_ncc_lung:
        reasons.append(f"lung NCC after registration {rep['ncc_lung_after']:.3f} < {cfg.min_ncc_lung}")
    rep["reliable"] = not reasons
    rep["quality"] = "acceptable" if not reasons else "unreliable"
    rep["failure_reasons"] = reasons
    return RegistrationResult(transform, moved, moved_mask.astype(np.uint8), rep)


def identity_pair(fixed, moving, fixed_mask, moving_mask) -> RegistrationResult:
    """Ablation helper: physical overlay without registration."""
    moved = resample_like(moving, fixed)
    mm = sitk.GetImageFromArray(moving_mask.astype(np.uint8)); mm.CopyInformation(moving)
    moved_mask = sitk.GetArrayFromImage(resample_like(mm, fixed, is_mask=True))
    fa = sitk.GetArrayFromImage(fixed)
    rep = dict(method="none", lung_dice_after=dice(fixed_mask, moved_mask),
               ncc_lung_after=ncc(fa, sitk.GetArrayFromImage(moved), fixed_mask),
               reliable=False, quality="not_registered",
               failure_reasons=["registration disabled (ablation)"])
    return RegistrationResult(sitk.Euler3DTransform(), moved, moved_mask.astype(np.uint8), rep)
