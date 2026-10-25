"""Orchestrator connecting preprocessing, anomaly detection, localisation,
registration, candidate matching, measurements and (temporal) VQA (spec §9).

Every stage is a separately testable module; this file only wires them.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml

from ..data.dicom_loader import load_dicom_dir
from ..data.preprocessing import PreprocConfig, ProcessedExam, preprocess_series, resample_like
from ..data.registration import RegistrationConfig, identity_pair, register_pair
from ..models.anomaly_detector import BaseDetector, IntensityBaseline, load_detector
from ..models.temporal_vqa import compose_single, compose_temporal, DISCLAIMER
from ..models.vqa import (ANSWERS, TASKS, TEMPORAL_TASKS, Tokenizer, QuestionRouter, VQAModel,
                          answer_mask, keyword_route)
from ..temporal.candidate_matching import MatchConfig, classify_unmatched, match_candidates
from ..temporal.measurements import pair_measurements
from .features import build_input
from .localization import extract_candidates

log = logging.getLogger(__name__)


@dataclass
class ExamAnalysis:
    exam_ref: str
    processed: ProcessedExam
    anomaly: object
    candidates: list
    candidate_mask: np.ndarray
    timings: dict = field(default_factory=dict)

    def summary(self):
        return dict(exam_ref=self.exam_ref, qc=self.processed.qc,
                    volume_score=self.anomaly.volume_score, method=self.anomaly.method,
                    n_candidates=len(self.candidates), candidates=self.candidates,
                    shape_zyx=list(self.processed.normalized.shape),
                    spacing_xyz=list(self.processed.image.GetSpacing()), timings=self.timings)


def detector_meta(p: ProcessedExam):
    return {"original_slice_spacing": float(p.original_meta.get("original_spacing", [0, 0, 3.0])[2]), "hu": p.hu}


def fov_box_mm(img: sitk.Image):
    sz = img.GetSize()
    pts = np.array([img.TransformIndexToPhysicalPoint((x, y, z)) for x in (0, sz[0] - 1)
                    for y in (0, sz[1] - 1) for z in (0, sz[2] - 1)])
    return pts.min(0).tolist() + pts.max(0).tolist()


class Orchestrator:
    def __init__(self, config="configs/inference.yaml", root=".", load_vqa=True, detector: BaseDetector | None = None):
        self.root = Path(root)
        if isinstance(config, dict):
            cfg = config
        else:
            cfg = yaml.safe_load((self.root / config).read_text()) if (self.root / config).exists() else {}
        self.cfg = cfg
        self.pre_cfg = PreprocConfig.from_dict(cfg.get("preprocessing"))
        if not Path(self.pre_cfg.lungmask_weights).is_absolute():
            self.pre_cfg.lungmask_weights = str(self.root / self.pre_cfg.lungmask_weights)
        self.reg_cfg = RegistrationConfig.from_dict(cfg.get("registration"))
        self.match_cfg = MatchConfig.from_dict(cfg.get("matching"))
        self.loc_cfg = cfg.get("localization", {"min_volume_mm3": 30.0})
        self.stable_band = cfg.get("stable_band_pct", 15.0)
        self.device = "cuda" if torch.cuda.is_available() and cfg.get("use_gpu", True) else "cpu"
        self.warnings = []
        self.detector = detector or self._load_detector()
        self.threshold = self.loc_cfg.get("threshold") or getattr(self.detector, "threshold", None) or 3.5
        # candidate-level rule selected on validation (scripts/select_candidate_filter.py); config overrides
        self.min_peak = self.loc_cfg.get("min_peak_score") or getattr(self.detector, "candidate_min_peak_score", None)
        self.min_volume = (self.loc_cfg.get("min_volume_mm3")
                           or getattr(self.detector, "candidate_min_volume_mm3", None) or 30.0)
        self.router, self.vqa_single, self.vqa_temporal, self.tok, self.vqa_meta = (
            self._load_vqa() if load_vqa else (None, None, None, None, {}))

    # ------------------------------------------------------------------ loading
    def _load_detector(self) -> BaseDetector:
        kind = self.cfg.get("detector", "contextflow")
        ck = self.root / self.cfg.get("detector_checkpoint", f"experiments/{kind}/best.pt")
        if kind != "intensity_baseline" and ck.exists():
            return load_detector(kind, str(ck), self.device, root=self.root)
        self.warnings.append(f"model weights for '{kind}' not found at {ck}; using intensity baseline")
        base = self.root / "experiments/intensity_baseline/state.json"
        return load_detector("intensity_baseline", str(base) if base.exists() else "")

    def _load_vqa(self):
        ck = self.root / self.cfg.get("vqa_checkpoint", "experiments/vqa/vqa_with_anomaly.pt")
        if not ck.exists():
            self.warnings.append(f"VQA weights not found at {ck}; keyword routing + candidate-derived answers only")
            return None, None, None, None, {}
        s = torch.load(ck, map_location=self.device, weights_only=False)
        tok = Tokenizer(vocab=s["vocab"])
        router = QuestionRouter(tok); router.load_state_dict(s["router"])
        inch = 3 if s["use_anomaly"] else 2
        w = s.get("width", 32)
        single = VQAModel(tok, inch, temporal=False, width=w); single.load_state_dict(s["single"])
        temporal = VQAModel(tok, inch, temporal=True, width=w); temporal.load_state_dict(s["temporal"])
        meta = dict(use_anomaly=s["use_anomaly"], T_single=s.get("temperature_single", 1.0),
                    T_temporal=s.get("temperature_temporal", 1.0), calibrated=s.get("calibrated_on_val", False))
        return router.eval(), single.eval(), temporal.eval(), tok, meta

    # ------------------------------------------------------------------ single exam
    def analyze_dicom(self, path, exam_ref, series_uid=None) -> ExamAnalysis:
        t0 = time.time()
        loaded = load_dicom_dir(path, series_uid)
        p = preprocess_series(loaded, self.pre_cfg)
        a = self.analyze_processed(p, exam_ref)
        a.timings["load_preprocess_s"] = round(time.time() - t0 - sum(a.timings.values()), 2)
        return a

    def analyze_processed(self, p: ProcessedExam, exam_ref) -> ExamAnalysis:
        t0 = time.time()
        out = self.detector.score(p.normalized, p.lung_mask, detector_meta(p))
        t1 = time.time()
        cands, cmask = extract_candidates(out.map, p.lung_mask, p.image, self.threshold,
                                          self.min_volume, prefix="C", min_peak_score=self.min_peak)
        hu = p.hu
        cd = []
        for k, c in enumerate(cands, start=1):
            d = c.to_dict()
            d["mean_hu"] = float(hu[cmask == k].mean())
            cd.append(d)
        return ExamAnalysis(exam_ref, p, out, cd, cmask,
                            dict(anomaly_s=round(t1 - t0, 2), localization_s=round(time.time() - t1, 2)))

    # ------------------------------------------------------------------ temporal
    def temporal(self, earlier: ExamAnalysis, later: ExamAnalysis, register=True):
        fixed, moving = later.processed, earlier.processed
        reg = (register_pair(fixed.image, moving.image, fixed.lung_mask, moving.lung_mask, self.reg_cfg)
               if register else identity_pair(fixed.image, moving.image, fixed.lung_mask, moving.lung_mask))
        amap = sitk.GetImageFromArray(earlier.anomaly.map.astype(np.float32)); amap.CopyInformation(moving.image)
        warped = sitk.GetArrayFromImage(resample_like(amap, fixed.image, transform=reg.transform, default=0.0))
        e_cands, e_mask = extract_candidates(warped, reg.moving_mask_resampled, fixed.image, self.threshold,
                                             self.min_volume, prefix="E", min_peak_score=self.min_peak)
        e_cands = [c.to_dict() for c in e_cands]
        l_cands = [dict(c) for c in later.candidates]
        for c in l_cands:
            c["candidate_id"] = "L" + c["candidate_id"][1:]
        m = match_candidates(e_cands, l_cands, self.match_cfg)
        status = classify_unmatched(m, e_cands, l_cands, reg.reliable,
                                    fov_earlier_mm=fov_box_mm(moving.image), fov_later_mm=fov_box_mm(fixed.image),
                                    cfg=self.match_cfg)
        entities = []
        for mt in m.matches:
            e, l = e_cands[mt["earlier"]], l_cands[mt["later"]]
            entities.append(dict(match_status="matched", earlier=e, later=l, match=mt,
                                 measurements=pair_measurements(e, l)))
        for s in status:
            c = (e_cands if s["side"] == "earlier" else l_cands)[s["index"]]
            e = c if s["side"] == "earlier" else None
            l = c if s["side"] == "later" else None
            entities.append(dict(match_status=s["status"], earlier=e, later=l, measurements=pair_measurements(e, l)))
        def size(ent):
            return max(ent["earlier"]["volume_mm3"] if ent["earlier"] else 0, ent["later"]["volume_mm3"] if ent["later"] else 0)
        entities.sort(key=size, reverse=True)
        counts = dict(n_matched=len(m.matches),
                      n_new=sum(s["status"] == "potentially_new" for s in status),
                      n_resolved=sum(s["status"] == "potentially_resolved" for s in status),
                      n_uncertain=sum(s["status"] == "uncertain" for s in status))
        primary = dict(entities[0], **counts) if entities else dict(match_status="none", earlier=None,
                                                                    later=None, measurements=None, **counts)
        return dict(registration=reg.report, entities=entities, primary=primary, counts=counts,
                    warped_earlier_map=warped, earlier_on_later=sitk.GetArrayFromImage(reg.moving_resampled),
                    earlier_lung_on_later=reg.moving_mask_resampled, earlier_candidates_on_later=e_cands,
                    later_candidates=l_cands)

    # ------------------------------------------------------------------ VQA
    def route(self, question):
        if self.router is None:
            return keyword_route(question), None, "keyword_only"
        return self.router.route(question)

    @torch.no_grad()
    def _vqa(self, task, question, later: ExamAnalysis, temporal_ctx=None):
        if self.vqa_single is None or task not in TASKS or task in ("measurement", "appearance", "unsupported"):
            return None, None
        ua = self.vqa_meta["use_anomaly"]
        p = later.processed
        x_l, box = build_input(p.normalized, p.lung_mask, later.anomaly.map, use_anomaly=ua)
        q = self.tok.batch([question]); t = torch.tensor([TASKS.index(task)])
        if task in TEMPORAL_TASKS:
            if temporal_ctx is None:
                return None, None
            from ..data.preprocessing import window_normalize
            e_norm = window_normalize(temporal_ctx["earlier_on_later"], self.pre_cfg.window_center,
                                      self.pre_cfg.window_width, self.pre_cfg.normalize)
            x_e, _ = build_input(e_norm, temporal_ctx["earlier_lung_on_later"], temporal_ctx["warped_earlier_map"],
                                 box=box, use_anomaly=ua)
            logits, _ = self.vqa_temporal(q, t, x_l[None], x_e[None]); T = self.vqa_meta["T_temporal"]
        else:
            logits, _ = self.vqa_single(q, t, x_l[None]); T = self.vqa_meta["T_single"]
        pr = F.softmax(logits / T + answer_mask(t), -1)[0]
        k = int(pr.argmax())
        return ANSWERS[k], float(pr[k])

    def ask(self, question, later: ExamAnalysis, earlier: ExamAnalysis | None = None, temporal_ctx=None):
        task, rconf, how = self.route(question)
        base = dict(routing=dict(task=task, confidence=rconf, method=how))
        if task == "unsupported":
            return dict(base, question=question, task=task, status="unsupported",
                        answer_text="This question is outside the scope of the system (chest-CT suspicious-region "
                                    "detection, localisation and change assessment). " + DISCLAIMER)
        if task == "appearance":
            c = max(later.candidates, key=lambda c: c["max_score"]) if later.candidates else None
            txt = ("No suspicious region exceeded the threshold; insufficient evidence." if not c else
                   f"Candidate {c['candidate_id']}: mean density {c['mean_hu']:.0f} HU, candidate-mask volume "
                   f"{c['volume_mm3']:.0f} mm³, longest diameter {c['longest_diameter_mm']:.1f} mm, {c['zone']} "
                   f"{c['laterality']} lung. The system does not classify lesion type or malignancy.")
            return dict(base, question=question, task=task, answer_text=txt + " " + DISCLAIMER,
                        evidence=[c] if c else [], status="answered" if c else "insufficient_evidence")
        if task in TEMPORAL_TASKS:
            if earlier is None:
                return dict(base, question=question, task=task, status="insufficient_evidence",
                            answer_text="This is a temporal question; select an earlier examination to compare.")
            if temporal_ctx is None:
                temporal_ctx = self.temporal(earlier, later)
            ans, conf = self._vqa(task, question, later, temporal_ctx)
            r = compose_temporal(question, task, ans, conf, temporal_ctx["primary"], temporal_ctx["registration"],
                                 earlier.exam_ref, later.exam_ref, self.stable_band,
                                 calibrated=self.vqa_meta.get("calibrated", False))
            return dict(base, **r)
        ans, conf = self._vqa(task, question, later)
        if ans is None:
            from ..models.temporal_vqa import candidate_derived_answer
            ans = candidate_derived_answer(task, later.candidates)
        r = compose_single(question, task, ans, conf, later.candidates, later.exam_ref,
                           calibrated=self.vqa_meta.get("calibrated", False))
        return dict(base, **r)
