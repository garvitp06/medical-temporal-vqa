"""FastAPI backend for the research prototype (spec §12).

Run:  uvicorn src.api.app:app --port 8000      (from the repository root)
The built React UI (frontend/dist) is served at "/" when present.
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..data.dataset import anonymize_id
from ..data.dicom_loader import DicomValidationError, discover_series
from ..data.preprocessing import ProcessedExam
from ..inference.pipeline import ExamAnalysis, Orchestrator
from ..models.anomaly_detector import AnomalyOutput
from .schemas import AnomalyRequest, AskRequest, CompareRequest
from .storage import Registry, _json_default

ROOT = Path(os.environ.get("MTVQA_ROOT", Path(__file__).resolve().parents[2]))
STORAGE = Path(os.environ.get("MTVQA_STORAGE", ROOT / "storage"))
log = logging.getLogger("mtvqa.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Medical Temporal VQA — research prototype",
              description="SYNTHETIC/RESEARCH USE ONLY. Outputs are suspicious regions for clinician review, not diagnoses.",
              version="0.1.0")
registry = Registry(STORAGE)
_orch: Orchestrator | None = None
_orch_lock = threading.Lock()
_cache: "OrderedDict[str, ExamAnalysis]" = OrderedDict()
_temporal_cache: "OrderedDict[str, dict]" = OrderedDict()
CACHE_SIZE = 6


def orch() -> Orchestrator:
    global _orch
    with _orch_lock:
        if _orch is None:
            _orch = Orchestrator(os.environ.get("MTVQA_CONFIG", "configs/inference.yaml"), root=ROOT)
        return _orch


def _remember(cache, key, val):
    cache[key] = val; cache.move_to_end(key)
    while len(cache) > CACHE_SIZE:
        cache.popitem(last=False)


# --------------------------------------------------------------------------- persistence of analyses
def save_analysis(sid, a: ExamAnalysis):
    d = registry.study_dir(sid) / "processed"
    a.processed.save(d, "exam")
    np.savez_compressed(d / "anomaly.npz", map=a.anomaly.map.astype(np.float16),
                        slice_scores=a.anomaly.slice_scores, volume_score=a.anomaly.volume_score,
                        candidate_mask=a.candidate_mask.astype(np.int16))
    (d / "candidates.json").write_text(json.dumps(a.candidates, default=_json_default))


def load_analysis(sid) -> ExamAnalysis:
    if sid in _cache:
        _cache.move_to_end(sid); return _cache[sid]
    s = registry.get_study(sid)
    if not s:
        raise HTTPException(404, "study not found")
    if s["status"] != "ready":
        raise HTTPException(409, f"study is not ready (status: {s['status']})")
    d = registry.study_dir(sid) / "processed"
    img = sitk.ReadImage(str(d / "exam_hu.nii.gz"))
    norm = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "exam_norm.nii.gz")))
    lung = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "exam_lung.nii.gz")))
    meta = json.loads((d / "exam_preproc.json").read_text())
    z = np.load(d / "anomaly.npz")
    a = ExamAnalysis(sid, ProcessedExam(img, norm, lung, meta["original_meta"], meta["qc"], meta["ops"]),
                     AnomalyOutput(None, z["map"].astype(np.float32), z["slice_scores"], float(z["volume_score"]),
                                   s["summary"]["detector"]),
                     json.loads((d / "candidates.json").read_text()), z["candidate_mask"].astype(np.int32))
    _remember(_cache, sid, a)
    return a


def process_study(sid):
    try:
        registry.update_study(sid, status="processing")
        src = registry.study_dir(sid) / "dicom"
        o = orch()
        a = o.analyze_dicom(src, exam_ref=sid)
        save_analysis(sid, a)
        md = a.processed.original_meta
        summary = dict(metadata={k: v for k, v in md.items() if k not in ("study_uid", "series_uid")},
                       qc=a.processed.qc, preprocessing_ops=a.processed.ops,
                       detector=a.anomaly.method, threshold=o.threshold, volume_score=a.anomaly.volume_score,
                       slice_scores=a.anomaly.slice_scores.tolist(), candidates=a.candidates,
                       shape_zyx=list(a.processed.normalized.shape), spacing_xyz=list(a.processed.image.GetSpacing()),
                       timings=a.timings, warnings=o.warnings + a.processed.qc.get("warnings", []),
                       synthetic="SYNTHETIC" in str(md.get("ImageComments", "")).upper())
        registry.update_study(sid, status="ready", study_date=md.get("StudyDate"), summary=summary,
                              synthetic=int(summary["synthetic"]))
        _remember(_cache, sid, a)
        log.info("study %s processed (%d candidates)", sid, len(a.candidates))
    except DicomValidationError as exc:
        registry.update_study(sid, status="failed", error=f"invalid DICOM series: {exc}")
    except Exception as exc:  # noqa: BLE001
        log.exception("processing failed for study %s", sid)
        registry.update_study(sid, status="failed", error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    o = orch()
    return dict(status="ok", device=o.device, gpu_available=torch.cuda.is_available(),
                detector=o.detector.name, threshold=o.threshold, vqa_loaded=o.vqa_single is not None,
                warnings=o.warnings, disclaimer="Research prototype. Not for clinical use.")


@app.post("/studies/upload")
async def upload(background: BackgroundTasks, files: list[UploadFile] = File(...)):
    sid = registry.new_study()
    dst = registry.study_dir(sid) / "dicom"; dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        data = await f.read()
        name = Path(f.filename or f"file{n}").name
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for m in z.infolist():
                    if not m.is_dir():
                        (dst / f"z{n}_{Path(m.filename).name}").write_bytes(z.read(m)); n += 1
        else:
            (dst / f"{n:05d}_{name}").write_bytes(data); n += 1
    series = discover_series(dst)
    if not series:
        registry.update_study(sid, status="failed", error="no DICOM files found in upload")
        raise HTTPException(400, "no readable DICOM files in upload")
    import pydicom
    ds = pydicom.dcmread(str(series[0].files[0]), stop_before_pixels=True)
    registry.update_study(sid, patient_key=anonymize_id(str(ds.get("PatientID", "unknown"))))
    background.add_task(process_study, sid)
    return dict(study_id=sid, status="processing", n_files=n, n_series=len(series))


@app.post("/demo/synthetic")
def load_synthetic_demo(background: BackgroundTasks, n_patients: int = 2):
    """Import SYNTHETIC test-split phantom studies (clearly labelled) for demos."""
    split_f = ROOT / "data/splits/patient_split.json"
    man_f = ROOT / "data/raw/synthetic/manifest.json"
    if not man_f.exists():
        raise HTTPException(404, "synthetic data not generated; run scripts/prepare_data.py")
    split = json.loads(split_f.read_text()) if split_f.exists() else {}
    man = json.loads(man_f.read_text())
    pats = sorted({r["patient_id"] for r in man if split.get(r["patient_id"]) == "test" and r["abnormal"]})[:n_patients]
    out = []
    for r in [r for r in man if r["patient_id"] in pats]:
        sid = registry.new_study(synthetic=True)
        dst = registry.study_dir(sid) / "dicom"
        shutil.copytree(ROOT / "data/raw/synthetic" / r["dicom_dir"], dst, dirs_exist_ok=True)
        registry.update_study(sid, patient_key=anonymize_id(r["patient_id"]))
        background.add_task(process_study, sid)
        out.append(dict(study_id=sid, timepoint=r["timepoint"]))
    return dict(imported=out, label="SYNTHETIC DEMO DATA")


@app.get("/studies")
def list_studies():
    return registry.list_studies()


@app.get("/studies/{study_id}")
def get_study(study_id: str):
    s = registry.get_study(study_id)
    if not s:
        raise HTTPException(404, "study not found")
    return s


def _png(img: np.ndarray) -> Response:
    ok, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "private, max-age=600"})


def render_ct(hu2d, wc, ww):
    lo, hi = wc - ww / 2, wc + ww / 2
    return (np.clip((hu2d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def render_heat(m2d, lung2d, thr=3.0):
    """Heatmap scaled to the detector's validation threshold (0.5x..2x thr)."""
    vmin, vmax = 0.5 * thr, 2.0 * thr
    v = np.clip((m2d - vmin) / (vmax - vmin), 0, 1)
    rgb = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = ((v > 0) & (lung2d > 0)).astype(np.float32) * (0.25 + 0.6 * v)
    return np.dstack([rgb, (alpha * 255).astype(np.uint8)])


@app.get("/studies/{study_id}/slices/{z}")
def study_slice(study_id: str, z: int, layer: str = Query("ct", pattern="^(ct|anomaly)$"),
                wc: float = -600, ww: float = 1500):
    a = load_analysis(study_id)
    hu = a.processed.hu
    if not 0 <= z < hu.shape[0]:
        raise HTTPException(400, "slice index out of range")
    if layer == "ct":
        return _png(render_ct(hu[z], wc, ww))
    return _png(render_heat(a.anomaly.map[z], a.processed.lung_mask[z], orch().threshold))


@app.post("/anomaly-detection")
def anomaly_detection(req: AnomalyRequest):
    a = load_analysis(req.study_id)
    s = registry.get_study(req.study_id)["summary"]
    payload = dict(study_id=req.study_id, detector=s["detector"], threshold=s["threshold"],
                   volume_score=a.anomaly.volume_score, slice_scores=s["slice_scores"], candidates=a.candidates,
                   score_semantics="normalised anomaly score (SD units vs validation normal lung); not a probability",
                   disclaimer="Suspicious regions require professional review; not a diagnosis.")
    rid = registry.save_result("anomaly", [req.study_id], payload)
    return dict(payload, result_id=rid)


@app.post("/vqa/ask")
def ask(req: AskRequest):
    a = load_analysis(req.study_id)
    e = load_analysis(req.earlier_study_id) if req.earlier_study_id else None
    tctx = _temporal(e, a) if e else None
    ans = orch().ask(req.question, a, e, tctx)
    rid = registry.save_result("vqa", [x for x in (req.earlier_study_id, req.study_id) if x], ans)
    return dict(ans, result_id=rid)


def _temporal(e: ExamAnalysis, l: ExamAnalysis):
    key = f"{e.exam_ref}->{l.exam_ref}"
    if key not in _temporal_cache:
        _remember(_temporal_cache, key, orch().temporal(e, l))
    return _temporal_cache[key]


@app.post("/temporal-vqa/compare")
def compare(req: CompareRequest):
    se, sl = registry.get_study(req.earlier_study_id), registry.get_study(req.later_study_id)
    if not se or not sl:
        raise HTTPException(404, "study not found")
    if se["patient_key"] != sl["patient_key"]:
        raise HTTPException(400, "earlier and later studies belong to different patients")
    if (se["study_date"] or "") > (sl["study_date"] or ""):
        raise HTTPException(400, "earlier study date is after the later study date")
    e, l = load_analysis(req.earlier_study_id), load_analysis(req.later_study_id)
    tctx = _temporal(e, l)
    ans = orch().ask(req.question, l, e, tctx)
    payload = dict(answer=ans, registration=tctx["registration"], entities=tctx["entities"], counts=tctx["counts"],
                   earlier_study_id=req.earlier_study_id, later_study_id=req.later_study_id)
    rid = registry.save_result("temporal", [req.earlier_study_id, req.later_study_id], payload)
    d = STORAGE / "results"; d.mkdir(exist_ok=True)
    np.savez_compressed(d / f"{rid}.npz", earlier_hu=tctx["earlier_on_later"].astype(np.float16),
                        earlier_map=tctx["warped_earlier_map"].astype(np.float16),
                        earlier_lung=tctx["earlier_lung_on_later"].astype(np.uint8))
    return dict(payload, result_id=rid)


@app.get("/results/{result_id}")
def get_result(result_id: str):
    r = registry.get_result(result_id)
    if not r:
        raise HTTPException(404, "result not found")
    return r


@app.get("/results/{result_id}/slices/{z}")
def result_slice(result_id: str, z: int, layer: str = Query("ct", pattern="^(ct|anomaly)$"),
                 wc: float = -600, ww: float = 1500):
    f = STORAGE / "results" / f"{result_id}.npz"
    if not f.exists():
        raise HTTPException(404, "no registered volume for this result")
    d = np.load(f)
    hu = d["earlier_hu"]
    if not 0 <= z < hu.shape[0]:
        raise HTTPException(400, "slice index out of range")
    if layer == "ct":
        return _png(render_ct(hu[z].astype(np.float32), wc, ww))
    return _png(render_heat(d["earlier_map"][z].astype(np.float32), d["earlier_lung"][z], orch().threshold))


@app.get("/results/{result_id}/report")
def export_report(result_id: str):
    """Structured research report. Contains only model outputs, measurements and
    their limitations — no diagnosis is added."""
    r = registry.get_result(result_id)
    if not r:
        raise HTTPException(404, "result not found")
    studies = [registry.get_study(s) for s in r["study_ids"]]
    rep = dict(report_type="research_prototype_output", generated_from_result=result_id,
               synthetic_data=any(s["synthetic"] for s in studies if s),
               studies=[dict(study_id=s["study_id"], study_date=s["study_date"], qc=(s["summary"] or {}).get("qc"),
                             detector=(s["summary"] or {}).get("detector")) for s in studies if s],
               content=r["payload"],
               limitations=["Model outputs are suspicious regions for review, not diagnoses.",
                            "Volumes are candidate-mask volumes from thresholded anomaly maps, not expert segmentations.",
                            "Models were trained and evaluated on SYNTHETIC phantoms only; no clinical validation."])
    return JSONResponse(rep, headers={"Content-Disposition": f"attachment; filename=report_{result_id}.json"})


@app.exception_handler(ValueError)
def value_error(_, exc):
    return JSONResponse(status_code=422, content=dict(detail=str(exc)))


_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
