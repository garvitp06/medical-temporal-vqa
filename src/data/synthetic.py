"""Synthetic chest-CT phantom generator (SYNTHETIC DATA — NOT CLINICAL).

Produces 16-bit DICOM series of analytic chest phantoms so that every part of
the pipeline (DICOM ordering, HU conversion, resampling, registration, anomaly
detection, temporal matching, VQA) can be exercised end-to-end with exact
ground truth. Nothing produced here is a real patient or a real finding.

Coordinate convention: DICOM patient coordinates (LPS, millimetres).
  +x -> patient Left, +y -> Posterior, +z -> Superior.
A phantom is defined in a patient-fixed "anatomy frame". Each examination
samples that anatomy through a rigid transform (simulated repositioning), so
two examinations of the same synthetic patient are NOT voxel-aligned and need
registration — exactly as with real serial scans.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
SYNTHETIC_TAG = "SYNTHETIC PHANTOM - NOT A REAL PATIENT"


# --------------------------------------------------------------------------- #
# Anatomy description
# --------------------------------------------------------------------------- #
@dataclass
class Lesion:
    lesion_id: str
    center: list            # anatomy frame, mm
    radii: list             # semi-axes, mm (t1 size)
    hu: float
    change: str             # increased|decreased|stable|new|resolved
    scale_t2: float         # radius multiplier at t2 (0 => absent)
    present_t1: bool = True


@dataclass
class Anatomy:
    patient_id: str
    body_axes: list         # (a_x, a_y) mm
    lung_centers: list      # [(x,y,z) right, (x,y,z) left]
    lung_axes: list         # [(ax,ay,az) right, left]
    vessels: list           # list of (p0, p1, radius)
    lesions: list = field(default_factory=list)


def _rng(seed):
    return np.random.default_rng(seed)


def make_anatomy(patient_id: str, abnormal: bool, seed: int) -> Anatomy:
    r = _rng(seed)
    body = [r.uniform(150, 170), r.uniform(105, 120)]
    lz = r.uniform(-5, 5)
    lung_axes = [[r.uniform(52, 60), r.uniform(62, 72), r.uniform(95, 110)] for _ in range(2)]
    lung_centers = [[-r.uniform(62, 70), r.uniform(-2, 6), lz],
                    [r.uniform(62, 70), r.uniform(-2, 6), lz]]
    vessels = []
    for side in range(2):
        cx, cy, cz = lung_centers[side]
        hilum = np.array([cx * 0.45, cy, cz])
        for _ in range(9):
            d = r.normal(size=3)
            d /= np.linalg.norm(d)
            length = r.uniform(35, 75)
            p1 = hilum + d * length
            vessels.append((hilum.tolist(), p1.tolist(), float(r.uniform(1.5, 3.0))))
    anat = Anatomy(patient_id, body, lung_centers, lung_axes, vessels)
    if abnormal:
        n = int(r.integers(1, 3))
        changes = ["increased", "decreased", "stable", "new", "resolved"]
        for k in range(n):
            side = int(r.integers(0, 2))
            c = np.array(lung_centers[side], float)
            ax = np.array(lung_axes[side], float)
            # sample a point well inside the lung ellipsoid
            while True:
                u = r.uniform(-0.6, 0.6, size=3)
                if np.sum(u ** 2) < 0.36:
                    break
            center = (c + u * ax).tolist()
            rad = float(r.uniform(5.0, 11.0))
            radii = [rad * r.uniform(0.9, 1.1) for _ in range(3)]
            change = changes[int(r.integers(0, len(changes)))]
            scale = {"increased": r.uniform(1.25, 1.55), "decreased": r.uniform(0.55, 0.78),
                     "stable": r.uniform(0.98, 1.02), "new": 1.0, "resolved": 0.0}[change]
            anat.lesions.append(Lesion(f"{patient_id}-L{k+1}", center, radii,
                                       float(r.uniform(15, 60)), change, float(scale),
                                       present_t1=(change != "new")))
    return anat


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _seg_dist(P, a, b):
    a = np.asarray(a); b = np.asarray(b)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    proj = a + t[..., None] * ab
    return np.linalg.norm(P - proj, axis=-1)


def rigid_matrix(rot_deg, trans):
    ax, ay, az = np.deg2rad(rot_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx, np.asarray(trans, float)


def lesion_state(les: Lesion, timepoint: int):
    """Return radii at a timepoint, or None if absent."""
    if timepoint == 1:
        return list(les.radii) if les.present_t1 else None
    if les.scale_t2 <= 0:
        return None
    return [r_ * les.scale_t2 for r_ in les.radii]


def render(anat: Anatomy, timepoint: int, shape_zyx, spacing_xyz, origin_xyz,
           rot_deg, trans_mm, noise_hu, seed):
    """Render HU volume (z,y,x) and integer lesion label map for one exam.

    World point X maps to anatomy frame A = R^T (X - t).
    """
    nz, ny, nx = shape_zyx
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    R, t = rigid_matrix(rot_deg, trans_mm)
    zz, yy, xx = np.meshgrid(oz + np.arange(nz) * sz, oy + np.arange(ny) * sy,
                             ox + np.arange(nx) * sx, indexing="ij")
    W = np.stack([xx, yy, zz], -1)
    A = (W - t) @ R          # row-vector form of R^T (W - t)
    X, Y, Z = A[..., 0], A[..., 1], A[..., 2]

    hu = np.full(shape_zyx, -1000.0, np.float32)
    bx, by = anat.body_axes
    body = (X / bx) ** 2 + (Y / by) ** 2 <= 1.0
    inner = (X / (bx - 12)) ** 2 + (Y / (by - 12)) ** 2 <= 1.0
    hu[body] = -100.0                      # subcutaneous fat
    hu[inner] = 40.0                       # soft tissue
    spine = ((X / 18) ** 2 + ((Y - by + 35) / 18) ** 2) <= 1.0
    hu[spine & inner] = 650.0
    lungs = np.zeros(shape_zyx, bool)
    for c, a in zip(anat.lung_centers, anat.lung_axes):
        e = ((X - c[0]) / a[0]) ** 2 + ((Y - c[1]) / a[1]) ** 2 + ((Z - c[2]) / a[2]) ** 2
        lungs |= e <= 1.0
    lungs &= inner
    hu[lungs] = -860.0
    for p0, p1, rad in anat.vessels:
        d = _seg_dist(A, p0, p1)
        v = (d <= rad) & lungs
        hu[v] = np.maximum(hu[v], -80.0 + 40.0 * (1 - d[v] / rad))
    labels = np.zeros(shape_zyx, np.uint8)
    for k, les in enumerate(anat.lesions, start=1):
        radii = lesion_state(les, timepoint)
        if radii is None:
            continue
        c = les.center
        e = ((X - c[0]) / radii[0]) ** 2 + ((Y - c[1]) / radii[1]) ** 2 + ((Z - c[2]) / radii[2]) ** 2
        m = e <= 1.0
        labels[m] = k
        # soft edge: partial-volume ramp in the outer shell
        shell = (e > 1.0) & (e <= 1.25) & lungs
        hu[m] = les.hu
        hu[shell] = np.maximum(hu[shell], -860 + (les.hu + 860) * (1.25 - e[shell]) / 0.25)
    r = _rng(seed)
    noise = r.normal(0, noise_hu, size=shape_zyx).astype(np.float32)
    hu = hu + noise
    return np.clip(hu, -1024, 3071).astype(np.float32), labels


# --------------------------------------------------------------------------- #
# DICOM writing
# --------------------------------------------------------------------------- #
def write_dicom_series(hu_zyx, spacing_xyz, origin_xyz, out_dir: Path, *, patient_id,
                       study_uid, series_uid, study_date, series_desc="SYN CHEST CT",
                       slope=1.0, intercept=-1024.0, shuffle_seed=0):
    """Write a 16-bit CT series. File names and InstanceNumbers are shuffled on
    purpose so that loaders must sort on ImagePositionPatient."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nz = hu_zyx.shape[0]
    order = _rng(shuffle_seed).permutation(nz)
    frame_uid = generate_uid()
    for k in range(nz):
        stored = np.round((hu_zyx[k] - intercept) / slope)
        stored = np.clip(stored, 0, 65535).astype(np.uint16)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE
        sop = generate_uid()
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        fname = out_dir / f"IMG_{int(order[k]):04d}.dcm"
        ds = FileDataset(str(fname), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CT_IMAGE_STORAGE
        ds.SOPInstanceUID = sop
        ds.Modality = "CT"
        ds.PatientName = "SYNTHETIC^PHANTOM"
        ds.PatientID = patient_id
        ds.PatientIdentityRemoved = "YES"
        ds.ImageComments = SYNTHETIC_TAG
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.StudyDate = study_date
        ds.SeriesDescription = series_desc
        ds.SeriesNumber = 2
        ds.InstanceNumber = int(order[k]) + 1          # deliberately not spatial
        ds.ImagePositionPatient = [float(origin_xyz[0]), float(origin_xyz[1]),
                                   float(origin_xyz[2] + k * spacing_xyz[2])]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [float(spacing_xyz[1]), float(spacing_xyz[0])]  # row, col
        ds.SliceThickness = float(spacing_xyz[2])
        ds.Rows, ds.Columns = stored.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleIntercept = float(intercept)
        ds.RescaleSlope = float(slope)
        ds.RescaleType = "HU"
        ds.KVP = 120
        ds.ConvolutionKernel = "SYN_B"
        ds.PixelData = stored.tobytes()
        ds.save_as(str(fname), enforce_file_format=True)
    return out_dir


# --------------------------------------------------------------------------- #
# QA generation (reference answers derived ONLY from ground truth)
# --------------------------------------------------------------------------- #
def _laterality(x_world):
    return "left" if x_world > 0 else "right"   # LPS: +x = patient left


def _zone(z_anat, anat: Anatomy):
    zc = anat.lung_centers[0][2]; az = max(a[2] for a in anat.lung_axes)
    rel = (z_anat - (zc - az)) / (2 * az)
    return "lower" if rel < 1 / 3 else ("middle" if rel < 2 / 3 else "upper")


def build_generator_qa(anat, exams):
    """Return QA pairs for each exam and for the temporal pair."""
    qa = []
    for ex in exams:
        present = [les for les in anat.lesions if lesion_state(les, ex["timepoint"]) is not None]
        ex_id = ex["exam_id"]
        qa.append(dict(scope="single", exam_id=ex_id, category="existence",
                       question="Is there a suspicious region in this scan?",
                       answer="yes" if present else "no"))
        if present:
            sides = sorted({_laterality(les.center[0]) for les in present})
            lat = sides[0] if len(sides) == 1 else "both"
            main = max(present, key=lambda l_: np.prod(lesion_state(l_, ex["timepoint"])))
        else:
            lat, main = "none", None
        qa.append(dict(scope="single", exam_id=ex_id, category="localization",
                       question="Which lung contains the suspicious finding?", answer=lat))
        qa.append(dict(scope="single", exam_id=ex_id, category="localization",
                       question="Which zone of the lung contains the largest suspicious region?",
                       answer=_zone(main.center[2], anat) if main else "none"))
        qa.append(dict(scope="single", exam_id=ex_id, category="evidence",
                       question="Where is the suspicious region?",
                       answer=main.lesion_id if main else "none",
                       reference_lesion=main.lesion_id if main else None))
    e1, e2 = exams
    if anat.lesions:
        def size_at(l_, tp):
            s = lesion_state(l_, tp)
            return 0 if s is None else np.prod(s)
        main = max(anat.lesions, key=lambda l_: max(size_at(l_, 1), size_at(l_, 2)))
        change = main.change
    else:
        main, change = None, "none"
    qa.append(dict(scope="temporal", exam_id=e2["exam_id"], prior_exam_id=e1["exam_id"],
                   category="progression",
                   question="Has the suspected lesion increased or decreased in size since the previous scan?",
                   answer=change, reference_lesion=main.lesion_id if main else None))
    any_new = any(l_.change == "new" for l_ in anat.lesions)
    qa.append(dict(scope="temporal", exam_id=e2["exam_id"], prior_exam_id=e1["exam_id"],
                   category="comparison",
                   question="Is there a new suspicious region compared with the previous examination?",
                   answer="yes" if any_new else "no"))
    return qa


# --------------------------------------------------------------------------- #
# Cohort generation
# --------------------------------------------------------------------------- #
def generate_cohort(out_root: Path, n_patients=36, frac_abnormal=0.67, seed=7,
                    shape_xy=128, write_dicom=True):
    import SimpleITK as sitk
    out_root = Path(out_root)
    dicom_root = out_root / "dicom"
    gt_root = out_root / "ground_truth"
    gt_root.mkdir(parents=True, exist_ok=True)
    r = _rng(seed)
    manifest, all_qa, lesions_table = [], [], []
    n_abn = int(round(n_patients * frac_abnormal))
    flags = np.array([True] * n_abn + [False] * (n_patients - n_abn))
    r.shuffle(flags)
    for p in range(n_patients):
        pid = f"SYN-{p+1:04d}"
        anat = make_anatomy(pid, bool(flags[p]), seed * 1000 + p)
        base_date = _dt.date(2024, 1, 1) + _dt.timedelta(days=int(r.integers(0, 300)))
        exams = []
        for tp in (1, 2):
            sxy = float(r.uniform(2.6, 2.9))
            sz = float(r.choice([2.5, 3.0, 3.5]))
            nz = int(math.ceil(250 / sz))
            spacing = (sxy, sxy, sz)
            origin = (-sxy * shape_xy / 2, -sxy * shape_xy / 2 + r.uniform(-8, 8), -125.0 + r.uniform(-6, 6))
            if tp == 1:
                rot, trans = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            else:
                rot = tuple(r.uniform(-4, 4, size=3).tolist())
                trans = tuple(r.uniform(-10, 10, size=3).tolist())
            noise = float(r.uniform(12, 22))
            hu, lab = render(anat, tp, (nz, shape_xy, shape_xy), spacing, origin, rot, trans,
                             noise, seed * 7919 + p * 2 + tp)
            exam_id = f"{pid}-T{tp}"
            date = (base_date + _dt.timedelta(days=0 if tp == 1 else int(r.integers(60, 240)))).strftime("%Y%m%d")
            study_uid, series_uid = generate_uid(), generate_uid()
            series_dir = dicom_root / pid / exam_id
            if write_dicom:
                write_dicom_series(hu, spacing, origin, series_dir, patient_id=pid, study_uid=study_uid,
                                   series_uid=series_uid, study_date=date, shuffle_seed=p * 10 + tp)
            img = sitk.GetImageFromArray(lab)
            img.SetSpacing(spacing); img.SetOrigin(origin)
            sitk.WriteImage(img, str(gt_root / f"{exam_id}_lesions.nii.gz"))
            exams.append(dict(exam_id=exam_id, patient_id=pid, timepoint=tp, study_date=date,
                              dicom_dir=str(series_dir.relative_to(out_root)),
                              lesion_mask=str((gt_root / f"{exam_id}_lesions.nii.gz").relative_to(out_root)),
                              abnormal=int(lab.max() > 0), spacing=list(spacing),
                              sim_rotation_deg=list(rot), sim_translation_mm=list(trans)))
            R, t = rigid_matrix(rot, trans)
            for k, les in enumerate(anat.lesions, start=1):
                radii = lesion_state(les, tp)
                vox = int((lab == k).sum())
                world_c = (R @ np.asarray(les.center) + t).tolist()
                lesions_table.append(dict(exam_id=exam_id, patient_id=pid, timepoint=tp,
                                          lesion_id=les.lesion_id, label=k, change=les.change,
                                          present=radii is not None,
                                          center_world_mm=world_c,
                                          analytic_volume_mm3=(4 / 3 * math.pi * float(np.prod(radii))) if radii else 0.0,
                                          voxel_volume_mm3=vox * float(np.prod(spacing)),
                                          laterality=_laterality(les.center[0]),
                                          zone=_zone(les.center[2], anat)))
        manifest.extend(exams)
        all_qa.extend(build_generator_qa(anat, exams))
        (gt_root / f"{pid}_anatomy.json").write_text(json.dumps(asdict(anat), indent=1))
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_root / "qa_pairs.json").write_text(json.dumps(all_qa, indent=1))
    (out_root / "lesions.json").write_text(json.dumps(lesions_table, indent=1))
    (out_root / "README_SYNTHETIC.txt").write_text(
        "All data in this folder is SYNTHETIC (analytic phantoms). It contains no patient "
        "information and no real clinical findings. Generated by src/data/synthetic.py.\n")
    return manifest
