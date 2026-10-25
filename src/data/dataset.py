"""Manifest handling, patient-level splitting and training datasets (spec §5).

Manifest: a JSON list of exam records with at least
  exam_id, patient_id, timepoint, dicom_dir, abnormal (0/1/None)
optional: lesion_mask (path), study_date, site, scanner.
Records without annotations are kept; nothing is fabricated for them.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from ..models.anomaly_detector import stack_25d, thickness_category


def load_manifest(path) -> list[dict]:
    return json.loads(Path(path).read_text())


def patient_level_split(manifest, fractions=(0.6, 0.2, 0.2), seed=13, stratify_key="abnormal",
                        held_out_key=None, held_out_values=None):
    """Assign every PATIENT (not exam/slice) to train/val/test.

    Stratified on whether the patient has any abnormal exam. If ``held_out_key``
    (e.g. 'site' or 'scanner') is given, patients whose value is in
    ``held_out_values`` form an extra 'held_out' split."""
    by_pat = defaultdict(list)
    for r in manifest:
        by_pat[r["patient_id"]].append(r)
    groups = defaultdict(list)
    split = {}
    for pid, exams in by_pat.items():
        if held_out_key and any(str(e.get(held_out_key)) in set(map(str, held_out_values or [])) for e in exams):
            split[pid] = "held_out"
            continue
        strat = int(any((e.get(stratify_key) or 0) for e in exams))
        groups[strat].append(pid)
    rng = np.random.default_rng(seed)
    for strat, pids in groups.items():
        pids = sorted(pids)
        rng.shuffle(pids)
        n = len(pids)
        n_tr = int(round(fractions[0] * n)); n_va = int(round(fractions[1] * n))
        for i, pid in enumerate(pids):
            split[pid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return split


def split_summary(manifest, split, qa=None, lesions=None):
    out = {}
    for name in sorted(set(split.values())):
        pats = {p for p, s in split.items() if s == name}
        ex = [r for r in manifest if r["patient_id"] in pats]
        ann = [r for r in ex if r.get("lesion_mask")]
        out[name] = dict(patients=len(pats), examinations=len(ex), volumes=len(ex),
                         abnormal_exams=sum(int(r.get("abnormal") or 0) for r in ex),
                         annotated_exams=len(ann),
                         temporal_pairs=sum(1 for p in pats if sum(r["patient_id"] == p for r in ex) >= 2))
        if qa is not None:
            out[name]["qa_pairs"] = sum(1 for q in qa if q["exam_id"].rsplit("-T", 1)[0] in pats)
        if lesions is not None:
            out[name]["annotated_lesion_instances"] = sum(1 for l in lesions if l["patient_id"] in pats and l["present"])
    return out


def assert_no_patient_leakage(split, manifest):
    seen = {}
    for r in manifest:
        s = split[r["patient_id"]]
        if r["patient_id"] in seen and seen[r["patient_id"]] != s:
            raise AssertionError(f"patient {r['patient_id']} appears in two splits")
        seen[r["patient_id"]] = s
    return True


def anonymize_id(pid: str, salt="mtvqa") -> str:
    return hashlib.sha256((salt + pid).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
class ProcessedStore:
    """Access to cached processed exams written by scripts/prepare_data.py."""

    def __init__(self, root):
        self.root = Path(root)

    def path(self, exam_id, kind):
        return self.root / f"{exam_id}_{kind}.nii.gz"

    def image(self, exam_id, kind="hu") -> sitk.Image:
        return sitk.ReadImage(str(self.path(exam_id, kind)))

    def array(self, exam_id, kind):
        return sitk.GetArrayFromImage(self.image(exam_id, kind))

    def meta(self, exam_id):
        return json.loads((self.root / f"{exam_id}_preproc.json").read_text())

    def has(self, exam_id, kind):
        return self.path(exam_id, kind).exists()


class PatchDataset(Dataset):
    """Random 2.5D patches centred on lung voxels, for flow training.

    protocol='normal_only': only exams with abnormal == 0 (configurable, since
    the original project's protocol is unverified)."""

    def __init__(self, store: ProcessedStore, exams: list[dict], patch=64, ctx_slices=1,
                 samples_per_epoch=4000, protocol="normal_only", seed=0, jitter_noise=1 / 128, fixed=False):
        if protocol == "normal_only":
            exams = [e for e in exams if int(e.get("abnormal") or 0) == 0]
        self.items = []
        for e in exams:
            norm = store.array(e["exam_id"], "norm").astype(np.float32)
            lung = store.array(e["exam_id"], "lung").astype(np.uint8)
            meta = store.meta(e["exam_id"])
            thick = meta["original_meta"]["original_spacing"][2]
            coords = np.argwhere(lung > 0)
            zz = np.where(lung.reshape(lung.shape[0], -1).any(1))[0]
            self.items.append(dict(norm=norm, lung=lung, coords=coords, z0=zz.min(), z1=zz.max(),
                                   cat=thickness_category(thick)))
        if not self.items:
            raise ValueError("no exams available for the selected training protocol")
        self.patch, self.ctx, self.n = patch, ctx_slices, samples_per_epoch
        self.rng = np.random.default_rng(seed)
        self.seed, self.fixed = seed, fixed   # fixed=True -> identical samples every epoch (validation)
        self.noise = jitter_noise

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.fixed:
            self.rng = np.random.default_rng(self.seed * 100003 + i)
        it = self.items[self.rng.integers(len(self.items))]
        z, y, x = it["coords"][self.rng.integers(len(it["coords"]))]
        h = self.patch // 2
        sl = stack_25d(it["norm"], int(z), self.ctx)
        sl = np.pad(sl, ((0, 0), (h, h), (h, h)), constant_values=-1.0)
        p = sl[:, y:y + self.patch, x:x + self.patch].copy()
        if self.rng.random() < 0.5:
            p = p[:, :, ::-1].copy()                      # left/right flip augmentation
        p = p + self.rng.uniform(0, self.noise, size=p.shape).astype(np.float32)   # dequantisation
        pos = (z - it["z0"]) / max(1, it["z1"] - it["z0"])
        return (torch.from_numpy(p).float(), torch.tensor([pos], dtype=torch.float32),
                torch.tensor([it["cat"]], dtype=torch.long))


class FeaturePatchDataset(Dataset):
    """Pre-extracted pretrained-feature crops (see scripts/extract_features.py).
    File holds x (N,C,h,w) float16, cont (N,1), disc (N,1)."""

    def __init__(self, path, noise_std=0.0):
        d = torch.load(path, weights_only=False)
        self.x, self.cont, self.disc = d["x"], d["cont"], d["disc"]
        self.noise = noise_std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i].float()
        if self.noise:
            x = x + self.noise * torch.randn_like(x)
        return x, self.cont[i], self.disc[i]
