"""Build VQA training/evaluation tensors from processed exams, cached
registrations and detector anomaly maps."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from ..data.preprocessing import resample_like, window_normalize
from ..inference.features import build_input, lung_crop_box, token_of_voxel
from ..inference.localization import physical_to_index
from ..models.vqa import TASKS, qa_task


def load_map(root: Path, kind: str, exam_id: str):
    return np.load(root / "experiments" / kind / "maps" / f"{exam_id}.npz")["map"].astype(np.float32)


class VQATensorCache:
    """Per-exam single tensors and per-pair temporal tensors (earlier warped
    onto the later grid with the cached registration transform)."""

    def __init__(self, root, store, manifest, lesions, detector_kind, use_anomaly=True,
                 register=True, pre_window=(-600, 1500, "window_minus1_1")):
        self.root, self.store, self.kind, self.ua, self.register = Path(root), store, detector_kind, use_anomaly, register
        self.les = defaultdict(list)
        for l in lesions:
            self.les[l["exam_id"]].append(l)
        self.by_id = {r["exam_id"]: r for r in manifest}
        self.win = pre_window
        self.single, self.boxes, self.temporal = {}, {}, {}

    def save(self, path):
        torch.save(dict(single=self.single, boxes=self.boxes, temporal=self.temporal, ua=self.ua,
                        kind=self.kind, register=self.register), path)

    def load(self, path):
        d = torch.load(path, weights_only=False)
        if (d["ua"], d["kind"], d["register"]) == (self.ua, self.kind, self.register):
            self.single, self.boxes, self.temporal = d["single"], d["boxes"], d["temporal"]
            return True
        return False

    def single_input(self, exam_id):
        if exam_id not in self.single:
            norm = self.store.array(exam_id, "norm"); lung = self.store.array(exam_id, "lung")
            amap = load_map(self.root, self.kind, exam_id) if self.ua else None
            box = lung_crop_box(lung)
            x, _ = build_input(norm, lung, amap, box, use_anomaly=self.ua)
            self.single[exam_id], self.boxes[exam_id] = x, box
        return self.single[exam_id], self.boxes[exam_id]

    def temporal_input(self, earlier_id, later_id, patient_id):
        key = (earlier_id, later_id)
        if key not in self.temporal:
            x_l, box = self.single_input(later_id)
            fixed = self.store.image(later_id); moving = self.store.image(earlier_id)
            tfm_path = self.root / "experiments/registration" / f"{patient_id}.tfm"
            tfm = sitk.ReadTransform(str(tfm_path)) if (self.register and tfm_path.exists()) else None
            hu_e = sitk.GetArrayFromImage(resample_like(moving, fixed, transform=tfm))
            lung_img = self.store.image(earlier_id, "lung")
            lung_e = sitk.GetArrayFromImage(resample_like(lung_img, fixed, is_mask=True, transform=tfm))
            amap_e = None
            if self.ua:
                a = sitk.GetImageFromArray(load_map(self.root, self.kind, earlier_id)); a.CopyInformation(moving)
                amap_e = sitk.GetArrayFromImage(resample_like(a, fixed, transform=tfm, default=0.0))
            norm_e = window_normalize(hu_e, *self.win)
            x_e, _ = build_input(norm_e, lung_e, amap_e, box, use_anomaly=self.ua)
            self.temporal[key] = (x_e, x_l)
        return self.temporal[key]

    def region_token(self, exam_id, lesion_id):
        if not lesion_id:
            return -1
        img = self.store.image(exam_id)
        _, box = self.single_input(exam_id)
        for l in self.les[exam_id]:
            if l["lesion_id"] == lesion_id and l["present"]:
                zyx = physical_to_index(img, l["center_world_mm"])
                return token_of_voxel(zyx, box)
        return -1


def build_items(qa, split, split_name, cache: VQATensorCache, manifest):
    items = []
    for q in qa:
        pid = q["exam_id"].rsplit("-T", 1)[0]
        if split.get(pid) != split_name or q["exam_id"] not in cache.by_id:
            continue
        task = qa_task(q)
        region = -1
        if q.get("reference_lesion"):
            region = cache.region_token(q["exam_id"], q["reference_lesion"])
        items.append(dict(patient_id=pid, task=task, task_id=TASKS.index(task), question=q["question"],
                          answer=q["answer"], exam_id=q["exam_id"], prior_exam_id=q.get("prior_exam_id"),
                          region=region, scope=q["scope"]))
    return items
