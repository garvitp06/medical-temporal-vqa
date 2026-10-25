"""Fit the PCA of the frozen pretrained lungmask encoder features on TRAIN
normal exams and build feature-crop training/validation sets for the
feature-space flows.

Usage: python scripts/extract_features.py [--slices 12 --crops 6 --crop 32]
"""
import argparse
import json
import time

import numpy as np
import torch

from _common import ROOT, data_context
from src.models.anomaly_detector import thickness_category
from src.models.feature_flow import LungmaskFeatures
from src.training.utils import set_seed


def crops_for(fx, store, exams, n_slices, n_crops, size, rng):
    xs, conts, discs = [], [], []
    for e in exams:
        eid = e["exam_id"]
        hu = store.array(eid, "hu"); lung = store.array(eid, "lung") > 0
        zz = np.where(lung.any((1, 2)))[0]; z0, z1 = zz.min(), zz.max()
        cat = thickness_category(store.meta(eid)["original_meta"]["original_spacing"][2])
        sel = rng.choice(zz, min(n_slices, len(zz)), replace=False)
        feats = fx(torch.from_numpy(hu[sel].astype(np.float32)))
        h = size // 2
        for j, z in enumerate(sel):
            f = torch.nn.functional.pad(feats[j], (h, h, h, h), mode="replicate")
            pts = np.argwhere(lung[z])
            for p in pts[rng.choice(len(pts), n_crops, replace=len(pts) < n_crops)]:
                y, x = p
                xs.append(f[:, y:y + size, x:x + size].half())
                conts.append([(z - z0) / max(1, z1 - z0)]); discs.append([cat])
    return dict(x=torch.stack(xs), cont=torch.tensor(conts, dtype=torch.float32), disc=torch.tensor(discs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=12); ap.add_argument("--crops", type=int, default=6)
    ap.add_argument("--crop", type=int, default=32); ap.add_argument("--components", type=int, default=64)
    ap.add_argument("--protocol", default="normal_only")
    a = ap.parse_args()
    set_seed(0); torch.set_num_threads(2)
    _, _, _, store, by = data_context()
    out = ROOT / "experiments/feature_extractor"; out.mkdir(parents=True, exist_ok=True)
    fx = LungmaskFeatures(str(ROOT / "weights/unet_r231-d5d2fc3d.pth"), a.components)
    pick = lambda ex: [e for e in ex if a.protocol != "normal_only" or not e["abnormal"]]
    tr, va = pick(by["train"]), pick(by["val"])
    rng = np.random.default_rng(0)
    t0 = time.time()
    vecs = []
    for e in tr:                                   # PCA on TRAIN normal lung features only
        hu = store.array(e["exam_id"], "hu"); lung = store.array(e["exam_id"], "lung") > 0
        zs = rng.choice(np.where(lung.any((1, 2)))[0], 4, replace=False)
        f = fx.raw(torch.from_numpy(hu[zs].astype(np.float32)))
        for j, z in enumerate(zs):
            v = f[j].permute(1, 2, 0)[torch.from_numpy(lung[z])]
            vecs.append(v[torch.randperm(len(v))[:600]])
    explained = fx.fit_pca(torch.cat(vecs))
    torch.save(dict(pca_mean=fx.pca_mean, pca_proj=fx.pca_proj, n_components=a.components,
                    explained_variance=explained, fitted_on=[e["exam_id"] for e in tr]), out / "pca.pt")
    print(f"PCA {a.components} comps explains {explained:.4f} ({time.time() - t0:.0f}s)", flush=True)
    for name, ex in (("train", tr), ("val", va)):
        d = crops_for(fx, store, ex, a.slices, a.crops, a.crop, rng)
        torch.save(d, out / f"{name}_crops.pt")
        print(name, tuple(d["x"].shape), f"{time.time() - t0:.0f}s", flush=True)
    (out / "info.json").write_text(json.dumps(dict(args=vars(a), explained_variance=explained,
                                                   n_train_exams=len(tr), n_val_exams=len(va),
                                                   extractor="lungmask U-Net R231 down_path[1]+[2] (frozen, pretrained)"), indent=1))


if __name__ == "__main__":
    main()
