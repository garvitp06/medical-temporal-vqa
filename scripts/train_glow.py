"""Train the GLOW generalist on 2.5D lung patches.
Usage: python scripts/train_glow.py --config configs/glow.yaml"""
import argparse

import torch

from _common import ROOT, data_context, load_yaml
from src.data.dataset import PatchDataset
from src.models.anomaly_detector import build_glow
from src.training.flow_trainer import train_flow
from src.training.utils import RunLog, set_seed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/glow.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    set_seed(cfg["seed"]); torch.set_num_threads(max(1, torch.get_num_threads()))
    _, _, split, store, by = data_context()
    tc = cfg["train"]
    tr = PatchDataset(store, by["train"], tc["patch"], cfg["model"]["ctx_slices"], tc["samples_per_epoch"], tc["protocol"], seed=cfg["seed"])
    va = PatchDataset(store, by["val"], tc["patch"], cfg["model"]["ctx_slices"], tc["val_samples"], tc["protocol"], seed=cfg["seed"] + 1, fixed=True)
    out = ROOT / cfg["out_dir"]; out.mkdir(parents=True, exist_ok=True)
    run = RunLog(out, cfg, "glow")
    run.set(train_exams=[e["exam_id"] for e in by["train"] if tc["protocol"] != "normal_only" or not e["abnormal"]],
            val_exams=[e["exam_id"] for e in by["val"] if tc["protocol"] != "normal_only" or not e["abnormal"]],
            dataset="synthetic_chest_ct_v1 (SYNTHETIC)")
    model = build_glow(cfg["model"])
    run.set(n_parameters=sum(p.numel() for p in model.parameters()))
    train_flow(model, tr, va, cfg, str(out), run)


if __name__ == "__main__":
    main()
