"""Train GLOW on frozen pretrained lungmask features (run extract_features.py first).
Usage: python scripts/train_feature_glow.py --config configs/feature_glow.yaml"""
import argparse

import torch

from _common import ROOT, load_yaml
from src.data.dataset import FeaturePatchDataset
from src.models.anomaly_detector import build_glow
from src.training.flow_trainer import train_flow
from src.training.utils import RunLog, set_seed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/feature_glow.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    set_seed(cfg["seed"])
    fdir = ROOT / "experiments/feature_extractor"
    tr = FeaturePatchDataset(fdir / "train_crops.pt", cfg["train"].get("feature_noise_std", 0.0))
    va = FeaturePatchDataset(fdir / "val_crops.pt")
    out = ROOT / cfg["out_dir"]; out.mkdir(parents=True, exist_ok=True)
    run = RunLog(out, cfg, "feature_glow")
    model = build_glow(cfg["model"])
    run.set(n_parameters=sum(p.numel() for p in model.parameters()), n_train_crops=len(tr), n_val_crops=len(va),
            feature_extractor="pretrained lungmask R231 (frozen) + PCA(64) fitted on train normal exams",
            dataset="synthetic_chest_ct_v1 (SYNTHETIC)")
    train_flow(model, tr, va, cfg, str(out), run)


if __name__ == "__main__":
    main()
