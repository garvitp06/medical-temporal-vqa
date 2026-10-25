"""Train the ContextFlow++-style specialist on top of the FROZEN GLOW generalist.
Usage: python scripts/train_contextflow.py --config configs/contextflow.yaml"""
import argparse

import torch

from _common import ROOT, data_context, load_yaml
from src.data.dataset import FeaturePatchDataset, PatchDataset
from src.models.anomaly_detector import build_glow
from src.models.contextflow import ContextFlowPP
from src.training.flow_trainer import train_flow
from src.training.utils import RunLog, set_seed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/contextflow.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / a.config)
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    set_seed(cfg["seed"])
    gck = torch.load(ROOT / cfg["generalist_checkpoint"], map_location="cpu", weights_only=False)
    cfg["model"] = gck["config"]["model"]
    g = build_glow(cfg["model"]); g.load_state_dict(gck["state_dict"])
    cf = cfg["contextflow"]
    model = ContextFlowPP(g, 1, cf["discrete_cardinalities"], cf["emb_dim"], cf["cn_hidden"], cf["discrete_mode"])
    _, _, split, store, by = data_context()
    tc = cfg["train"]
    if cfg["model"].get("feature_space"):          # specialist on pretrained-feature crops
        fdir = ROOT / "experiments/feature_extractor"
        tr = FeaturePatchDataset(fdir / "train_crops.pt", tc.get("feature_noise_std", 0.0))
        va = FeaturePatchDataset(fdir / "val_crops.pt")
    else:
        tr = PatchDataset(store, by["train"], tc["patch"], cfg["model"]["ctx_slices"], tc["samples_per_epoch"], tc["protocol"], seed=cfg["seed"] + 10)
        va = PatchDataset(store, by["val"], tc["patch"], cfg["model"]["ctx_slices"], tc["val_samples"], tc["protocol"], seed=cfg["seed"] + 1, fixed=True)
    out = ROOT / cfg["out_dir"]; out.mkdir(parents=True, exist_ok=True)
    run = RunLog(out, cfg, "contextflow")
    run.set(generalist_val_nll=gck["val_nll"], n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            note="generalist frozen; specialist = context encoder + additive conditional network")
    train_flow(model, tr, va, cfg, str(out), run)


if __name__ == "__main__":
    main()
