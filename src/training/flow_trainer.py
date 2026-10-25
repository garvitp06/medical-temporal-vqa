"""Maximum-likelihood training loop shared by GLOW and the ContextFlow++-style
specialist. Tracks train/val NLL (nats/dim), gradient norms, non-finite
steps, LR, epoch timing; selects the checkpoint with the lowest val NLL."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.contextflow import ContextFlowPP
from ..models.glow import nll_per_dim
from .utils import RunLog


def _forward(model, x, cont, disc):
    if isinstance(model, ContextFlowPP):
        return model(x, cont, disc)
    return model(x)


@torch.no_grad()
def evaluate_nll(model, loader, device):
    model.eval()
    vals = []
    for x, cont, disc in loader:
        x, cont, disc = x.to(device), cont.to(device), disc.to(device)
        lp, _ = _forward(model, x, cont, disc)
        vals.append(nll_per_dim(lp, x).cpu())
    v = torch.cat(vals)
    return float(v.mean()), float(v.std())


def train_flow(model, train_ds, val_ds, cfg, out_dir, run: RunLog, device="cpu", extra_ckpt=None):
    tc = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=tc["lr"])
    steps_per_epoch = math.ceil(len(train_ds) / tc["batch_size"])
    total = steps_per_epoch * tc["epochs"]
    warm = tc.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    tl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=tc["batch_size"], num_workers=0)
    if not isinstance(model, ContextFlowPP):          # ActNorm data-dependent init
        x0, _, _ = next(iter(tl))
        model.train(); model(x0.to(device))
    best = float("inf"); step = 0
    for ep in range(tc["epochs"]):
        model.train()
        t0 = time.time(); losses, gnorms, nonfinite = [], [], 0
        for x, cont, disc in tl:
            x, cont, disc = x.to(device), cont.to(device), disc.to(device)
            lp, _ = _forward(model, x, cont, disc)
            loss = nll_per_dim(lp, x).mean()
            if not torch.isfinite(loss):
                nonfinite += 1; opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            g = torch.nn.utils.clip_grad_norm_(params, tc.get("grad_clip", 50.0))
            if not torch.isfinite(g):
                nonfinite += 1; opt.zero_grad(); continue
            opt.step(); sched.step(); step += 1
            losses.append(loss.item()); gnorms.append(float(g))
        val_m, val_s = evaluate_nll(model, vl, device)
        rec = dict(epoch=ep + 1, step=step, train_nll=float(np.mean(losses)) if losses else float("nan"),
                   val_nll=val_m, val_nll_std=val_s, grad_norm_mean=float(np.mean(gnorms)) if gnorms else None,
                   grad_norm_max=float(np.max(gnorms)) if gnorms else None, nonfinite_steps=nonfinite,
                   lr=opt.param_groups[0]["lr"], epoch_s=round(time.time() - t0, 1))
        run.log(**rec)
        print(rec, flush=True)
        if val_m < best:
            best = val_m
            ck = dict(state_dict=model.state_dict(), config=cfg, epoch=ep + 1, val_nll=val_m)
            if extra_ckpt:
                ck.update(extra_ckpt)
            torch.save(ck, f"{out_dir}/best.pt")
    run.set(best_val_nll=best, checkpoint=f"{out_dir}/best.pt",
            checkpoint_selection="lowest validation NLL (nats/dim) on held-out validation patients")
    return best
