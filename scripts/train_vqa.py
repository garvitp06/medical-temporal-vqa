"""Train the question router, the single-exam VQA model and the temporal VQA
model; fit temperature scaling on validation; save one checkpoint.

Usage: python scripts/train_vqa.py --detector contextflow [--no-anomaly] [--no-register]
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from _common import ROOT, data_context, load_yaml
from src.models.vqa import (ANSWERS, PARAPHRASES, TASKS, VQAModel, answer_mask, default_tokenizer,
                            train_router)
from src.training.utils import RunLog, set_seed
from src.training.vqa_data import VQATensorCache, build_items

SWAP = {"left": "right", "right": "left"}


def norm_answer(it):
    return ("region" if it["answer"] != "none" else "none") if it["task"] == "where" else it["answer"]


def flip_token(tok, g=(4, 6, 6)):
    if tok < 0:
        return tok
    cz, rem = divmod(tok, g[1] * g[2]); cy, cx = divmod(rem, g[2])
    return (cz * g[1] + cy) * g[2] + (g[2] - 1 - cx)


def batch_tensors(items, cache, temporal, augment, rng):
    xs_l, xs_e, q, t, y, reg = [], [], [], [], [], []
    for it in items:
        if temporal:
            x_e, x_l = cache.temporal_input(it["prior_exam_id"], it["exam_id"], it["patient_id"])
        else:
            x_l, _ = cache.single_input(it["exam_id"]); x_e = None
        ans, r = norm_answer(it), it["region"]
        text = it["question"]
        if augment:
            if rng.random() < 0.5:
                text = rng.choice(PARAPHRASES[it["task"]])
            if rng.random() < 0.5:                   # left/right flip (x axis = last dim)
                x_l = torch.flip(x_l, dims=[-1]); x_e = torch.flip(x_e, dims=[-1]) if x_e is not None else None
                ans = SWAP.get(ans, ans); r = flip_token(r)
        xs_l.append(x_l); xs_e.append(x_e); q.append(text); t.append(it["task_id"])
        y.append(ANSWERS.index(ans)); reg.append(r)
    out = dict(x_l=torch.stack(xs_l), q=q, t=torch.tensor(t), y=torch.tensor(y), region=torch.tensor(reg))
    out["x_e"] = torch.stack(xs_e) if temporal else None
    return out


def run_model(model, tok, b):
    logits, region = model(tok.batch(b["q"]), b["t"], b["x_l"], b["x_e"])
    return logits + answer_mask(b["t"]), region


def train_one(model, tok, items, val_items, cache, temporal, cfg, run, tag):
    opt = torch.optim.AdamW(model.parameters(), cfg["lr"], weight_decay=1e-4)
    rng = random.Random(cfg["seed"])
    best, best_state = -1, None
    for ep in range(cfg["epochs"]):
        model.train(); rng.shuffle(items); losses = []
        for i in range(0, len(items), cfg["batch_size"]):
            b = batch_tensors(items[i:i + cfg["batch_size"]], cache, temporal, cfg["paraphrase_augmentation"], rng)
            logits, region = run_model(model, tok, b)
            loss = F.cross_entropy(logits, b["y"])
            m = b["region"] >= 0
            if m.any():
                loss = loss + cfg["region_loss_weight"] * F.cross_entropy(region[m], b["region"][m])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(loss.item())
        acc = evaluate(model, tok, val_items, cache, temporal)["accuracy"]
        run.log(model=tag, epoch=ep + 1, train_loss=float(np.mean(losses)), val_accuracy=acc)
        print(tag, ep + 1, f"loss {np.mean(losses):.3f} val_acc {acc:.3f}", flush=True)
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best


@torch.no_grad()
def predict(model, tok, items, cache, temporal):
    model.eval()
    logits, regions = [], []
    for i in range(0, len(items), 16):
        b = batch_tensors(items[i:i + 16], cache, temporal, False, None)
        lg, rg = run_model(model, tok, b)
        logits.append(lg); regions.append(rg)
    return (torch.cat(logits), torch.cat(regions)) if logits else (torch.zeros(0, len(ANSWERS)), None)


def evaluate(model, tok, items, cache, temporal):
    if not items:
        return dict(accuracy=float("nan"))
    lg, _ = predict(model, tok, items, cache, temporal)
    y = torch.tensor([ANSWERS.index(norm_answer(it)) for it in items])
    return dict(accuracy=float((lg.argmax(1) == y).float().mean()))


def fit_temperature(logits, y):
    best = (1.0, float("inf"))
    for T in np.linspace(0.3, 5.0, 95):
        nll = float(F.cross_entropy(logits / T, y))
        if nll < best[1]:
            best = (float(T), nll)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vqa.yaml")
    ap.add_argument("--detector", default="contextflow")
    ap.add_argument("--no-anomaly", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / a.config)
    if a.epochs:
        cfg["epochs"] = a.epochs
    set_seed(cfg["seed"])
    dcfg, manifest, split, store, _ = data_context()
    qa = json.loads((ROOT / dcfg["qa_pairs"]).read_text())
    lesions = json.loads((ROOT / dcfg["lesions"]).read_text())
    ua = not a.no_anomaly
    cache = VQATensorCache(ROOT, store, manifest, lesions, a.detector, use_anomaly=ua)
    tag = "vqa_with_anomaly" if ua else "vqa_without_anomaly"
    out = ROOT / "experiments/vqa"; out.mkdir(parents=True, exist_ok=True)
    cache_f = out / f"tensor_cache_{tag}.pt"
    cached = cache_f.exists() and cache.load(cache_f)
    run = RunLog(out / tag, dict(cfg, detector=a.detector, use_anomaly=ua), tag)
    tok = default_tokenizer()
    router = train_router(tok, seed=cfg["seed"])
    sets = {s: build_items(qa, split, s, cache, manifest) for s in ("train", "val", "test")}
    run.set(n_items={s: dict(single=sum(i["scope"] == "single" for i in v), temporal=sum(i["scope"] == "temporal" for i in v))
                     for s, v in sets.items()})
    if not cached:
        for it in sets["train"] + sets["val"] + sets["test"]:      # build every input once
            if it["scope"] == "temporal":
                cache.temporal_input(it["prior_exam_id"], it["exam_id"], it["patient_id"])
            else:
                cache.single_input(it["exam_id"])
        cache.save(cache_f)
    inch = 3 if ua else 2
    models, temps = {}, {}
    width = cfg.get("width", 16)
    for temporal in (False, True):
        name = "temporal" if temporal else "single"
        tr = [i for i in sets["train"] if (i["scope"] == "temporal") == temporal]
        va = [i for i in sets["val"] if (i["scope"] == "temporal") == temporal]
        t0 = time.time()
        torch.manual_seed(cfg["seed"])
        m = VQAModel(tok, inch, grid=tuple(cfg["grid"]), temporal=temporal, width=width)
        tcfg = dict(cfg)
        if temporal and cfg.get("init_temporal_from_single", True):
            # transfer encoder, question embedding, fusion and heads from the trained single-exam model
            missing = m.load_state_dict(models["single"].state_dict(), strict=False)
            run.log(model=name, note="initialised from single-exam model", missing_keys=list(missing.missing_keys))
            tcfg["lr"] = cfg.get("temporal_lr", cfg["lr"])
        tcfg["epochs"] = cfg.get("temporal_epochs", cfg["epochs"]) if temporal else cfg["epochs"]
        best = train_one(m, tok, tr, va, cache, temporal, tcfg, run, name)
        lg, _ = predict(m, tok, va, cache, temporal)
        y = torch.tensor([ANSWERS.index(norm_answer(i)) for i in va])
        temps[name] = fit_temperature(lg, y)
        models[name] = m
        run.log(model=name, best_val_accuracy=best, temperature=temps[name], train_seconds=round(time.time() - t0, 1))
        torch.save(dict(model=name, state=m.state_dict(), temperature=temps[name]), out / f"{tag}_{name}_partial.pt")
    ck = dict(vocab=tok.vocab, router=router.state_dict(), single=models["single"].state_dict(),
              temporal=models["temporal"].state_dict(), use_anomaly=ua, detector=a.detector,
              temperature_single=temps["single"], temperature_temporal=temps["temporal"],
              calibrated_on_val=True, config=cfg, width=width,
              note="temperature fitted on SYNTHETIC validation patients; not calibrated for clinical data")
    torch.save(ck, out / f"{tag}.pt")
    run.set(checkpoint=str(out / f"{tag}.pt"), checkpoint_selection="best validation answer accuracy per model")
    print("saved", out / f"{tag}.pt")


if __name__ == "__main__":
    main()
