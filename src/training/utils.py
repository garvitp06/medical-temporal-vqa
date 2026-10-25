"""Experiment bookkeeping: seeds, environment capture, JSON logging (spec §14)."""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment():
    import importlib
    vers = {}
    for m in ["torch", "numpy", "scipy", "SimpleITK", "pydicom", "monai", "sklearn", "fastapi", "lungmask"]:
        try:
            mod = importlib.import_module(m)
            vers[m] = getattr(mod, "__version__", None) or getattr(mod, "Version_VersionString", lambda: "?")()
        except Exception:
            vers[m] = None
    return dict(python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor(),
                cpu_count=os.cpu_count(), cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                torch_threads=torch.get_num_threads(), versions=vers)


def load_yaml(p):
    return yaml.safe_load(Path(p).read_text())


class RunLog:
    def __init__(self, out_dir, config: dict, name: str):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.rec = dict(name=name, started=time.strftime("%Y-%m-%dT%H:%M:%S"), config=config,
                        environment=environment(), history=[], notes=[])
        self.t0 = time.time()

    def log(self, **kw):
        self.rec["history"].append(kw)
        self.save()

    def set(self, **kw):
        self.rec.update(kw)
        self.save()

    def save(self):
        self.rec["runtime_s"] = round(time.time() - self.t0, 1)
        (self.dir / "run.json").write_text(json.dumps(self.rec, indent=1, default=str))
