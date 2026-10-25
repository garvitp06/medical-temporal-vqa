"""Shared helpers for scripts (path setup, split/exam lookup)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.utils import load_yaml  # noqa: E402


def data_context(data_cfg="configs/data.yaml"):
    from src.data.dataset import ProcessedStore, load_manifest
    cfg = load_yaml(ROOT / data_cfg)
    manifest = load_manifest(ROOT / cfg["manifest"])
    split = json.loads((ROOT / cfg["split_file"]).read_text())
    store = ProcessedStore(ROOT / cfg["processed_root"])
    manifest = [r for r in manifest if store.has(r["exam_id"], "norm")]
    by_split = {}
    for r in manifest:
        by_split.setdefault(split[r["patient_id"]], []).append(r)
    return cfg, manifest, split, store, by_split
