"""Run the full pipeline on one or two DICOM series from the command line.

Single exam:   python scripts/run_inference.py --later path/to/dicom_dir -q "Where is the suspicious region?"
Paired exams:  python scripts/run_inference.py --earlier dirA --later dirB -q "Has the lesion grown?"
Writes JSON (and an evidence PNG) to --out.
"""
import argparse
import json
from pathlib import Path

from _common import ROOT
from src.inference.pipeline import Orchestrator
from src.inference.visualize import evidence_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--later", required=True)
    ap.add_argument("--earlier")
    ap.add_argument("-q", "--question", action="append", required=True)
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--out", default="reports/inference")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    o = Orchestrator(a.config, root=ROOT)
    later = o.analyze_dicom(a.later, "later")
    earlier = o.analyze_dicom(a.earlier, "earlier") if a.earlier else None
    tctx = o.temporal(earlier, later) if earlier else None
    answers = [o.ask(q, later, earlier, tctx) for q in a.question]
    rec = dict(warnings=o.warnings, later=later.summary(), earlier=earlier.summary() if earlier else None,
               registration=tctx["registration"] if tctx else None,
               entities=tctx["entities"] if tctx else None, answers=answers)
    (out / "result.json").write_text(json.dumps(rec, indent=1, default=float))
    evidence_figure(later, tctx, out / "evidence.png", thr=o.threshold)
    for ans in answers:
        print(f"Q: {ans['question']}\nA: {ans['answer_text']}\n")
    print("wrote", out / "result.json", out / "evidence.png")


if __name__ == "__main__":
    main()
