#!/usr/bin/env bash
# Full reproduction: data -> flows -> scoring -> registration -> VQA -> evaluation -> tests.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/download_weights.sh
python scripts/prepare_data.py --workers 2
python scripts/register_pairs.py
python scripts/train_glow.py
python scripts/extract_features.py
python scripts/train_feature_glow.py
python scripts/train_contextflow.py
for k in intensity_baseline glow feature_glow contextflow; do
  python scripts/score_exams.py --kind "$k"
  python scripts/select_candidate_filter.py --kind "$k"
done
python scripts/train_vqa.py --detector contextflow
python scripts/train_vqa.py --detector contextflow --no-anomaly
python scripts/evaluate.py
python -m pytest -q
