#!/usr/bin/env bash
set -euo pipefail

PROJ=${1:-/home/xuxj/2026/dyn_hetero_community}
PY=${2:-python}

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="$PROJ/src:${PYTHONPATH:-}"

mkdir -p "$PROJ/data/processed" "$PROJ/results/smoke"

$PY "$PROJ/scripts/datasets/prepare_dataset.py" \
  --dataset arxiv \
  --archive /home/xuxj/2026/datasets/arxiv.zip \
  --max-edges 40000 \
  --out-pt "$PROJ/data/processed/arxiv_smoke.pt"

$PY -m dhc.train.run_train \
  --data "$PROJ/data/processed/arxiv_smoke.pt" \
  --out-dir "$PROJ/results/smoke/dhc_full" \
  --epochs 3 --batch-size 256 --lr 0.001

$PY -m dhc.train.run_train \
  --data "$PROJ/data/processed/arxiv_smoke.pt" \
  --out-dir "$PROJ/results/smoke/tgn_like" \
  --epochs 3 --batch-size 256 --lr 0.001 \
  --no-relation-memory --no-community

$PY -m dhc.train.run_train \
  --data "$PROJ/data/processed/arxiv_smoke.pt" \
  --out-dir "$PROJ/results/smoke/rel_tgn" \
  --epochs 3 --batch-size 256 --lr 0.001 \
  --no-community

$PY -m dhc.train.run_train \
  --data "$PROJ/data/processed/arxiv_smoke.pt" \
  --out-dir "$PROJ/results/smoke/cdgp_like" \
  --epochs 3 --batch-size 256 --lr 0.001 \
  --no-relation-memory

$PY - <<'PY'
import json
from pathlib import Path
import pandas as pd

base = Path('/home/xuxj/2026/dyn_hetero_community/results/smoke')
rows = []
for name in ['tgn_like', 'rel_tgn', 'cdgp_like', 'dhc_full']:
    m = json.loads((base / name / 'metrics.json').read_text())
    rows.append({
        'model': name,
        'test_mrr': m['test']['mrr'],
        'test_auc': m['test']['auc'],
        'test_ap': m['test']['ap'],
    })

df = pd.DataFrame(rows)
df.to_csv(base / 'baseline_comparison.csv', index=False)
print(df)
PY

$PY "$PROJ/scripts/visualization/plot_training_curves.py" \
  --csv "$PROJ/results/smoke/dhc_full/train_history.csv" \
  --out "$PROJ/results/smoke/dhc_full/training_curve.png" || true

echo "Smoke experiment finished."
