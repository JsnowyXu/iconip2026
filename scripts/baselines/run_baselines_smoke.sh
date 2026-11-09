#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/xuxj/2026}
PROJ=${2:-/home/xuxj/2026/dyn_hetero_community}
PY=${3:-python}
export KMP_DUPLICATE_LIB_OK=TRUE
TIMEOUT_BIN=$(command -v timeout || true)
TIMEOUT_SEC=${BASELINE_TIMEOUT_SEC:-900}

mkdir -p "$PROJ/results/baseline_status"
STATUS="$PROJ/results/baseline_status/run_status.log"
: > "$STATUS"

log() {
  echo "[$(date +'%F %T')] $1" | tee -a "$STATUS"
}

log "Preparing adapter data from unified arxiv sample"
$PY "$PROJ/scripts/datasets/export_baseline_formats.py" \
  --input "$PROJ/data/processed/arxiv_smoke.pt" \
  --name arxiv_smoke \
  --out-dir "$PROJ/data/interim/baseline_adapters/arxiv_smoke"

# TGN baseline
TGN_DIR="$ROOT/baselines/src/tgn-master/tgn-master"
if [ -d "$TGN_DIR" ]; then
  log "Found TGN baseline at $TGN_DIR"
  mkdir -p "$TGN_DIR/data"
  cp "$PROJ/data/interim/baseline_adapters/arxiv_smoke/tgn/arxiv_smoke.csv" "$TGN_DIR/data/"
  cp "$PROJ/data/interim/baseline_adapters/arxiv_smoke/tgn/ml_arxiv_smoke.csv" "$TGN_DIR/data/"
  cp "$PROJ/data/interim/baseline_adapters/arxiv_smoke/tgn/ml_arxiv_smoke.npy" "$TGN_DIR/data/" || true
  cp "$PROJ/data/interim/baseline_adapters/arxiv_smoke/tgn/ml_arxiv_smoke_node.npy" "$TGN_DIR/data/" || true
  log "TGN adapter data copied. Running 1-epoch smoke test..."
  (
    cd "$TGN_DIR"
    if [ -n "$TIMEOUT_BIN" ]; then
      "$TIMEOUT_BIN" "$TIMEOUT_SEC" python train_self_supervised.py --data arxiv_smoke --n_epoch 1 --bs 200 --n_runs 1 --n_degree 10 --gpu 0
    else
      python train_self_supervised.py --data arxiv_smoke --n_epoch 1 --bs 200 --n_runs 1 --n_degree 10 --gpu 0
    fi
  ) >> "$STATUS" 2>&1 || log "TGN smoke run skipped/failed (see log)."
else
  log "TGN baseline not found"
fi

# CDGP baseline
CDGP_DIR="$ROOT/baselines/src/cdgp/CDGP"
if [ -d "$CDGP_DIR" ]; then
  log "Found CDGP baseline at $CDGP_DIR"
  "$PY" "$PROJ/scripts/baselines/prepare_cdgp_adapter.py" \
    --input "$PROJ/data/processed/arxiv_smoke.pt" \
    --out-dir "$CDGP_DIR/data/arxiv_smoke" \
    --name arxiv_smoke >> "$STATUS" 2>&1 || true
  log "Running 1-epoch CDGP smoke test..."
  (
    cd "$CDGP_DIR"
    if [ -n "$TIMEOUT_BIN" ]; then
      "$TIMEOUT_BIN" "$TIMEOUT_SEC" env KMP_DUPLICATE_LIB_OK=TRUE python train.py --data arxiv_smoke --n_epoch 1 --n_runs 1 --bs 200 --patience 2 --gpu 0
    else
      KMP_DUPLICATE_LIB_OK=TRUE python train.py --data arxiv_smoke --n_epoch 1 --n_runs 1 --bs 200 --patience 2 --gpu 0
    fi
  ) >> "$STATUS" 2>&1 || log "CDGP smoke run skipped/failed (see log)."
else
  log "CDGP baseline not found"
fi

# TGB2 baseline
TGB2_DIR="$ROOT/baselines/src/TGB2-main/TGB2-main"
if [ -d "$TGB2_DIR" ]; then
  log "Found TGB2 baseline at $TGB2_DIR"
  mkdir -p "$TGB2_DIR/datasets/arxiv_smoke"
  cp "$PROJ/data/interim/baseline_adapters/arxiv_smoke/tgb2/arxiv_smoke_edgelist.csv" "$TGB2_DIR/datasets/arxiv_smoke/"
  cp "$PROJ/data/interim/baseline_adapters/arxiv_smoke/tgb2/arxiv_smoke_nodetype.csv" "$TGB2_DIR/datasets/arxiv_smoke/"
  log "TGB2 adapter files copied."
  log "Running TGB2 module custom smoke..."
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" "$TIMEOUT_SEC" "$PY" "$PROJ/scripts/baselines/run_tgb2_custom_smoke.py" \
      --data "$PROJ/data/processed/arxiv_smoke.pt" \
      --epochs 1 --batch-size 200 \
      --out "$PROJ/results/baseline_status/tgb2_custom_metrics.csv" >> "$STATUS" 2>&1 \
      || log "TGB2 custom smoke run skipped/failed (see log)."
  else
    "$PY" "$PROJ/scripts/baselines/run_tgb2_custom_smoke.py" \
      --data "$PROJ/data/processed/arxiv_smoke.pt" \
      --epochs 1 --batch-size 200 \
      --out "$PROJ/results/baseline_status/tgb2_custom_metrics.csv" >> "$STATUS" 2>&1 \
      || log "TGB2 custom smoke run skipped/failed (see log)."
  fi

  log "Preparing TGB2 official dataset mapping (thgl-github) ..."
  "$PY" "$PROJ/scripts/baselines/prepare_tgb2_official_dataset.py" \
    --input "$PROJ/data/processed/arxiv_smoke.pt" \
    --tgb2-root "$TGB2_DIR" \
    --dataset-name thgl-github \
    --max-edges 10000 >> "$STATUS" 2>&1 \
    || log "TGB2 official dataset prep skipped/failed (see log)."

  log "Running TGB2 official thgl-github/tgn.py smoke..."
  TGB2_OFFICIAL_LOG="$PROJ/results/baseline_status/tgb2_official_tgn.log"
  if [ -n "$TIMEOUT_BIN" ]; then
    (
      cd "$TGB2_DIR"
      PYTHONPATH="$TGB2_DIR:${PYTHONPATH:-}" \
      KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$TIMEOUT_BIN" "$TIMEOUT_SEC" python examples/linkproppred/thgl-github/tgn.py --num_epoch 1 --bs 200 --seed 1
    ) > "$TGB2_OFFICIAL_LOG" 2>&1 || log "TGB2 official smoke run skipped/failed (see log)."
  else
    (
      cd "$TGB2_DIR"
      PYTHONPATH="$TGB2_DIR:${PYTHONPATH:-}" \
      KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      python examples/linkproppred/thgl-github/tgn.py --num_epoch 1 --bs 200 --seed 1
    ) > "$TGB2_OFFICIAL_LOG" 2>&1 || log "TGB2 official smoke run skipped/failed (see log)."
  fi

  "$PY" - <<PY >> "$STATUS" 2>&1 || true
import re
from pathlib import Path
log_path = Path("$PROJ/results/baseline_status/tgb2_official_tgn.log")
if not log_path.exists():
    print("TGB2 official parser: log file not found.")
    raise SystemExit(0)
text = log_path.read_text(encoding="utf-8", errors="ignore")
val = re.findall(r"Validation mrr:\\s+([0-9.]+)", text)
test = re.findall(r"Test: mrr:\\s+([0-9.]+)", text)
if val and test:
    out = Path("$PROJ/results/baseline_status/tgb2_official_tgn_metrics.csv")
    out.write_text(f"split,mrr\\nval,{val[-1]}\\ntest,{test[-1]}\\n", encoding="utf-8")
    print(f"TGB2 official metrics parsed into {out}")
else:
    print("TGB2 official parser: metric patterns not found in log.")
PY
else
  log "TGB2 baseline not found"
fi

# DyGLib baseline
DYGLIB_DIR="$ROOT/baselines/src/DyGLib_TGB-master/DyGLib_TGB-master"
if [ -d "$DYGLIB_DIR" ]; then
  log "Found DyGLib baseline at $DYGLIB_DIR"
  log "Running DyGLib module custom smoke..."
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" "$TIMEOUT_SEC" env KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PY" "$PROJ/scripts/baselines/run_dyglib_custom_smoke.py" \
      --data "$PROJ/data/processed/arxiv_smoke.pt" \
      --epochs 1 --batch-size 200 --max-edges 10000 \
      --out "$PROJ/results/baseline_status/dyglib_custom_metrics.csv" >> "$STATUS" 2>&1 \
      || log "DyGLib custom smoke run skipped/failed (see log)."
  else
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PY" "$PROJ/scripts/baselines/run_dyglib_custom_smoke.py" \
      --data "$PROJ/data/processed/arxiv_smoke.pt" \
      --epochs 1 --batch-size 200 --max-edges 10000 \
      --out "$PROJ/results/baseline_status/dyglib_custom_metrics.csv" >> "$STATUS" 2>&1 \
      || log "DyGLib custom smoke run skipped/failed (see log)."
  fi
else
  log "DyGLib baseline not found"
fi

$PY - <<PY >> "$STATUS" 2>&1 || true
import re
from pathlib import Path

status = Path("$STATUS").read_text(encoding="utf-8", errors="ignore")
old = re.findall(r"Test statistics: Old nodes -- auc: ([0-9.eE+-]+), ap: ([0-9.eE+-]+)", status)
new = re.findall(r"Test statistics: New nodes -- auc: ([0-9.eE+-]+), ap: ([0-9.eE+-]+)", status)
if old and new:
    out = Path("$PROJ/results/baseline_status/tgn_external_metrics.csv")
    out.write_text(
        "split,auc,ap\n"
        f"old_nodes,{old[-1][0]},{old[-1][1]}\n"
        f"new_nodes,{new[-1][0]},{new[-1][1]}\n",
        encoding="utf-8",
    )
    print(f"TGN metrics parsed into {out}")
else:
    print("TGN metrics parser: no complete old/new nodes results found.")
PY

log "Baseline smoke staging completed"
