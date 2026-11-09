#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/xuxj/2026}
PROJ=${2:-/home/xuxj/2026/dyn_hetero_community}
PY=${3:-python}
TGB2_DIR="$ROOT/baselines/src/TGB2-main/TGB2-main"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
export PYTHONPATH="$PROJ/src:$TGB2_DIR:${PYTHONPATH:-}"
TIMEOUT_BIN=$(command -v timeout || true)
TIMEOUT_SEC=${BASELINE_TIMEOUT_SEC:-600}
LOG_DIR="$PROJ/results/patent_suite/logs"

mkdir -p "$PROJ/data/processed" "$PROJ/results/patent_suite" "$PROJ/results/patent_suite/baselines" "$LOG_DIR"

RAW_ARCHIVE="$ROOT/datasets/raw-data.zip"

run_prepare() {
  local ds=$1
  local out_pt="$PROJ/data/processed/${ds}_unified.pt"
  local adapter_dir="$PROJ/data/interim/baseline_adapters/${ds}"
  if [ -f "$out_pt" ] && [ -d "$adapter_dir" ]; then
    echo "[run_patent_suite] skip prepare ${ds}: existing $out_pt and $adapter_dir"
    return 0
  fi
  $PY "$PROJ/scripts/datasets/prepare_dataset.py" \
    --dataset "$ds" \
    --archive "$RAW_ARCHIVE" \
    --max-edges 120000 \
    --out-pt "$out_pt" \
    --export-baselines \
    --baseline-out-dir "$adapter_dir"
}

build_task_views_if_needed() {
  local input_pt=$1
  local name_prefix=$2
  local expected_out=$3
  if [ -f "$expected_out" ]; then
    echo "[run_patent_suite] skip task views ${name_prefix}: existing $expected_out"
    return 0
  fi
  $PY "$PROJ/scripts/datasets/build_task_views.py" \
    --input "$input_pt" \
    --out-dir "$PROJ/data/processed/tasks" \
    --name-prefix "$name_prefix"
}

run_logged_step() {
  local target_path=$1
  local log_name=$2
  shift 2
  if [ -f "$target_path" ]; then
    echo "[run_patent_suite] skip $(basename "$target_path"): existing $target_path"
    return 0
  fi
  "$@" > "$LOG_DIR/$log_name" 2>&1
}

train_dhc_task() {
  local data_pt=$1
  local run_name=$2
  local out_dir="$PROJ/results/patent_suite/${run_name}"
  local metrics_json="$out_dir/metrics.json"
  if [ -f "$metrics_json" ]; then
    echo "[run_patent_suite] skip ${run_name}: existing $metrics_json"
  else
    $PY -m dhc.train.run_train \
      --data "$data_pt" \
      --out-dir "$out_dir" \
      --epochs 2 \
      --batch-size 256 \
      --lr 0.001 \
      --save-memory-artifacts \
      --ml-threshold-search
  fi

  local trace_pt="$out_dir/memory_trace.pt"
  if [ -f "$trace_pt" ]; then
    local plot_png="$out_dir/memory_viz/memory_evolution.png"
    if [ -f "$plot_png" ]; then
      echo "[run_patent_suite] skip memory plot for ${run_name}: existing $plot_png"
    else
      $PY "$PROJ/scripts/visualization/plot_memory_evolution.py" \
        --trace "$trace_pt" \
        --data "$data_pt" \
        --out-dir "$out_dir/memory_viz"
    fi
  else
    echo "[run_patent_suite] skip memory plot for ${run_name}: missing $trace_pt"
  fi
}

run_prepare raw_cn
run_prepare raw_us

build_task_views_if_needed \
  "$PROJ/data/processed/raw_cn_unified.pt" \
  raw_cn \
  "$PROJ/data/processed/tasks/raw_cn_edge_patent_ipc.pt"

build_task_views_if_needed \
  "$PROJ/data/processed/raw_us_unified.pt" \
  raw_us \
  "$PROJ/data/processed/tasks/raw_us_edge_patent_ipc.pt"

train_dhc_task "$PROJ/data/processed/tasks/raw_cn_edge_patent_ipc.pt" "dhc_raw_cn_edge_patent_ipc"
train_dhc_task "$PROJ/data/processed/tasks/raw_cn_edge_assignee_ipc_transductive.pt" "dhc_raw_cn_edge_assignee_ipc_trans"
train_dhc_task "$PROJ/data/processed/tasks/raw_cn_edge_assignee_ipc_inductive.pt" "dhc_raw_cn_edge_assignee_ipc_ind"
train_dhc_task "$PROJ/data/processed/tasks/raw_cn_node_patent_multilabel.pt" "dhc_raw_cn_node_patent_multilabel"

train_dhc_task "$PROJ/data/processed/tasks/raw_us_edge_patent_ipc.pt" "dhc_raw_us_edge_patent_ipc"
train_dhc_task "$PROJ/data/processed/tasks/raw_us_edge_assignee_ipc_transductive.pt" "dhc_raw_us_edge_assignee_ipc_trans"
train_dhc_task "$PROJ/data/processed/tasks/raw_us_edge_assignee_ipc_inductive.pt" "dhc_raw_us_edge_assignee_ipc_ind"
train_dhc_task "$PROJ/data/processed/tasks/raw_us_node_patent_multilabel.pt" "dhc_raw_us_node_patent_multilabel"

# Baseline custom runs on the main relation task.
run_logged_step "$PROJ/results/patent_suite/baselines/tgb2_raw_cn_assignee_ipc.csv" "tgb2_raw_cn_assignee_ipc.log" \
  "$PY" "$PROJ/scripts/baselines/run_tgb2_custom_smoke.py" \
  --data "$PROJ/data/processed/tasks/raw_cn_edge_assignee_ipc_transductive.pt" \
  --epochs 1 \
  --batch-size 200 \
  --out "$PROJ/results/patent_suite/baselines/tgb2_raw_cn_assignee_ipc.csv"

run_logged_step "$PROJ/results/patent_suite/baselines/tgb2_raw_cn_patent_ipc.csv" "tgb2_raw_cn_patent_ipc.log" \
  "$PY" "$PROJ/scripts/baselines/run_tgb2_custom_smoke.py" \
  --data "$PROJ/data/processed/tasks/raw_cn_edge_patent_ipc.pt" \
  --epochs 1 \
  --batch-size 200 \
  --out "$PROJ/results/patent_suite/baselines/tgb2_raw_cn_patent_ipc.csv"

run_logged_step "$PROJ/results/patent_suite/baselines/dyglib_tgn_raw_cn_assignee_ipc.csv" "dyglib_tgn_raw_cn_assignee_ipc.log" \
  "$PY" "$PROJ/scripts/baselines/run_dyglib_custom_smoke.py" \
  --data "$PROJ/data/processed/tasks/raw_cn_edge_assignee_ipc_transductive.pt" \
  --epochs 1 \
  --batch-size 200 \
  --max-edges 10000 \
  --out "$PROJ/results/patent_suite/baselines/dyglib_tgn_raw_cn_assignee_ipc.csv"

run_logged_step "$PROJ/results/patent_suite/baselines/dyglib_tgn_raw_cn_patent_ipc.csv" "dyglib_tgn_raw_cn_patent_ipc.log" \
  "$PY" "$PROJ/scripts/baselines/run_dyglib_custom_smoke.py" \
  --data "$PROJ/data/processed/tasks/raw_cn_edge_patent_ipc.pt" \
  --epochs 1 \
  --batch-size 200 \
  --max-edges 10000 \
  --out "$PROJ/results/patent_suite/baselines/dyglib_tgn_raw_cn_patent_ipc.csv"

# External TGN baseline on task adapters (homogeneous protocol).
echo "[run_patent_suite] skip external TGN baseline (not required)."

run_logged_step "$PROJ/results/patent_suite/baselines/dyglib_model_zoo_raw_cn_assignee_ipc.csv" "dyglib_model_zoo_raw_cn_assignee_ipc.log" \
  "$PY" "$PROJ/scripts/baselines/run_dyglib_model_zoo_smoke.py" \
  --data "$PROJ/data/processed/tasks/raw_cn_edge_assignee_ipc_transductive.pt" \
  --models TGN,TGAT,CAWN,TCL,GraphMixer,DyGFormer \
  --epochs 1 \
  --batch-size 128 \
  --max-edges 4000 \
  --out "$PROJ/results/patent_suite/baselines/dyglib_model_zoo_raw_cn_assignee_ipc.csv"

run_logged_step "$PROJ/results/patent_suite/baselines/dyglib_model_zoo_raw_cn_node_task.csv" "dyglib_model_zoo_raw_cn_node_task.log" \
  "$PY" "$PROJ/scripts/baselines/run_dyglib_model_zoo_smoke.py" \
  --data "$PROJ/data/processed/tasks/raw_cn_node_patent_multilabel.pt" \
  --models TGN,TGAT,CAWN,TCL,GraphMixer,DyGFormer \
  --epochs 1 \
  --batch-size 128 \
  --max-edges 5000 \
  --out "$PROJ/results/patent_suite/baselines/dyglib_model_zoo_raw_cn_node_task.csv"

$PY - <<'PY'
import json
from pathlib import Path
import pandas as pd

proj = Path("/home/xuxj/2026/dyn_hetero_community")
rows = []

for mpath in sorted((proj / "results/patent_suite").glob("dhc_*/metrics.json")):
    m = json.loads(mpath.read_text(encoding="utf-8"))
    rows.append({
        "model": mpath.parent.name,
        "task": mpath.parent.name.replace("dhc_", ""),
        "test_mrr": m.get("test", {}).get("mrr"),
        "test_auc": m.get("test", {}).get("auc"),
        "test_ap": m.get("test", {}).get("ap"),
        "node_ml_micro_f1": m.get("node_eval", {}).get("node_ml_micro_f1"),
        "node_ml_map": m.get("node_eval", {}).get("node_ml_map"),
        "avg_epoch_time_sec": m.get("avg_epoch_time_sec"),
    })

for c in [
    proj / "results/patent_suite/baselines/tgb2_raw_cn_assignee_ipc.csv",
    proj / "results/patent_suite/baselines/tgb2_raw_cn_patent_ipc.csv",
    proj / "results/patent_suite/baselines/dyglib_tgn_raw_cn_assignee_ipc.csv",
    proj / "results/patent_suite/baselines/dyglib_tgn_raw_cn_patent_ipc.csv",
    proj / "results/patent_suite/baselines/tgn_external_patent_metrics.csv",
    proj / "results/patent_suite/baselines/dyglib_model_zoo_raw_cn_assignee_ipc.csv",
    proj / "results/patent_suite/baselines/dyglib_model_zoo_raw_cn_node_task.csv",
]:
    if c.exists():
        df = pd.read_csv(c)
        df["model"] = df.get("model", "baseline")
        df["task"] = c.stem
        rows.extend(df.to_dict(orient="records"))

out = pd.DataFrame(rows)
out_csv = proj / "results/patent_suite/summary.csv"
out.to_csv(out_csv, index=False)
md_lines = [
    "# Patent Suite Summary",
    "",
    f"- rows: {len(out)}",
    f"- dhc runs: {int(out['model'].astype(str).str.startswith('dhc_').sum())}",
    f"- baseline rows: {int((~out['model'].astype(str).str.startswith('dhc_')).sum())}",
    "",
]
try:
    md_lines.append(out.to_markdown(index=False))
except Exception:
    md_lines.append(out.to_string(index=False))
md_lines.append("")
(proj / "results/patent_suite/summary.md").write_text("\n".join(md_lines), encoding="utf-8")
print(out.head(20))
PY

echo "Patent suite finished. Summary: $PROJ/results/patent_suite/summary.csv"
