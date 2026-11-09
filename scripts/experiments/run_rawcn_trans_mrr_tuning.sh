#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="${1:-/home/xuxj/2026/dyn_hetero_community}"
cd "$PROJECT_ROOT"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate mmlab
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

COMMON_ARGS=(
  --data "$PROJECT_ROOT/data/processed-auto/tasks/raw_cn_edge_assignee_ipc_transductive.pt"
  --seed 42
  --epochs 10
  --batch-size 512
  --lr 0.001
  --early-stop-patience 5
  --ml-threshold-search
  --monitor-metric val_mrr
  --gpu-id 0
  --save-memory-artifacts
)

python -m dhc.train.run_train \
  "${COMMON_ARGS[@]}" \
  --out-dir "$PROJECT_ROOT/results-auto/head2head_rawcn_current_manual/raw_cn_edge_assignee_ipc_transductive__main_fullbench_mrr__s42" \
  --lambda-cl 0.05 \
  --recent-neighbor-size 12

python -m dhc.train.run_train \
  "${COMMON_ARGS[@]}" \
  --out-dir "$PROJECT_ROOT/results-auto/head2head_rawcn_current_manual/raw_cn_edge_assignee_ipc_transductive__patent_strong_mrr__s42" \
  --lambda-cl 0.10 \
  --recent-neighbor-size 10

python -m dhc.train.run_train \
  "${COMMON_ARGS[@]}" \
  --out-dir "$PROJECT_ROOT/results-auto/head2head_rawcn_current_manual/raw_cn_edge_assignee_ipc_transductive__patent_stronger_cl_mrr__s42" \
  --lambda-cl 0.15 \
  --recent-neighbor-size 12
