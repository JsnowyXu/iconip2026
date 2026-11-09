#!/usr/bin/env bash
set -euo pipefail
cd /home/xuxj/2026/dyn_hetero_community
export PYTHONPATH="/home/xuxj/2026/dyn_hetero_community/src:$PYTHONPATH"
python -m dhc.train.run_train --data /home/xuxj/2026/dyn_hetero_community/data/processed/platform_tasks/arxiv_edge_transductive.pt --out-dir /home/xuxj/2026/dyn_hetero_community/results/platform_runs/platform_live_train_arxiv_verify_20260323 --seed 42 --epochs 1 --batch-size 256 --lr 0.001 --gpu-id 0 --early-stop-patience 1 --recent-neighbor-size 8 --lambda-cl 0.05 --lambda-ml 0.7 --community-assign-temperature 1.0 --community-assign-topk 0 --typed-negative-ratio 0.6 --popular-negative-ratio 0.2 --ml-pos-weight
