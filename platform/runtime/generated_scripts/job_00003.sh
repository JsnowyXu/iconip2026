#!/usr/bin/env bash
set -euo pipefail
cd /home/xuxj/2026/dyn_hetero_community
export PYTHONPATH="/home/xuxj/2026/dyn_hetero_community/src:$PYTHONPATH"
python scripts/datasets/build_paper_task_views.py --input /home/xuxj/2026/dyn_hetero_community/data/processed/arxiv_unified.pt --out-dir /home/xuxj/2026/dyn_hetero_community/data/processed/platform_tasks --name-prefix arxiv --target-rel 0
