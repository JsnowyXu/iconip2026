#!/usr/bin/env bash
set -u

cd /home/xuxj/2026/dyn_relation_memory || exit 1

TASKS=(
  data/processed/tasks/patent_cn_edge_org_codeIII_transductive.pt
  data/processed/tasks/patent_cn_edge_org_codeIII_inductive.pt
  data/processed/tasks/patent_in_edge_org_codeIII_transductive.pt
  data/processed/tasks/patent_in_edge_org_codeIII_inductive.pt
  data/processed/tasks/patent_us_edge_org_codeIII_transductive.pt
  data/processed/tasks/patent_us_edge_org_codeIII_inductive.pt
  data/processed/paper_tasks/aminer_paper_citation_edge_transductive.pt
  data/processed/paper_tasks/aminer_paper_citation_edge_inductive.pt
  data/processed/paper_tasks/amazon_movies_edge_transductive.pt
  data/processed/paper_tasks/amazon_movies_edge_inductive.pt
  data/processed/tgb2_tasks/thgl_software_edge_transductive.pt
  data/processed/tgb2_tasks/thgl_software_edge_inductive.pt
)

mkdir -p results/edge_only_formal/baselines/metrics
mkdir -p results/edge_only_formal/baselines/logs

for data in "${TASKS[@]}"; do
  task=$(basename "$data" .pt)
  out="results/edge_only_formal/baselines/metrics/dyglib_model_zoo__${task}.csv"
  log="results/edge_only_formal/baselines/logs/dyglib_model_zoo__${task}.log"
  if [[ -s "$out" ]] && grep -q "TGAT,ok" "$out" && grep -q "GraphMixer,ok" "$out" && grep -q "DyGFormer,ok" "$out"; then
    echo "[$(date --iso-8601=seconds)] skip existing $task"
    continue
  fi

  echo "[$(date --iso-8601=seconds)] run $task"
  CUDA_VISIBLE_DEVICES=0 /home/xuxj/.conda/envs/pytorch/bin/python scripts/baselines/run_dyglib_model_zoo_smoke.py \
    --data "$data" \
    --models TGAT,GraphMixer,DyGFormer \
    --epochs 20 \
    --batch-size 256 \
    --max-edges 50000 \
    --gpu-id 0 \
    --eval-num-negs 20 \
    --out "$out" > "$log" 2>&1
  rc=$?
  echo "[$(date --iso-8601=seconds)] done $task rc=$rc"
  if [[ $rc -ne 0 ]]; then
    tail -80 "$log" || true
  fi
done
