#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/xuxj/2026}
PROJ=${2:-/home/xuxj/2026/dyn_hetero_community}
MAX_EDGES=${3:-120000}
export KMP_DUPLICATE_LIB_OK=TRUE
TEXT_ENCODER=${TEXT_ENCODER:-hybrid_hash}
TEXT_DIM=${TEXT_DIM:-64}
TEXT_MODEL_NAME=${TEXT_MODEL_NAME:-}
TEXT_BATCH_SIZE=${TEXT_BATCH_SIZE:-128}

mkdir -p "$PROJ/data/processed" "$PROJ/data/interim/baseline_adapters"

TEXT_ARGS=(--text-encoder "$TEXT_ENCODER" --text-dim "$TEXT_DIM" --text-batch-size "$TEXT_BATCH_SIZE")
if [[ -n "$TEXT_MODEL_NAME" ]]; then
  TEXT_ARGS+=(--text-model-name "$TEXT_MODEL_NAME")
fi

ANALYZE_INPUTS=()

python "$PROJ/scripts/datasets/prepare_dataset.py" \
  --dataset arxiv \
  --archive "$ROOT/datasets/arxiv.zip" \
  --max-edges "$MAX_EDGES" \
  "${TEXT_ARGS[@]}" \
  --out-pt "$PROJ/data/processed/arxiv_unified.pt" \
  --export-baselines \
  --baseline-out-dir "$PROJ/data/interim/baseline_adapters/arxiv"
ANALYZE_INPUTS+=("$PROJ/data/processed/arxiv_unified.pt")

python "$PROJ/scripts/datasets/prepare_dataset.py" \
  --dataset amazon_movies \
  --archive "$ROOT/datasets/Amazon_movies.zip" \
  --max-edges "$MAX_EDGES" \
  "${TEXT_ARGS[@]}" \
  --out-pt "$PROJ/data/processed/amazon_movies_unified.pt" \
  --export-baselines \
  --baseline-out-dir "$PROJ/data/interim/baseline_adapters/amazon_movies"
ANALYZE_INPUTS+=("$PROJ/data/processed/amazon_movies_unified.pt")

python "$PROJ/scripts/datasets/prepare_dataset.py" \
  --dataset kg4ai \
  --archive "$ROOT/datasets/kg4ai_global_jsondl_1.0.1.tar.gz" \
  --max-edges "$MAX_EDGES" \
  "${TEXT_ARGS[@]}" \
  --out-pt "$PROJ/data/processed/kg4ai_unified.pt" \
  --export-baselines \
  --baseline-out-dir "$PROJ/data/interim/baseline_adapters/kg4ai"
ANALYZE_INPUTS+=("$PROJ/data/processed/kg4ai_unified.pt")

python "$PROJ/scripts/datasets/prepare_dataset.py" \
  --dataset raw_cn \
  --archive "$ROOT/datasets/raw-data.zip" \
  --max-edges "$MAX_EDGES" \
  "${TEXT_ARGS[@]}" \
  --out-pt "$PROJ/data/processed/raw_cn_unified.pt" \
  --export-baselines \
  --baseline-out-dir "$PROJ/data/interim/baseline_adapters/raw_cn"
ANALYZE_INPUTS+=("$PROJ/data/processed/raw_cn_unified.pt")

python "$PROJ/scripts/datasets/prepare_dataset.py" \
  --dataset raw_us \
  --archive "$ROOT/datasets/raw-data.zip" \
  --max-edges "$MAX_EDGES" \
  "${TEXT_ARGS[@]}" \
  --out-pt "$PROJ/data/processed/raw_us_unified.pt" \
  --export-baselines \
  --baseline-out-dir "$PROJ/data/interim/baseline_adapters/raw_us"
ANALYZE_INPUTS+=("$PROJ/data/processed/raw_us_unified.pt")

python "$PROJ/scripts/datasets/prepare_dataset.py" \
  --dataset dblp \
  --archive "$ROOT/datasets/DBLP-Citation-network-V18.zip" \
  --max-edges "$MAX_EDGES" \
  "${TEXT_ARGS[@]}" \
  --out-pt "$PROJ/data/processed/dblp_unified.pt" \
  --export-baselines \
  --baseline-out-dir "$PROJ/data/interim/baseline_adapters/dblp"
ANALYZE_INPUTS+=("$PROJ/data/processed/dblp_unified.pt")

if [[ -f "$ROOT/datasets/Aminer/AMiner-Paper.rar" ]]; then
  python "$PROJ/scripts/datasets/prepare_dataset.py" \
    --dataset aminer_paper \
    --archive "$ROOT/datasets/Aminer/AMiner-Paper.rar" \
    --max-edges "$MAX_EDGES" \
    "${TEXT_ARGS[@]}" \
    --out-pt "$PROJ/data/processed/aminer_paper_unified.pt" \
    --export-baselines \
    --baseline-out-dir "$PROJ/data/interim/baseline_adapters/aminer_paper"
  ANALYZE_INPUTS+=("$PROJ/data/processed/aminer_paper_unified.pt")
fi

python "$PROJ/scripts/datasets/analyze_datasets.py" \
  --inputs "${ANALYZE_INPUTS[@]}" \
  --out-dir "$PROJ/data/reports" \
  --skip-plots

echo "All datasets prepared and analyzed."
