#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/xuxj/2026}
PROJ=${2:-/home/xuxj/2026/dyn_hetero_community}
PY=${3:-python}

cd "$PROJ"

export PYTHONPATH="$PROJ/src:$ROOT/baselines/src/TGB2-main/TGB2-main:${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

bash scripts/experiments/run_patent_suite.sh "$ROOT" "$PROJ" "$PY"

$PY scripts/experiments/aggregate_full_benchmark.py \
  --manifest results/full_benchmark_repeats/run_manifest.csv \
  --out-dir results/full_benchmark_repeats

if [ -f results/full_benchmark_aminer_formal/run_manifest.csv ]; then
  $PY scripts/experiments/aggregate_full_benchmark.py \
    --manifest results/full_benchmark_aminer_formal/run_manifest.csv \
    --out-dir results/full_benchmark_aminer_formal

  $PY scripts/experiments/render_aminer_extension_table_to_latex.py \
    --aggregate-csv results/full_benchmark_aminer_formal/aggregate_mean_std.csv \
    --out-tex paper_draft/tables/aminer_extension_results.tex

  $PY scripts/experiments/render_aminer_extension_table_to_latex.py \
    --aggregate-csv results/full_benchmark_aminer_formal/aggregate_mean_std.csv \
    --out-tex paper_draft/ICONIP/tables/aminer_extension_results.tex
fi

if [ -f results/full_benchmark_aminer_formal/aggregate_mean_std.csv ] && [ -f results/baseline_formal_aminer/summary.csv ]; then
  $PY scripts/experiments/render_aminer_baseline_table_to_latex.py \
    --dhc-aggregate-csv results/full_benchmark_aminer_formal/aggregate_mean_std.csv \
    --baseline-summary-csv results/baseline_formal_aminer/summary.csv \
    --out-tex paper_draft/tables/aminer_baseline_results.tex

  $PY scripts/experiments/render_aminer_baseline_table_to_latex.py \
    --dhc-aggregate-csv results/full_benchmark_aminer_formal/aggregate_mean_std.csv \
    --baseline-summary-csv results/baseline_formal_aminer/summary.csv \
    --out-tex paper_draft/ICONIP/tables/aminer_baseline_results.tex
fi

$PY scripts/experiments/render_patent_tables_to_latex.py \
  --summary-csv results/patent_suite/summary.csv \
  --out-tex paper_draft/tables/patent_results.tex

$PY scripts/experiments/render_patent_tables_to_latex.py \
  --summary-csv results/patent_suite/summary.csv \
  --out-tex paper_draft/ICONIP/tables/patent_results.tex

$PY scripts/experiments/render_cuda_tables_to_latex.py \
  --summary-csv results/cuda_paper/summary.csv \
  --out-tex paper_draft/tables/cuda_results.tex

$PY scripts/experiments/render_cuda_tables_to_latex.py \
  --summary-csv results/cuda_paper/summary.csv \
  --out-tex paper_draft/ICONIP/tables/cuda_results.tex

$PY scripts/experiments/render_cuda_academic_tables_to_latex.py \
  --summary-csv results/cuda_paper_academic/summary.csv \
  --out-tex paper_draft/tables/cuda_academic_results.tex

$PY scripts/experiments/render_cuda_academic_tables_to_latex.py \
  --summary-csv results/cuda_paper_academic/summary.csv \
  --out-tex paper_draft/ICONIP/tables/cuda_academic_results.tex

if [ -d results/tgb2_thgl_github_full_rel ] && [ -d results/tgb2_thgl_software_full_rel ]; then
  $PY scripts/experiments/render_tgb2_thg_tables_to_latex.py \
    --dataset-root "thgl_github=$PROJ/results/tgb2_thgl_github_full_rel" \
    --dataset-root "thgl_software=$PROJ/results/tgb2_thgl_software_full_rel" \
    --main-out paper_draft/tables/tgb2_thg_main.tex \
    --ablation-out paper_draft/tables/tgb2_thg_ablation.tex

  $PY scripts/experiments/render_tgb2_thg_tables_to_latex.py \
    --dataset-root "thgl_github=$PROJ/results/tgb2_thgl_github_full_rel" \
    --dataset-root "thgl_software=$PROJ/results/tgb2_thgl_software_full_rel" \
    --main-out paper_draft/ICONIP/tables/tgb2_thg_main.tex \
    --ablation-out paper_draft/ICONIP/tables/tgb2_thg_ablation.tex
fi

$PY scripts/experiments/build_baseline_task_tables.py   --full-aggregate-csv results/full_benchmark_repeats/aggregate_mean_std.csv   --out-dir results/dataset_statistics

$PY scripts/experiments/render_baseline_task_tables_to_latex.py   --input-dir results/dataset_statistics   --out-dir paper_draft/tables

$PY scripts/experiments/render_baseline_task_tables_to_latex.py   --input-dir results/dataset_statistics   --out-dir paper_draft/ICONIP/tables

if [ -f scripts/experiments/update_tracking_docs.py ]; then
  $PY scripts/experiments/update_tracking_docs.py --project-root "$PROJ" || true
fi

echo "[refresh_paper_artifacts] done"
