# ICONIP Figure Sources

Run `python scripts/experiments/render_iconip_mrr_artifacts.py --project-root /home/xuxj/2026/dyn_relation_memory` to regenerate the MRR-focused paper artifacts.

| Figure | Code | Data | Note |
|---|---|---|---|
| `best_module_ablation_mrr.png` | `scripts/experiments/render_iconip_mrr_artifacts.py` | `results/best_config_module_ablation/iconip_summary.csv` | Patent-US and THGL-Software MRR module ablation |
| `best_config_sensitivity_mrr.png` | `scripts/experiments/render_iconip_mrr_artifacts.py` | `results/best_config_sensitivity/job_plan.csv + completed metrics.json` | MRR-focused sensitivity boxplots |
| `patent_cn_sensitivity_mrr.png` | `scripts/experiments/plot_patent_cn_sensitivity_mrr.py` | `ICONIP/data/patent_cn_sensitivity_mrr.csv` | Patent-CN parameter sensitivity curves |
| `patent_cn_sensitivity_best_settings.csv` | `scripts/experiments/plot_patent_cn_sensitivity_mrr.py` | `ICONIP/data/patent_cn_sensitivity_mrr.csv` | Best Patent-CN setting per task type and parameter |
| `patent_us_sensitivity_mrr_no_community_iconip.png` | `scripts/experiments/plot_patent_cn_sensitivity_mrr.py` | `ICONIP/data/patent_us_sensitivity_mrr_iconip.csv` | Patent-US non-community parameter sensitivity curves |
| `patent_us_sensitivity_mrr_box_no_community_iconip.png` | `scripts/experiments/plot_patent_cn_sensitivity_mrr.py` | `ICONIP/data/patent_us_sensitivity_mrr_iconip.csv` | Patent-US non-community parameter-level MRR boxplots |
| `patent_us_sensitivity_mrr_value_box_no_community_iconip.png` | `scripts/experiments/plot_patent_cn_sensitivity_mrr.py` | `ICONIP/data/patent_us_sensitivity_mrr_iconip.csv` | Patent-US non-community value-level MRR boxplots matching `best_config_sensitivity_mrr.png` |
| `patent_us_sensitivity_mrr_delta_value_box_no_community_iconip.png` | `scripts/experiments/plot_patent_cn_sensitivity_mrr.py` | `ICONIP/data/patent_us_sensitivity_mrr_iconip.csv` | Patent-US value-level boxplots using MRR gain over each task anchor |
| `patent_us_sensitivity_best_settings_no_community_iconip.csv` | `scripts/experiments/plot_patent_cn_sensitivity_mrr.py` | `ICONIP/data/patent_us_sensitivity_mrr_iconip.csv` | Best Patent-US non-community setting per task type and parameter |
| `runtime_performance_patent_cn_thgl.png` | `scripts/experiments/plot_runtime_performance_comparison.py` | `ICONIP/data/runtime_performance_patent_cn_thgl.csv` | DHC vs TAMI MRR and time-to-best comparison |
| `cross_relation_mrr_base_vs_best.png` | `scripts/experiments/render_iconip_mrr_artifacts.py` | `ICONIP/data/cross_relation_boost_summary.csv` | MRR-only cross-relation base vs best boost |
| `cross_relation_mrr_delta_variants.png` | `scripts/experiments/render_iconip_mrr_artifacts.py` | `ICONIP/data/cross_relation_boost_summary.csv` | MRR-only cross-relation delta |
| `community_case_mrr_churn.png` | `scripts/experiments/render_iconip_mrr_artifacts.py` | `results/community_memory_case_study/*/README.md + monitor_assignment_churn.csv` | MRR vs community assignment churn |
| `bar_f1_by_dataset.png` | `scripts/experiments/plot_grid_search_figures.py` | `original grid-search summary/recommended CSV; not currently present in dyn_relation_memory/results` | Node-task micro-F1 figure |
| `bar_base_vs_recommended_auc_ap.png` | `scripts/experiments/plot_grid_search_figures.py` | `original grid-search summary/recommended CSV; not currently present in dyn_relation_memory/results` | Node-task AUC/AP figure |
| `scatter_pareto_ap_vs_f1.png` | `scripts/experiments/plot_grid_search_figures.py` | `original grid-search summary/recommended CSV; not currently present in dyn_relation_memory/results` | Node-task AP/F1 Pareto figure |
| `line_parameter_sensitivity.png` | `scripts/experiments/plot_grid_search_figures.py` | `original grid-search summary/recommended CSV; not currently present in dyn_relation_memory/results` | Node-task parameter sensitivity |
| `training_curves_recommended.png` | `scripts/experiments/plot_grid_search_figures.py` | `recommended run train_history.csv files` | Recommended node-task training curves |
| `memory_evolution_raw_us.png` | `scripts/visualization/plot_memory_evolution.py` | `selected run memory_trace.pt` | Memory evolution diagnostic |
| `raw_us_boost_community_trajectory.png` | `scripts/visualization/plot_community_vector_evolution.py` | `selected run memory_trace.pt` | Community centroid trajectory |
| `raw_us_boost_community_drift.png` | `scripts/visualization/plot_community_vector_evolution.py` | `selected run memory_trace.pt` | Community drift curve |

## Rebuild Commands

Regenerate the MRR artifact bundle and export the current Patent-CN sensitivity CSV:

```bash
python scripts/experiments/render_iconip_mrr_artifacts.py \
  --project-root /home/xuxj/2026/dyn_relation_memory
```

After replacing or supplementing `ICONIP/data/patent_cn_sensitivity_mrr.csv` with results from other servers, regenerate the Patent-CN sensitivity figure and best-setting CSV:

```bash
python scripts/experiments/plot_patent_cn_sensitivity_mrr.py \
  --csv /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_cn_sensitivity_mrr.csv \
  --out /home/xuxj/2026/dyn_relation_memory/ICONIP/figures/patent_cn_sensitivity_mrr.png \
  --best-out /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_cn_sensitivity_best_settings.csv
```

The sensitivity CSV columns are `dataset`, `task_name`, `task_type`, `parameter`, `value`, `anchor_value`, `anchor_test_mrr`, `test_mrr`, `test_auc`, `test_ap`, `delta_mrr_vs_anchor`, `run_name`, and `metrics_path`.

Regenerate the Patent-US non-community sensitivity figure without overwriting the Patent-CN figure:

```bash
python scripts/experiments/plot_patent_cn_sensitivity_mrr.py \
  --csv /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_us_sensitivity_mrr_iconip.csv \
  --out /home/xuxj/2026/dyn_relation_memory/ICONIP/figures/patent_us_sensitivity_mrr_no_community_iconip.png \
  --best-out /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_us_sensitivity_best_settings_no_community_iconip.csv \
  --exclude-community \
  --params typed_negative_ratio,popular_negative_ratio,recent_neighbor_size,neighbor_aggregation \
  --title "Patent-US MRR sensitivity without community parameters"
```

Regenerate the Patent-US non-community parameter-level MRR boxplot:

```bash
python scripts/experiments/plot_patent_cn_sensitivity_mrr.py \
  --csv /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_us_sensitivity_mrr_iconip.csv \
  --out /home/xuxj/2026/dyn_relation_memory/ICONIP/figures/patent_us_sensitivity_mrr_box_no_community_iconip.png \
  --best-out /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_us_sensitivity_best_settings_box_no_community_iconip.csv \
  --exclude-community \
  --params typed_negative_ratio,popular_negative_ratio,recent_neighbor_size,neighbor_aggregation \
  --plot-kind box \
  --title "Patent-US MRR sensitivity distribution without community parameters"
```

Regenerate the Patent-US non-community value-level MRR boxplot matching `best_config_sensitivity_mrr.png`:

```bash
python scripts/experiments/plot_patent_cn_sensitivity_mrr.py \
  --csv /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_us_sensitivity_mrr_iconip.csv \
  --out /home/xuxj/2026/dyn_relation_memory/ICONIP/figures/patent_us_sensitivity_mrr_value_box_no_community_iconip.png \
  --best-out /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_us_sensitivity_best_settings_value_box_no_community_iconip.csv \
  --exclude-community \
  --params typed_negative_ratio,popular_negative_ratio,recent_neighbor_size,neighbor_aggregation \
  --plot-kind value-box \
  --title "Patent-US MRR sensitivity distributions by parameter value" \
  --no-title
```

For Patent-US, raw `test_mrr` mixes inductive and transductive anchors (about 0.90 vs 0.947), so the box length is dominated by task-type baseline rather than parameter sensitivity. To preserve the sensitivity pattern across task types, use MRR gain over each task anchor:

```bash
python scripts/experiments/plot_patent_cn_sensitivity_mrr.py \
  --csv /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_us_sensitivity_mrr_iconip.csv \
  --out /home/xuxj/2026/dyn_relation_memory/ICONIP/figures/patent_us_sensitivity_mrr_delta_value_box_no_community_iconip.png \
  --best-out /home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_us_sensitivity_best_settings_delta_value_box_no_community_iconip.csv \
  --exclude-community \
  --params typed_negative_ratio,popular_negative_ratio,recent_neighbor_size,neighbor_aggregation \
  --plot-kind value-box \
  --box-metric delta_mrr \
  --title "Patent-US MRR gain distributions by parameter value" \
  --no-title
```

Regenerate the runtime-performance comparison CSV and figure:

```bash
python scripts/experiments/plot_runtime_performance_comparison.py \
  --project-root /home/xuxj/2026/dyn_relation_memory \
  --csv /home/xuxj/2026/dyn_relation_memory/ICONIP/data/runtime_performance_patent_cn_thgl.csv \
  --out /home/xuxj/2026/dyn_relation_memory/ICONIP/figures/runtime_performance_patent_cn_thgl.png
```

After updating `ICONIP/data/runtime_performance_patent_cn_thgl.csv` with measurements from another server, redraw the figure without rebuilding the CSV:

```bash
python scripts/experiments/plot_runtime_performance_comparison.py \
  --csv /home/xuxj/2026/dyn_relation_memory/ICONIP/data/runtime_performance_patent_cn_thgl.csv \
  --out /home/xuxj/2026/dyn_relation_memory/ICONIP/figures/runtime_performance_patent_cn_thgl.png \
  --from-csv
```
