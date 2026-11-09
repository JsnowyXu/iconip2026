# Frontier Visual Analysis Inventory

Generated only from existing result files in this workspace.

## Available Data
- memory_trace.pt: 155 file(s)
  - `results/aminer_dhc_formal/aminer_paper_citation_edge_inductive__seed42/memory_trace.pt`
  - `results/aminer_dhc_formal/aminer_paper_citation_edge_transductive__seed42/memory_trace.pt`
  - `results/aminer_dhc_formal/aminer_paper_coauthor_edge_inductive__seed42/memory_trace.pt`
  - `results/aminer_dhc_formal/aminer_paper_coauthor_edge_transductive__seed42/memory_trace.pt`
  - `results/aminer_dhc_formal/aminer_paper_node_multilabel__seed42/memory_trace.pt`
  - `results/best_config_module_ablation/patent_cn_edge_org_codeIII_inductive__full__s42/memory_trace.pt`
  - `results/best_config_module_ablation/patent_cn_edge_org_codeIII_inductive__no_community__s42/memory_trace.pt`
  - `results/best_config_module_ablation/patent_cn_edge_org_codeIII_inductive__no_edge_type__s42/memory_trace.pt`
  - ... 147 more
- train_history.csv: 169 file(s)
  - `results/aminer_dhc_formal/aminer_paper_citation_edge_inductive__seed42/train_history.csv`
  - `results/aminer_dhc_formal/aminer_paper_citation_edge_transductive__seed42/train_history.csv`
  - `results/aminer_dhc_formal/aminer_paper_coauthor_edge_inductive__seed42/train_history.csv`
  - `results/aminer_dhc_formal/aminer_paper_coauthor_edge_transductive__seed42/train_history.csv`
  - `results/aminer_dhc_formal/aminer_paper_node_multilabel__seed42/train_history.csv`
  - `results/best_config_module_ablation/patent_cn_edge_org_codeIII_inductive__full__s42/train_history.csv`
  - `results/best_config_module_ablation/patent_cn_edge_org_codeIII_inductive__no_community__s42/train_history.csv`
  - `results/best_config_module_ablation/patent_cn_edge_org_codeIII_inductive__no_edge_type__s42/train_history.csv`
  - ... 161 more
- best_config_sensitivity/job_plan.csv: 1 file(s)
  - `results/best_config_sensitivity/job_plan.csv`
- best_config_sensitivity/best_run_catalog.csv: 1 file(s)
  - `results/best_config_sensitivity/best_run_catalog.csv`
- relation_temporal_context_case_study/*.csv: 3 file(s)
  - `results/relation_temporal_context_case_study/aggregate_case_study_evidence.csv`
  - `results/relation_temporal_context_case_study/relation_memory_ablation_case.csv`
  - `results/relation_temporal_context_case_study/temporal_context_neighbor_size_case.csv`
- community_memory_case_study memory_evolution.csv: 8 file(s)
  - `results/community_memory_case_study/patent_cn_edge_org_codeIII_inductive/memory_evolution/memory_evolution.csv`
  - `results/community_memory_case_study/patent_cn_edge_org_codeIII_transductive/memory_evolution/memory_evolution.csv`
  - `results/community_memory_case_study/patent_in_edge_org_codeIII_inductive/memory_evolution/memory_evolution.csv`
  - `results/community_memory_case_study/patent_in_edge_org_codeIII_transductive/memory_evolution/memory_evolution.csv`
  - `results/community_memory_case_study/patent_us_edge_org_codeIII_inductive/memory_evolution/memory_evolution.csv`
  - `results/community_memory_case_study/patent_us_edge_org_codeIII_transductive/memory_evolution/memory_evolution.csv`
  - `results/community_memory_case_study/thgl_software_edge_inductive/memory_evolution/memory_evolution.csv`
  - `results/community_memory_case_study/thgl_software_edge_transductive/memory_evolution/memory_evolution.csv`
- baseline_audit_20260520/edgebank/*.csv: 12 file(s)
  - `results/baseline_audit_20260520/edgebank/edgebank__aminer_paper_citation_edge_inductive.csv`
  - `results/baseline_audit_20260520/edgebank/edgebank__aminer_paper_citation_edge_transductive.csv`
  - `results/baseline_audit_20260520/edgebank/edgebank__dblp_edge_inductive.csv`
  - `results/baseline_audit_20260520/edgebank/edgebank__dblp_edge_transductive.csv`
  - `results/baseline_audit_20260520/edgebank/edgebank__patent_cn_edge_org_codeIII_inductive.csv`
  - `results/baseline_audit_20260520/edgebank/edgebank__patent_cn_edge_org_codeIII_transductive.csv`
  - `results/baseline_audit_20260520/edgebank/edgebank__patent_in_edge_org_codeIII_inductive.csv`
  - `results/baseline_audit_20260520/edgebank/edgebank__patent_in_edge_org_codeIII_transductive.csv`
  - ... 4 more
- baseline_audit_20260520/dyglib/*.csv: 1 file(s)
  - `results/baseline_audit_20260520/dyglib/tgat_dygformer__patent_us_edge_org_codeIII_inductive.csv`
- referenceanalysis: 7 file(s)
  - `referenceanalysis/TKDE-图.pdf`
  - `referenceanalysis/专利演化趋势.png`
  - `referenceanalysis/交通参数敏感及casestudy.png`
  - `referenceanalysis/交通流速流量.png`
  - `referenceanalysis/地理交通.png`
  - `referenceanalysis/演化图.png`
  - `referenceanalysis/预测结果by年数据演化.png`

## Generated Figures
- `results/frontier_visual_analysis/figures/sensitivity_gain_landscape.png`
- `results/frontier_visual_analysis/figures/sensitivity_swing_heatmap.png`
- `results/frontier_visual_analysis/figures/dhc_epoch_loss_mrr_multi_task.png`
- `results/frontier_visual_analysis/figures/relation_memory_evolution_multi_task.png`
- `results/frontier_visual_analysis/figures/community_churn_multi_task.png`
- `results/frontier_visual_analysis/figures/raw_relation_memory_fingerprints.png`
- `results/frontier_visual_analysis/figures/relation_memory_ablation_delta_ap.png`
- `results/frontier_visual_analysis/figures/temporal_context_neighbor_window.png`
- `results/frontier_visual_analysis/figures/edgebank_history_hit_zero_tie_rates.png`
- `results/frontier_visual_analysis/figures/edgebank_mrr_vs_history_hit.png`

## Data Limits
- Raw `.pt` memory traces were read with a torch-enabled interpreter for `raw_relation_memory_fingerprints.png`; CSV memory summaries were also used for multi-task evolution curves.
- No per-edge prediction/error-case table was found by filename/content search, so no false-positive/false-negative case figure was generated.
- Sensitivity figures use completed runs with readable `metrics.json` paths from `results/best_config_sensitivity/job_plan.csv`.
- DyGFormer/TGAT audit files currently expose summary metrics only, not epoch-level loss/MRR histories; they were not drawn as epoch curves.
