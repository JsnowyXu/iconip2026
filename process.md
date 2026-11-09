# DHC Experiment Enhancement Process

## 2026-05-20

### Step 0 - Initialization
- Status: completed
- Request scope: strengthen DHC dynamic heterogeneous temporal link prediction experiments and update the paper without fabricating results.
- Active environment from README: `/home/xuxj/.conda/envs/pytorch/bin/python`, project root `/home/xuxj/2026/dyn_relation_memory`, dataset root `/home/xuxj/2026/datasets`, baseline root `/home/xuxj/2026/baselines/src`.
- Required tracking: every completed stage in this request is logged here.

### Step 1 - Code and Protocol Audit
- Status: completed
- Assigned: code/protocol audit agent plus local verification.
- Scope: task views, negative sampling, MRR/AUC/AP, train/test flow, reproducible commands.
- Findings: task views use target-relation supervised edges with heterogeneous context edges; transductive target edges use chronological 70/15/15 splits; inductive views hold out source-side unseen target nodes where feasible and remove context edges touching those nodes from train/val.
- Findings: DHC training uses one sampled negative destination per supervised edge, mixing random, node-type-consistent, and popular negatives according to configured ratios. Validation/test use 20 negatives per positive and causal replay; test history is train+val.
- Findings: MRR ranks one positive against sampled negative destinations with ties counted as `neg >= pos`; AUC/AP are binary metrics over positive and negative scores.
- Risk: evaluation negatives are sampled inside each runner rather than stored as a fixed negative set, so exact values can vary across reruns unless each baseline fixes its own seed.

### Step 2 - Baseline Extension Audit
- Status: completed
- Scope: traditional ML baselines, graph-based baselines DyGFormer, EdgeBank, TGAT, and existing TGN/TGN-edge/STHN/RepeatMixer/TAMI status.
- Findings: TGN, TGN-edge, STHN, RepeatMixer, EdgeBank, and DyGFormer runners exist; TGAT is supported by the DyGLib model-zoo runner but was missing from the campaign model list.
- Findings: no traditional ML baseline runner was present.
- Finding: the existing EdgeBank runner evaluated all split edges as positives, including `edge_label=0` context edges, which was inconsistent with the DHC/TGN supervised-edge metric protocol.
- Action: added `scripts/baselines/run_tabular_temporal_baselines.py` with XGBoost, RandomForest, LogisticRegression, and TemporalCount forecasting baselines using the same task split and 20-negative MRR/AUC/AP evaluation.
- Action: updated `run_edge_only_campaign.py` to include TGAT in the DyGLib model zoo list and add a `tabular_ml` baseline family.
- Action: updated `build_edge_only_reports.py` so EdgeBank, TGAT, XGBoost, RandomForest, LogisticRegression, and TemporalCount appear in report tables.
- Action: fixed `scripts/baselines/run_edgebank_smoke.py` to filter validation/test positives by `edge_label>0.5` before scoring.
- Verification: `py_compile` passed for the new and modified baseline/report scripts.
- Verification: full `tabular_ml` baseline campaign completed for `patent_us`, `thgl_software`, `aminer_paper`, and `dblp` across inductive/transductive task views.
- Outputs: `results/edge_only_formal/baselines/metrics/tabular_ml__*.csv` contains 8 task files; refreshed reports include XGBoost, RandomForest, LogisticRegression, TemporalCount, and corrected EdgeBank.

### Step 3 - Experiment Execution
- Status: completed
- Scope: run GPU-based experiments under the current train/val/test split and negative-sampling protocol; produce inductive/transductive main tables.
- Main outputs: refreshed tables are in `results/edge_only_formal/reports/edge_inductive_mrr.csv`, `edge_transductive_mrr.csv`, `edge_inductive_ap.csv`, `edge_transductive_ap.csv`, `edge_inductive_auc.csv`, and `edge_transductive_auc.csv`.
- Note: traditional ML results were run with `max_train_pos=200000` for training control but full validation/test positives and 20 sampled negatives. Common-neighbor feature computation is optional and was disabled in the formal campaign to keep full evaluation tractable.
- Note: DyGFormer/TGAT support is wired through the DyGLib model-zoo runner, but current formal model-zoo CSVs remain failed/missing for most tasks and should not be reported as successful results until rerun.

### Step 4 - Visualization and Case Study
- Status: partially completed
- Scope: relation-memory evolution, sensitivity/gain plots, convergence curves, temporal-context visualizations, and error/case analysis.
- Action: reused existing cross-relation Patent-US figure `ICONIP-old/figures/cross_relation_mrr_base_vs_best.png` and case-study table `results/relation_temporal_context_case_study/relation_temporal_context_case_table.tex` in the paper.
- Action: rewrote the case-study text to distinguish THGL relation-memory AP gains from Patent-CN neighbor-window behavior.
- Remaining: new frontiers-style memory trajectory, epoch-loss-MRR comparison curves, attention/context heatmaps, and error-case visualizations were not regenerated in this pass.

### Step 5 - Paper Update and Reviewer Check
- Status: partially completed
- Scope: update Experiments section, add tables/figures/case study, then review for fairness, clarity, style, and page budget.
- Action: updated `ICONIP/texfiles/experiments.tex` to the requested structure: Experiment Setup, Performance Comparison, Ablation and Sensitivity, Cross-relation Patent Analysis, and Relation-memory and Temporal Context Case Study.
- Action: replaced the main MRR tables with refreshed values from `results/edge_only_formal/reports`, including EdgeBank and traditional ML baselines.
- Reviewer note: the updated text no longer claims uniform DHC superiority; it explicitly discusses settings where STHN, TGN-edge, or historical/tabular baselines are stronger.
- Verification limitation: LaTeX compilation could not be run because neither `latexmk` nor `pdflatex` is installed in the current shell environment.

### Step 6 - Tabular Classical ML Recheck
- Status: completed
- Scope: recheck and rerun only XGBoost, RandomForest, LogisticRegression, and TemporalCount for `patent_us`, `patent_in`, `patent_cn`, `thgl_software`, `aminer_paper`, and `dblp`.
- Assigned: classical ML baseline agent `Boole` plus local verification.
- Finding: the runner already filters evaluation positives with `edge_label>0.5`, but method comparison used different sampled negatives because the same RNG object advanced across models.
- Action: updated `scripts/baselines/run_tabular_temporal_baselines.py` so all four methods use identical validation/test negative samples within each task (`seed+1001` for validation and `seed+2001` for test).
- Action: optimized TemporalCount evaluation to skip unused ML feature construction and corrected output `device` labels so RandomForest, LogisticRegression, and TemporalCount report `cpu`.
- Action: added `scripts/experiments/build_tabular_ml_reports.py` to write tabular-only report files under `results/edge_only_formal/reports/` without mixing with neural or historical graph baselines.
- Execution: reran `tabular_ml` for 6 datasets (`patent_us`, `patent_in`, `patent_cn`, `thgl_software`, `aminer_paper`, `dblp`) and both inductive/transductive task views; all 12 campaign jobs returned `0`.
- Outputs: `tabular_ml_edge_inductive_mrr.csv`, `tabular_ml_edge_inductive_auc.csv`, `tabular_ml_edge_inductive_ap.csv`, `tabular_ml_edge_transductive_mrr.csv`, `tabular_ml_edge_transductive_auc.csv`, `tabular_ml_edge_transductive_ap.csv`, `tabular_ml_test_metrics_long.csv`, and `tabular_ml_summary.md` under `results/edge_only_formal/reports/`.
- Verification: tabular-only long table has 48 rows = 6 datasets x 2 task types x 4 methods, and every metric table contains only `XGBoost`, `RandomForest`, `LogisticRegression`, and `TemporalCount`.
- Note: XGBoost logs include a warning about CPU numpy inputs with a CUDA booster falling back to DMatrix prediction. The jobs completed successfully; the warning is a speed/memory warning rather than a metric failure.

### Step 7 - DyGFormer/TGAT/EdgeBank and Remaining Agents
- Status: in progress
- Scope: confirm no previous background experiments are still running, then restart remaining baseline, visualization, case-study, and writing agents.
- Finding: no `dyn_relation_memory` experiment process was running before this step; the visible PyTorch process belonged to `/home/xuxj/2026/dyn_hetero_community`.
- Assigned: baseline agent `Parfit`, visualization agent `Huygens`, case-study agent `Galileo`, and writing agent `Bernoulli`.
- EdgeBank finding: the very low `0.047619` MRR values are explainable under the current sampled-ranking protocol when no positive evaluation pair appears in the positive history; with 20 negatives and all scores tied at zero, MRR equals `1/(20+1)`.
- EdgeBank verification: `results/tmp_edgebank_dblp_inductive_check.csv` shows DBLP inductive has `pos_history_hit_rate=0.0` and `all_zero_tie_rate=1.0`, so this case is a cold-start/history-miss failure rather than an MRR arithmetic bug.
- DyGLib finding: `run_dyglib_model_zoo_smoke.py` previously used all task split edges as positive interactions, including `edge_label=0` context edges; this was inconsistent with DHC/TGN supervised-edge metrics.
- Action: updated `scripts/baselines/run_dyglib_model_zoo_smoke.py` so edge tasks use only supervised `edge_label>0.5` split edges by default. The old behavior is available only with `--include-context-edges`.
- Verification: a DBLP smoke run for `TGAT,DyGFormer` completed successfully with supervised-only edges and wrote `results/tmp_dyglib_supervised_smoke.csv`.
- Case-study update: aggregate case-study evidence and LaTeX were updated by the case-study agent without fabricated edge-level examples.
- Writing update: Experiments staging comments and integration placeholders were added by the writing agent without reporting missing DyGFormer/TGAT results as successful.

### Step 7 - Experiments Section Staging
- Status: completed for staging; final integration pending baseline/visual/case agents.
- Scope: prepare `ICONIP/texfiles/experiments.tex` for the final experiment pass without inventing missing results.
- Action: added a LaTeX integration checklist at the top of the Experiments section, listing confirmed inputs and pending artifacts.
- Action: tightened baseline wording so DyGFormer/TGAT are described only as supported/pending model-zoo baselines, not as completed results.
- Action: marked insertion points for DyGFormer/TGAT rows, refreshed six-dataset tabular-only summaries, updated visualization artifacts, and final case-study text.
- Page-budget note: current staging keeps MRR as the main body metric and treats AP/AUC tables as optional/supplementary unless a figure is removed.
