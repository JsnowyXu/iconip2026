# Best-Config Module Ablation

This directory contains the controlled module ablation used by the ICONIP paper. Each task starts from the current best available full-module configuration and reruns four variants: `full`, `no_relation_memory`, `no_edge_type`, and `no_community`. The goal is not to tune a new champion but to ask which architectural signals are useful after the main configuration has already been selected.

## Scope

- Completed runs: 32 variants across 8 task views.
- Task families: Chinese, Indian, and U.S. patent `org_codeIII` link prediction, plus the TGB2.0 `thgl_software` relation-specific task.
- Metrics: `test_mrr`, `test_auc`, and `test_ap` are read from each run's `metrics.json`; `summary.csv` is the authoritative compact index.
- Paper artifacts: `ICONIP/tables/best_module_ablation_table.tex` and `ICONIP/figures/best_module_ablation_mrr.png`.

## Main Findings

The ablation is deliberately harder than a from-scratch module test because most anchors are already close to saturation. On the patent edge tasks, MRR and AUC are often above 0.99, so AP and small deltas are more informative than raw rank accuracy. Removing a module can occasionally improve AP by a few thousandths, which suggests that the best configuration has task-specific regularization effects rather than a universally dominant module stack.

The clearest positive role for the full model appears on `thgl_software_edge_inductive`: removing relation memory lowers AP from 0.5661 to 0.5466, removing edge-type features lowers AP to 0.5509, and removing community memory lowers AP to 0.5280. This is consistent with the TGB2.0 observation that temporal heterogeneous graphs need type-aware modeling and scalable relation handling. The transductive software split is less clean: `no_edge_type` reaches the best AP, which indicates that edge-type features can overfit when short-window software interactions are highly repetitive.

Across all eight tasks, the mean AP drop of removing relation memory is +0.0005, removing edge-type features is -0.0052, and removing community memory is +0.0084. Community memory has the largest average full-minus-ablated AP drop, but the sign is mixed across tasks. The practical interpretation is that community memory is useful as a mesoscopic stabilizer on noisy or larger streams, while saturated patent splits may prefer a lighter variant.

## Follow-up Strategy

The current evidence does not support rerunning the unchanged full model as the only core experiment. A better next step is task-adaptive module selection:

- For saturated patent edge views, use validation AP/MRR from this ablation and the sensitivity sweep to select either `full`, `no_relation_memory`, `no_edge_type`, or `no_community`.
- For TGB2.0 software inductive and other large inductive heterogeneous streams, keep relation/community memory unless validation clearly favors a lighter model, because removing them causes larger AP drops.
- After selecting one configuration per task, rerun the selected configurations over three seeds under the same chronological splits.
- Report the full model as the architecture and the selected lightweight variants as task-adaptive simplifications, rather than claiming every module is universally positive.

This strategy turns the negative or mixed ablation results into useful evidence: the modules are not wrong, but their benefit depends on recurrence, relation noise, split difficulty, and whether the task is already close to saturation.

## How to Read the Files

- `summary.csv` gives one row per task/variant and is the easiest file for quantitative comparison.
- `job_plan.csv` records the exact command line used for each run.
- Each run directory contains `metrics.json`, `train_history.csv`, `model_stats.json`, and saved memory artifacts when available.
- `module_locations.md` maps each ablated concept to the implementation files, which helps connect the experimental switch to the code path.

## Figure Interpretation

`best_module_ablation_mrr.png` groups each task by variant under the primary MRR metric. The figure uses coordinated colors plus hatch patterns so that the four settings remain distinguishable in grayscale or low-contrast prints. Values near 1.0 indicate saturated patent ranking tasks. The software tasks are more diagnostic because the MRR range is wider and module effects are easier to see. A drop after removing relation memory or community memory supports the paper's claim that DHC's relation and community states are complementary; an improvement after removal is treated as evidence of task-dependent simplification rather than as a contradiction.
