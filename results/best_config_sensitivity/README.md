# Best-Config Sensitivity

This directory contains a one-parameter-at-a-time sensitivity sweep around the current best configuration for each task. It varies negative sampling, local temporal context, and community-memory hyperparameters while keeping the rest of the training command fixed.

## Scope

- Planned runs: 192.
- Completed runs currently available on disk: 55.
- Completed task views: patent_cn_edge_org_codeIII_inductive, patent_cn_edge_org_codeIII_transductive, patent_in_edge_org_codeIII_inductive.
- Varied parameters: `typed_negative_ratio`, `popular_negative_ratio`, `recent_neighbor_size`, `neighbor_aggregation`, `community_assign_topk`, and `num_communities`.
- Paper artifacts: `ICONIP/tables/best_config_sensitivity_table.tex` and `ICONIP/figures/best_config_sensitivity_mrr.png`.

## Main Findings

The completed portion is already informative even though the sweep is not finished. The patent CN inductive and transductive tasks are stable over a broad range of settings: the best observed MRR remains around 0.996 and AP remains around 0.986--0.989 for many variants. This plateau means that DHC is not brittle to a single exact hyperparameter choice once the task-specific anchor has been selected.

Typed negative sampling is useful but not monotonic. Among completed runs, the best `typed_negative_ratio` setting by MRR is `0.2` on `patent_cn_edge_org_codeIII_inductive`. Popular-negative sampling is similarly task dependent; it changes the difficulty of the candidate set, but high ratios do not automatically improve AP.

Local temporal context has a visible effect. The best completed `recent_neighbor_size` by MRR is `16`, and the completed aggregation comparison slightly favors the task-specific choice rather than a universal `attn` or `mean` rule. Community hyperparameters also show a broad plateau: `num_communities=64` is currently the best completed setting by MRR, while nearby values usually remain close.

## How to Read the Files

- `job_plan.csv` is the complete plan, including runs that have not yet produced metrics.
- A run is considered complete when its `metrics.json` path exists.
- The figure `best_config_sensitivity_mrr.png` summarizes completed MRR distributions by parameter value. Wider boxes indicate stronger task interaction or incomplete coverage.
- The paper table reports the best completed setting by MRR within each varied parameter, not a final global hyperparameter recommendation.

## Caveats

The current sensitivity sweep is partial, so it should be used as evidence of robustness and qualitative trends rather than as a final hyperparameter search. The paper text should describe it as a completed-partial sweep unless the remaining jobs are run later.
