# Community Memory Case Study

- task: `patent_us_edge_org_codeIII_transductive`
- source_group: `targeted_tuning_followup`
- run_dir: `/home/xuxj/2026/dyn_relation_memory/results/targeted_edge_tuning_20260423_followup/patent_us_edge_org_codeIII_transductive__close_topk4_lr7e4_ap__s42`
- metrics_path: `/home/xuxj/2026/dyn_relation_memory/results/targeted_edge_tuning_20260423_followup/patent_us_edge_org_codeIII_transductive__close_topk4_lr7e4_ap__s42/metrics.json`
- test_mrr: `0.996349259489534`
- test_auc: `0.9996212297531918`
- test_ap: `0.9898178690849544`
- membership_mode: `monitor_snapshot`

## Quantitative Snapshot

- monitor nodes: `32`
- active final monitor communities: `3`
- largest final community share: `0.844`
- mean assignment churn: `0.678`
- mean changes among top-churn nodes: `12.88`
- mean confidence among top-churn nodes: `0.0175`

## Interpretation

The final assignment is moderately concentrated and churn remains noticeable. This case is useful for showing that community memory captures temporal movement rather than a static clustering snapshot.

The reported MRR/AP values describe predictive quality, while the community statistics describe the behavior of the soft memory layer. A large final community share means monitor nodes are concentrated in a small number of prototypes at the end of training. High churn means many monitor nodes changed their top-1 prototype between saved epochs. The two signals should be read together: high concentration with high churn indicates convergence after exploration; high concentration with low churn indicates stable assignment from early epochs.

## Artifact Guide

- `final_monitor_community_sizes.csv` and `final_monitor_community_sizes.png`: final distribution of monitored nodes over community prototypes.
- `final_monitor_community_type_mix.csv`: node-type composition of each final monitor community.
- `monitor_assignment_churn.csv` and `monitor_assignment_churn.png`: epochwise fraction of monitor nodes whose top community changed.
- `monitor_community_sizes_by_epoch.csv`, `monitor_community_heatmap.png`, and `monitor_community_size_evolution.png`: how community occupancy changes over training.
- `top_churn_nodes.csv` and `high_churn_member_trajectories.png`: examples of nodes with frequent community switches.
- `memory_evolution/`: memory-level plots exported from `plot_memory_evolution.py`.
- `community_vectors/`: community-prototype trajectory plots exported from `plot_community_vector_evolution.py`.

## Method Note

Monitor-node epochwise assignments are estimated from saved snapshot node memories and saved community vectors. `monitor_snapshot` artifacts are intended for fast paper visualization; exact replay artifacts, when present, replay train and validation history into the selected checkpoint before extracting final memberships.
