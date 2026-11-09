# Main Edge Visual Analysis Index

Generated artifacts are organized as `dataset/task/{node_memory,rel_memory,relation_type,community_memory,task_definition,performance,edge_semantics}`.

Top-level tables:
- `task_definitions.csv`: concrete link prediction target and context relations.
- `relation_split_counts.csv`: relation counts by train/val/test split.
- `dynamic_memory_availability.csv`: confirms node/relation_type/community dynamic traces.
- `performance_summary.csv`: DHC-main and baseline metrics/time/parameter summary.
- `baseline_epoch_curves.csv`: parsed baseline loss/val/time curves.
- `edge_semantic_importance.csv`: ablation deltas for edge/relation semantics.

Per-task visual files:
- `*_pca2d.csv` and `*_pca2d_all_epochs.png`: two-dimensional memory trajectories.
- `*_pca2d_first_last.png`: first-vs-last epoch semantic movement.
- `*_centroid_trajectory.png`: memory centroid evolution.
- `*_norm_by_epoch.csv/png`: memory magnitude evolution over epochs.
- `relation_composition.png` and `supervised_edges_by_split.png`: task semantics.
- `main_loss_metric_curve.png`, `baseline_loss_val_mrr_curves.png`, `main_epoch_time_curve.png`, `baseline_epoch_time_curves.png`, `test_mrr_by_model.png`, `avg_epoch_sec_by_model.png`: performance and time evidence.
- `figures/overview_test_mrr_by_model.png` and `figures/overview_avg_epoch_sec_by_model.png`: cross-task overview plots.

Datasets:
- `patent_us`
- `aminer_paper`
- `dblp`
- `thgl_software`
