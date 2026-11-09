# patent_us Process Log

## Scope
- Project root: `/home/xuxj/2026/dyn_relation_memory`
- Dataset root: `/home/xuxj/2026/datasets`
- Baseline root: `/home/xuxj/2026/baselines/src`
- Python env: `/home/xuxj/.conda/envs/pytorch/bin/python`
- GPU snapshot: `GPU 0 preferred after nvidia-smi check`

## Commands
- `nvidia-smi`
- `/home/xuxj/.conda/envs/pytorch/bin/python scripts/experiments/run_patent_preprocessing.py --project-root /home/xuxj/2026/dyn_relation_memory --dataset-root /home/xuxj/2026/datasets --python-bin /home/xuxj/.conda/envs/pytorch/bin/python`

## Main Counts
- `rows_total`: `200192`
- `rows_processed`: `200192`
- `rows_skipped_missing_org_or_code`: `0`
- `num_patent_nodes`: `200192`
- `num_org_nodes`: `520`
- `num_codeIII_nodes`: `609`
- `num_edges_patent_org`: `202361`
- `num_edges_patent_codeIII`: `417982`
- `num_edges_org_codeIII`: `422825`
- `num_edges_org_coorg`: `4596`
- `num_unique_org_code_pairs`: `42964`
- `num_unique_org_org_pairs`: `500`
- `num_multilabel_classes`: `609`
- `num_labeled_patents`: `200192`

## Outputs
- `/home/xuxj/2026/dyn_relation_memory/data/processed/patent_us_unified.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_us_edge_org_codeIII_transductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_us_edge_org_codeIII_inductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_us_node_patent_multilabel.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_us_edge_org_codeIII_transductive/tgn/ml_patent_us_edge_org_codeIII_transductive.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_us_edge_org_codeIII_inductive/tgn/ml_patent_us_edge_org_codeIII_inductive.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_us_node_patent_multilabel/tgn/ml_patent_us_node_patent_multilabel.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_us_edge_org_codeIII_transductive/tgb2/patent_us_edge_org_codeIII_transductive_edgelist.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_us_edge_org_codeIII_inductive/tgb2/patent_us_edge_org_codeIII_inductive_edgelist.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_us_node_patent_multilabel/tgb2/patent_us_node_patent_multilabel_edgelist.csv`## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 02:56:32
- launch `patent_us_edge_org_codeIII_inductive__seed42` on GPU `0`
## 2026-04-20 03:20:04
- finished `patent_us_edge_org_codeIII_inductive__seed42` rc=`0` elapsed_sec=`1411.2`
## 2026-04-20 03:20:04
- launch `patent_us_edge_org_codeIII_transductive__seed42` on GPU `0`
## 2026-04-20 03:47:20
- finished `patent_us_edge_org_codeIII_transductive__seed42` rc=`0` elapsed_sec=`1636.4`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 00:46:00
- launch `tgn_edgetype__patent_us_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-21 01:40:18
- finished `tgn_edgetype__patent_us_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`3257.9`
## 2026-04-21 01:40:18
- launch `tgn_edgetype__patent_us_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-21 02:28:50
- finished `tgn_edgetype__patent_us_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`2912.5`
## 2026-04-21 03:23:21
- launch `sthn__patent_us_edge_org_codeIII_inductive` on GPU `2`
## 2026-04-21 04:03:31
- launch `repeatmixer__patent_us_edge_org_codeIII_inductive` on GPU `1`
## 2026-04-21 04:37:55
- finished `sthn__patent_us_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`4473.9`
## 2026-04-21 04:37:55
- launch `sthn__patent_us_edge_org_codeIII_transductive` on GPU `2`
## 2026-04-21 05:33:36
- finished `repeatmixer__patent_us_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`5404.6`
## 2026-04-21 05:33:36
- launch `repeatmixer__patent_us_edge_org_codeIII_transductive` on GPU `1`
## 2026-04-21 06:04:29
- finished `sthn__patent_us_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`5194.5`
## 2026-04-21 07:32:57
- finished `repeatmixer__patent_us_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`7161.2`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 11:11:55
- launch `patent_us_edge_org_codeIII_inductive__seed42` on GPU `0`
## 2026-04-22 11:12:55
- launch `patent_us_edge_org_codeIII_transductive__seed42` on GPU `2`
## 2026-04-22 11:40:57
- finished `patent_us_edge_org_codeIII_inductive__seed42` rc=`0` elapsed_sec=`1741.7`
## 2026-04-22 11:42:57
- finished `patent_us_edge_org_codeIII_transductive__seed42` rc=`0` elapsed_sec=`1801.7`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-23 08:13:33
- launch targeted tuning `patent_us_edge_org_codeIII_transductive__mrr_topk8_lr5e4__s42` on GPU `1`
- variant: `mrr_topk8_lr5e4`
## 2026-04-23 09:37:22
- finished targeted tuning `patent_us_edge_org_codeIII_transductive__mrr_topk8_lr5e4__s42` rc=`0` elapsed_sec=`5029.4`
## 2026-04-23 09:37:22
- launch targeted tuning `patent_us_edge_org_codeIII_transductive__mrr_topk4_lr7e4__s42` on GPU `2`
- variant: `mrr_topk4_lr7e4`
## 2026-04-23 10:19:55
- finished targeted tuning `patent_us_edge_org_codeIII_transductive__mrr_topk4_lr7e4__s42` rc=`0` elapsed_sec=`2552.3`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-24 00:37:34
- launch follow-up tuning `patent_us_edge_org_codeIII_transductive__close_topk8_lr5e4_ap__s42` on GPU `3`
- variant: `close_topk8_lr5e4_ap`
## 2026-04-24 01:38:53
- launch follow-up tuning `patent_us_edge_org_codeIII_transductive__close_topk8_lr4e4_ap__s42` on GPU `1`
- variant: `close_topk8_lr4e4_ap`
## 2026-04-24 01:56:39
- finished follow-up tuning `patent_us_edge_org_codeIII_transductive__close_topk8_lr5e4_ap__s42` rc=`0` elapsed_sec=`4744.9`
## 2026-04-24 01:56:39
- launch follow-up tuning `patent_us_edge_org_codeIII_transductive__close_topk4_lr7e4_ap__s42` on GPU `3`
- variant: `close_topk4_lr7e4_ap`
## 2026-04-24 02:38:18
- launch `tgn__patent_us_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-24 03:02:13
- finished follow-up tuning `patent_us_edge_org_codeIII_transductive__close_topk4_lr7e4_ap__s42` rc=`0` elapsed_sec=`3934.0`
## 2026-04-24 03:08:43
- finished follow-up tuning `patent_us_edge_org_codeIII_transductive__close_topk8_lr4e4_ap__s42` rc=`0` elapsed_sec=`5390.5`
## 2026-04-24 03:21:21
- finished `tgn__patent_us_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`2582.4`
## 2026-04-24 03:21:21
- launch `tgn__patent_us_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-24 03:59:23
- finished `tgn__patent_us_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`2282.1`
## 2026-04-24 10:43:15
- launch `edgebank__patent_us_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-24 10:43:30
- finished `edgebank__patent_us_edge_org_codeIII_inductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-24 10:43:30
- launch `edgebank__patent_us_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-24 10:43:45
- finished `edgebank__patent_us_edge_org_codeIII_transductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-27 21:03:50
- launch `sthn__patent_us_edge_org_codeIII_inductive` on GPU `2`
## 2026-04-27 22:07:53
- finished `sthn__patent_us_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`3843.3`
## 2026-04-27 22:07:53
- launch `sthn__patent_us_edge_org_codeIII_transductive` on GPU `2`
## 2026-04-27 23:29:58
- finished `sthn__patent_us_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`4924.2`
## 2026-05-08 20:30:17
- TAMI baseline edge runs completed for `patent_us_edge_org_codeIII_inductive` and `patent_us_edge_org_codeIII_transductive`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/patent_us_edge_org_codeIII_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/patent_us_edge_org_codeIII_transductive.csv`
- Test metrics: inductive MRR `0.500000`, AUC `0.950000`, AP `0.500000`; transductive MRR `0.277153`, AUC `0.667615`, AP `0.087346`.

## 2026-05-08 22:12:13
- TAMI formal baseline rerun completed with `epochs=10`, `max_edges=50000`, `batch_size=256`, `eval_num_negs=20`, `early_stop_patience=5`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/patent_us_edge_org_codeIII_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/patent_us_edge_org_codeIII_transductive.csv`
- Test metrics: inductive MRR `0.666833`, AUC `0.887755`, AP `0.662930`; transductive MRR `0.563049`, AUC `0.836470`, AP `0.365132`.
## 2026-05-20 17:42:04
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-05-20 17:42:05
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-05-20 17:42:05
- launch `edgebank__patent_us_edge_org_codeIII_inductive` on GPU `0`
## 2026-05-20 17:42:20
- finished `edgebank__patent_us_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 17:42:20
- launch `edgebank__patent_us_edge_org_codeIII_transductive` on GPU `0`
## 2026-05-20 17:42:35
- finished `edgebank__patent_us_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 17:46:35
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-05-20 17:46:35
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-05-20 17:46:35
- launch `tabular_ml__patent_us_edge_org_codeIII_inductive` on GPU `0`
## 2026-05-20 17:49:35
- finished `tabular_ml__patent_us_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`180.2`
## 2026-05-20 17:49:35
- launch `tabular_ml__patent_us_edge_org_codeIII_transductive` on GPU `0`
## 2026-05-20 17:54:35
- finished `tabular_ml__patent_us_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`300.3`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-05-20 22:02:37
- launch `tabular_ml__patent_us_edge_org_codeIII_inductive` on GPU `0`
## 2026-05-20 22:04:52
- finished `tabular_ml__patent_us_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`135.1`
## 2026-05-20 22:04:52
- launch `tabular_ml__patent_us_edge_org_codeIII_transductive` on GPU `0`
## 2026-05-20 22:08:37
- finished `tabular_ml__patent_us_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`225.2`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `patent_us_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-05-20 22:32:25
- launch `edgebank__patent_us_edge_org_codeIII_inductive` on GPU `0`
## 2026-05-20 22:32:40
- finished `edgebank__patent_us_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 22:32:40
- launch `edgebank__patent_us_edge_org_codeIII_transductive` on GPU `0`
## 2026-05-20 22:32:55
- finished `edgebank__patent_us_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`15.0`
