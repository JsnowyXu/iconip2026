# patent_in Process Log

## Scope
- Project root: `/home/xuxj/2026/dyn_relation_memory`
- Dataset root: `/home/xuxj/2026/datasets`
- Baseline root: `/home/xuxj/2026/baselines/src`
- Python env: `/home/xuxj/.conda/envs/pytorch/bin/python`
- GPU snapshot: `GPU 0 selected after 2026-04-15 16:39 CST nvidia-smi check`

## Commands
- `nvidia-smi`
- `/home/xuxj/.conda/envs/pytorch/bin/python scripts/experiments/run_patent_preprocessing.py --project-root /home/xuxj/2026/dyn_relation_memory --dataset-root /home/xuxj/2026/datasets --python-bin /home/xuxj/.conda/envs/pytorch/bin/python`

## Main Counts
- `rows_total`: `160718`
- `rows_processed`: `160514`
- `rows_skipped_missing_org_or_code`: `204`
- `num_patent_nodes`: `160514`
- `num_org_nodes`: `3166`
- `num_codeIII_nodes`: `627`
- `num_edges_patent_org`: `186660`
- `num_edges_patent_codeIII`: `411184`
- `num_edges_org_codeIII`: `483440`
- `num_edges_org_coorg`: `73520`
- `num_unique_org_code_pairs`: `139748`
- `num_unique_org_org_pairs`: `8953`
- `num_multilabel_classes`: `627`
- `num_labeled_patents`: `160514`

## Outputs
- `/home/xuxj/2026/dyn_relation_memory/data/processed/patent_in_unified.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_in_edge_org_codeIII_transductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_in_edge_org_codeIII_inductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_in_node_patent_multilabel.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_in_edge_org_codeIII_transductive/tgn/ml_patent_in_edge_org_codeIII_transductive.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_in_edge_org_codeIII_inductive/tgn/ml_patent_in_edge_org_codeIII_inductive.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_in_node_patent_multilabel/tgn/ml_patent_in_node_patent_multilabel.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_in_edge_org_codeIII_transductive/tgb2/patent_in_edge_org_codeIII_transductive_edgelist.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_in_edge_org_codeIII_inductive/tgb2/patent_in_edge_org_codeIII_inductive_edgelist.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_in_node_patent_multilabel/tgb2/patent_in_node_patent_multilabel_edgelist.csv`

## Experiment Launch
- DHC suite command:
`CUDA_VISIBLE_DEVICES=1 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/experiments/run_patent_dhc_suite.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 1 --seeds 42 --epochs 20 --batch-size 512 --lr 1e-3 --early-stop-patience 5 --max-parallel 1 --poll-sec 20 --save-memory-artifacts`
- Baseline suite command:
`CUDA_VISIBLE_DEVICES=2 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_patent_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 2 --max-parallel 1 --epochs 10 --batch-size 256 --dyglib-max-edges 50000 --eval-num-negs 20 --poll-sec 20`
- DHC output root:
`/home/xuxj/2026/dyn_relation_memory/results/patent_dhc_formal`
- Baseline output root:
`/home/xuxj/2026/dyn_relation_memory/results/patent_baseline_formal`

## Progress Update
- `2026-04-15 22:05 CST`
- `patent_in` preprocessing, task-view generation, baseline adapter export, and dataset analysis have completed successfully.
- Current completed DHC output on disk:
  - `patent_in_edge_org_codeIII_transductive__seed42/metrics.json`
- Current repair rerun in progress:
`/home/xuxj/.conda/envs/pytorch/bin/python -m dhc.train.run_train --data /home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_in_edge_org_codeIII_inductive.pt --out-dir /home/xuxj/2026/dyn_relation_memory/results/patent_dhc_formal/patent_in_edge_org_codeIII_inductive__seed42 --seed 42 --epochs 20 --batch-size 512 --lr 0.001 --gpu-id 1 --early-stop-patience 5 --ml-threshold-search --save-memory-artifacts --task-type edge_inductive`
- Important check:
  - `patent_in_node_patent_multilabel` has not yet been rerun after the task-type fix and remains pending.
- Patent baseline suite was restarted in filtered mode without `STHN`:
`CUDA_VISIBLE_DEVICES=2 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_patent_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 2 --max-parallel 1 --epochs 10 --batch-size 256 --dyglib-max-edges 50000 --eval-num-negs 20 --poll-sec 20 --families tgb2_custom,dyglib_model_zoo,repeatmixer,edgebank`

## Progress Update
- `2026-04-15 22:10 CST`
- Active DHC repair remains in progress:
  - PID `264183`
  - command:
`/home/xuxj/.conda/envs/pytorch/bin/python -m dhc.train.run_train --data /home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_in_edge_org_codeIII_inductive.pt --out-dir /home/xuxj/2026/dyn_relation_memory/results/patent_dhc_formal/patent_in_edge_org_codeIII_inductive__seed42 --seed 42 --epochs 20 --batch-size 512 --lr 0.001 --gpu-id 1 --early-stop-patience 5 --ml-threshold-search --save-memory-artifacts --task-type edge_inductive`
- Runtime check:
  - the output directory has been updated by the rerun
  - `best_model.pt` refreshed at `2026-04-15 22:08 CST`
  - no final `metrics.json` yet at this checkpoint
- Follow-up already queued:
  - after this edge repair exits, a chained rerun will execute:
    - `patent_us_node_patent_multilabel`
    - `patent_cn_node_patent_multilabel`
    - `patent_in_node_patent_multilabel`
  - all with explicit `--task-type node_multilabel`

## Progress Update
- `2026-04-16 00:05 CST`
- `patent_in_edge_org_codeIII_inductive__seed42/metrics.json` is now present and valid after the explicit repair rerun.
- Patent node multilabel labels were re-audited directly from the task `.pt` file:
  - task view: `patent_in_node_patent_multilabel`
  - label tensor shape: `(160514, 627)`
  - total positive labels: `411184`
  - row-wise label count: `min=1`, `max=7`, `mean=2.5617`
- Conclusion:
  - the `patent_in` node multilabel target is correctly built
  - the remaining issue was evaluation/run routing, not label construction
- The previous chained node rerun was replaced by a clean node-only DHC suite relaunch and a clean patent node baseline relaunch at `2026-04-16 00:05 CST` so the new top-k node metrics are guaranteed to be written.
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 01:35:13
- launch `patent_in_edge_org_codeIII_inductive__seed42` on GPU `0`
## 2026-04-20 02:19:31
- finished `patent_in_edge_org_codeIII_inductive__seed42` rc=`0` elapsed_sec=`2657.4`
## 2026-04-20 02:19:31
- launch `patent_in_edge_org_codeIII_transductive__seed42` on GPU `0`
## 2026-04-20 02:56:32
- finished `patent_in_edge_org_codeIII_transductive__seed42` rc=`0` elapsed_sec=`2221.9`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:50:39
- launch `tgn_edgetype__patent_in_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:51:12
- launch `tgn_edgetype__patent_in_edge_org_codeIII_inductive` on GPU `1`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 23:48:42
- finished `tgn_edgetype__patent_in_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`3483.1`
## 2026-04-20 23:48:42
- launch `tgn_edgetype__patent_in_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-21 00:14:34
- launch `repeatmixer__patent_in_edge_org_codeIII_inductive` on GPU `1`
## 2026-04-21 00:22:57
- launch `sthn__patent_in_edge_org_codeIII_inductive` on GPU `2`
## 2026-04-21 00:46:00
- finished `tgn_edgetype__patent_in_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`3437.9`
## 2026-04-21 01:48:31
- finished `sthn__patent_in_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`5134.3`
## 2026-04-21 01:48:31
- launch `sthn__patent_in_edge_org_codeIII_transductive` on GPU `2`
## 2026-04-21 01:56:10
- finished `repeatmixer__patent_in_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`6095.3`
## 2026-04-21 01:56:10
- launch `repeatmixer__patent_in_edge_org_codeIII_transductive` on GPU `1`
## 2026-04-21 03:23:21
- finished `sthn__patent_in_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`5689.8`
## 2026-04-21 04:03:31
- finished `repeatmixer__patent_in_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`7641.5`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 10:08:37
- launch `patent_in_edge_org_codeIII_inductive__seed42` on GPU `0`
## 2026-04-22 10:38:38
- launch `patent_in_edge_org_codeIII_transductive__seed42` on GPU `2`
## 2026-04-22 11:11:55
- finished `patent_in_edge_org_codeIII_inductive__seed42` rc=`0` elapsed_sec=`3798.5`
## 2026-04-22 11:12:55
- finished `patent_in_edge_org_codeIII_transductive__seed42` rc=`0` elapsed_sec=`2056.9`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-23 07:16:15
- launch targeted tuning `patent_in_edge_org_codeIII_inductive__mrr_topk8_lr5e4__s42` on GPU `2`
- variant: `mrr_topk8_lr5e4`
## 2026-04-23 07:16:15
- launch targeted tuning `patent_in_edge_org_codeIII_transductive__mrr_topk8_lr5e4__s42` on GPU `3`
- variant: `mrr_topk8_lr5e4`
## 2026-04-23 08:13:33
- finished targeted tuning `patent_in_edge_org_codeIII_transductive__mrr_topk8_lr5e4__s42` rc=`0` elapsed_sec=`3438.1`
## 2026-04-23 08:15:48
- finished targeted tuning `patent_in_edge_org_codeIII_inductive__mrr_topk8_lr5e4__s42` rc=`0` elapsed_sec=`3573.3`
## 2026-04-23 08:28:19
- launch targeted tuning `patent_in_edge_org_codeIII_inductive__mrr_topk4_lr7e4__s42` on GPU `3`
- variant: `mrr_topk4_lr7e4`
## 2026-04-23 09:11:36
- launch targeted tuning `patent_in_edge_org_codeIII_transductive__mrr_topk4_lr7e4__s42` on GPU `1`
- variant: `mrr_topk4_lr7e4`
## 2026-04-23 09:44:08
- finished targeted tuning `patent_in_edge_org_codeIII_inductive__mrr_topk4_lr7e4__s42` rc=`0` elapsed_sec=`4549.0`
## 2026-04-23 10:36:11
- finished targeted tuning `patent_in_edge_org_codeIII_transductive__mrr_topk4_lr7e4__s42` rc=`0` elapsed_sec=`5074.6`
## 2026-04-23 23:02:12
- launch follow-up tuning `patent_in_edge_org_codeIII_transductive__close_topk8_lr5e4_ap__s42` on GPU `1`
- variant: `close_topk8_lr5e4_ap`
## 2026-04-23 23:02:12
- launch follow-up tuning `patent_in_edge_org_codeIII_transductive__close_topk8_lr4e4_ap__s42` on GPU `3`
- variant: `close_topk8_lr4e4_ap`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-24 00:24:03
- finished follow-up tuning `patent_in_edge_org_codeIII_transductive__close_topk8_lr4e4_ap__s42` rc=`0` elapsed_sec=`4910.5`
## 2026-04-24 00:24:03
- launch follow-up tuning `patent_in_edge_org_codeIII_transductive__close_topk4_lr7e4_ap__s42` on GPU `1`
- variant: `close_topk4_lr7e4_ap`
## 2026-04-24 00:37:34
- finished follow-up tuning `patent_in_edge_org_codeIII_transductive__close_topk8_lr5e4_ap__s42` rc=`0` elapsed_sec=`5721.3`
## 2026-04-24 01:06:43
- launch `tgn__patent_in_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-24 01:38:53
- finished follow-up tuning `patent_in_edge_org_codeIII_transductive__close_topk4_lr7e4_ap__s42` rc=`0` elapsed_sec=`4489.7`
## 2026-04-24 01:56:01
- finished `tgn__patent_in_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`2957.7`
## 2026-04-24 01:56:01
- launch `tgn__patent_in_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-24 02:38:18
- finished `tgn__patent_in_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`2537.4`
## 2026-04-24 10:42:45
- launch `edgebank__patent_in_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-24 10:43:00
- finished `edgebank__patent_in_edge_org_codeIII_inductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-24 10:43:00
- launch `edgebank__patent_in_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-24 10:43:15
- finished `edgebank__patent_in_edge_org_codeIII_transductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-27 18:05:26
- launch `sthn__patent_in_edge_org_codeIII_inductive` on GPU `2`
## 2026-04-27 19:34:00
- finished `sthn__patent_in_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`5314.5`
## 2026-04-27 19:34:01
- launch `sthn__patent_in_edge_org_codeIII_transductive` on GPU `2`
## 2026-04-27 21:03:50
- finished `sthn__patent_in_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`5389.6`
## 2026-05-08 20:30:17
- TAMI baseline edge runs completed for `patent_in_edge_org_codeIII_inductive` and `patent_in_edge_org_codeIII_transductive`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/patent_in_edge_org_codeIII_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/patent_in_edge_org_codeIII_transductive.csv`
- Test metrics: inductive MRR `0.592952`, AUC `0.895267`, AP `0.324814`; transductive MRR `0.513002`, AUC `0.755696`, AP `0.270945`.

## 2026-05-08 22:12:13
- TAMI formal baseline rerun completed with `epochs=10`, `max_edges=50000`, `batch_size=256`, `eval_num_negs=20`, `early_stop_patience=5`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/patent_in_edge_org_codeIII_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/patent_in_edge_org_codeIII_transductive.csv`
- Test metrics: inductive MRR `0.802958`, AUC `0.966958`, AP `0.644915`; transductive MRR `0.669564`, AUC `0.897265`, AP `0.499266`.
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-05-20 21:54:21
- launch `tabular_ml__patent_in_edge_org_codeIII_inductive` on GPU `0`
## 2026-05-20 21:58:22
- finished `tabular_ml__patent_in_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`240.2`
## 2026-05-20 21:58:22
- launch `tabular_ml__patent_in_edge_org_codeIII_transductive` on GPU `0`
## 2026-05-20 22:02:37
- finished `tabular_ml__patent_in_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`255.2`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `patent_in_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-05-20 22:31:55
- launch `edgebank__patent_in_edge_org_codeIII_inductive` on GPU `0`
## 2026-05-20 22:32:10
- finished `edgebank__patent_in_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 22:32:10
- launch `edgebank__patent_in_edge_org_codeIII_transductive` on GPU `0`
## 2026-05-20 22:32:25
- finished `edgebank__patent_in_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`15.0`
