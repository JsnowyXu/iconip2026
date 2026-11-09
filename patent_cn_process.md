# patent_cn Process Log

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
- `rows_total`: `200033`
- `rows_processed`: `195641`
- `rows_skipped_missing_org_or_code`: `4392`
- `num_patent_nodes`: `195641`
- `num_org_nodes`: `2160`
- `num_codeIII_nodes`: `589`
- `num_edges_patent_org`: `205448`
- `num_edges_patent_codeIII`: `258079`
- `num_edges_org_codeIII`: `270976`
- `num_edges_org_coorg`: `22772`
- `num_unique_org_code_pairs`: `62078`
- `num_unique_org_org_pairs`: `1268`
- `num_multilabel_classes`: `589`
- `num_labeled_patents`: `195641`

## Outputs
- `/home/xuxj/2026/dyn_relation_memory/data/processed/patent_cn_unified.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_cn_edge_org_codeIII_transductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_cn_edge_org_codeIII_inductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/tasks/patent_cn_node_patent_multilabel.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_cn_edge_org_codeIII_transductive/tgn/ml_patent_cn_edge_org_codeIII_transductive.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_cn_edge_org_codeIII_inductive/tgn/ml_patent_cn_edge_org_codeIII_inductive.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_cn_node_patent_multilabel/tgn/ml_patent_cn_node_patent_multilabel.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_cn_edge_org_codeIII_transductive/tgb2/patent_cn_edge_org_codeIII_transductive_edgelist.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_cn_edge_org_codeIII_inductive/tgb2/patent_cn_edge_org_codeIII_inductive_edgelist.csv`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/patent_cn_node_patent_multilabel/tgb2/patent_cn_node_patent_multilabel_edgelist.csv`

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
- `patent_cn` preprocessing, task-view generation, baseline adapter export, and dataset analysis have completed successfully.
- Current completed DHC outputs on disk:
  - `patent_cn_edge_org_codeIII_transductive__seed42/metrics.json`
  - `patent_cn_edge_org_codeIII_inductive__seed42/metrics.json`
  - `patent_cn_node_patent_multilabel__seed42/metrics.json`
- Important check:
  - the existing `patent_cn_node_patent_multilabel` run was produced before the task-type fix and should be rerun with explicit `--task-type node_multilabel`.
- Patent baseline suite was restarted in filtered mode without `STHN`:
`CUDA_VISIBLE_DEVICES=2 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_patent_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 2 --max-parallel 1 --epochs 10 --batch-size 256 --dyglib-max-edges 50000 --eval-num-negs 20 --poll-sec 20 --families tgb2_custom,dyglib_model_zoo,repeatmixer,edgebank`

## Progress Update
- `2026-04-15 22:10 CST`
- `patent_cn` edge-task DHC outputs remain valid:
  - `patent_cn_edge_org_codeIII_transductive__seed42/metrics.json`
  - `patent_cn_edge_org_codeIII_inductive__seed42/metrics.json`
- `patent_cn_node_patent_multilabel__seed42/metrics.json` is still the old pre-fix artifact and should not be treated as final.
- A chained rerun has been prepared to rerun the three patent node tasks with explicit:
`--task-type node_multilabel`
- That node rerun chain will start automatically after the active `patent_in_edge_org_codeIII_inductive` repair finishes.

## Progress Update
- `2026-04-16 00:05 CST`
- Patent node multilabel labels were re-audited directly from the task `.pt` file:
  - task view: `patent_cn_node_patent_multilabel`
  - label tensor shape: `(195641, 589)`
  - total positive labels: `258079`
  - row-wise label count: `min=1`, `max=9`, `mean=1.3191`
- Conclusion:
  - the `patent_cn` node multilabel target is correctly built
  - the old bad result was caused by the wrong task path being executed earlier, not by empty labels
- `patent_cn_node_patent_multilabel` has been included in the fresh node-only DHC rerun batch and the fresh node-only patent baseline rerun batch launched at `2026-04-16 00:05 CST`.
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 00:43:25
- launch `patent_cn_edge_org_codeIII_inductive__seed42` on GPU `0`
## 2026-04-20 01:09:57
- finished `patent_cn_edge_org_codeIII_inductive__seed42` rc=`0` elapsed_sec=`1591.4`
## 2026-04-20 01:09:57
- launch `patent_cn_edge_org_codeIII_transductive__seed42` on GPU `0`
## 2026-04-20 01:35:13
- finished `patent_cn_edge_org_codeIII_transductive__seed42` rc=`0` elapsed_sec=`1516.3`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 07:42:24
- launch `tgn_edgetype__patent_cn_edge_org_codeIII_inductive` on GPU `-1`
## 2026-04-20 08:49:27
- finished `tgn_edgetype__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`4023.6`
## 2026-04-20 08:49:27
- launch `sthn__patent_cn_edge_org_codeIII_inductive` on GPU `-1`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 09:52:08
- launch `sthn__patent_cn_edge_org_codeIII_inductive` on GPU `-1`
## 2026-04-20 09:52:08
- launch `repeatmixer__patent_cn_edge_org_codeIII_inductive` on GPU `-1`
## 2026-04-20 09:52:08
- launch `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive` on GPU `-1`
## 2026-04-20 09:52:09
- launch `tgn_edgetype__patent_cn_edge_org_codeIII_transductive` on GPU `-1`
## 2026-04-20 09:52:24
- finished `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-20 09:52:24
- launch `sthn__patent_cn_edge_org_codeIII_transductive` on GPU `-1`
## 2026-04-20 10:02:57
- diagnosed baseline anomaly in `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive`
- patched DyGLib adapter for current RepeatMixer-main APIs and relaunched standalone CPU repair job
- repair target: `results/edge_only_formal/baselines/metrics/repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive.csv`

## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 10:06:58
- launch `tgn_edgetype__patent_cn_edge_org_codeIII_inductive` on GPU `1`
## 2026-04-20 11:14:01
- finished `tgn_edgetype__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`4023.6`
## 2026-04-20 11:14:01
- launch `sthn__patent_cn_edge_org_codeIII_inductive` on GPU `1`
## 2026-04-20 12:44:36
- finished `sthn__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`5434.5`
## 2026-04-20 12:44:36
- launch `repeatmixer__patent_cn_edge_org_codeIII_inductive` on GPU `1`
## 2026-04-20 13:19:20
- launch `repeatmixer__patent_cn_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-20 13:42:39
- finished `repeatmixer__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`3483.0`
## 2026-04-20 13:42:39
- launch `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive` on GPU `1`
## 2026-04-20 13:59:22
- finished `repeatmixer__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`2402.1`
## 2026-04-20 13:59:22
- launch `tgn_edgetype__patent_cn_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-20 14:43:49
- finalized GPU baseline artifacts:
- `repeatmixer__patent_cn_edge_org_codeIII_inductive.csv`
- `tgn_edgetype__patent_cn_edge_org_codeIII_transductive.csv`
- GPU model-zoo repair remains running on GPU 1

## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 22:52:00
- launch `repeatmixer__patent_cn_edge_org_codeIII_transductive` on GPU `1`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-20 23:19:39
- launch `sthn__patent_cn_edge_org_codeIII_transductive` on GPU `2`
## 2026-04-21 00:14:34
- finished `repeatmixer__patent_cn_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`4954.3`
## 2026-04-21 00:22:57
- finished `sthn__patent_cn_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`3798.3`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 18:00:22
- launch `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive` on GPU `3`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:40:20
- launch `patent_cn_edge_org_codeIII_inductive__seed42` on GPU `1`
## 2026-04-21 22:40:20
- launch `patent_cn_edge_org_codeIII_transductive__seed42` on GPU `2`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-21 22:47:55
- launch `patent_cn_edge_org_codeIII_inductive__seed42` on GPU `1`
## 2026-04-21 22:47:55
- launch `patent_cn_edge_org_codeIII_transductive__seed42` on GPU `2`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 09:47:20
- launch `patent_cn_edge_org_codeIII_inductive__seed42` on GPU `0`
## 2026-04-22 09:47:20
- launch `patent_cn_edge_org_codeIII_transductive__seed42` on GPU `2`
## 2026-04-22 10:08:37
- finished `patent_cn_edge_org_codeIII_transductive__seed42` rc=`0` elapsed_sec=`1276.2`
## 2026-04-22 10:38:38
- finished `patent_cn_edge_org_codeIII_inductive__seed42` rc=`0` elapsed_sec=`3077.8`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:13:16
- launch `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-22 23:18:51
- launch `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-23 07:16:15
- launch targeted tuning `patent_cn_edge_org_codeIII_transductive__mrr_topk8_lr5e4__s42` on GPU `1`
- variant: `mrr_topk8_lr5e4`
## 2026-04-23 08:15:48
- launch targeted tuning `patent_cn_edge_org_codeIII_transductive__mrr_topk4_lr7e4__s42` on GPU `2`
- variant: `mrr_topk4_lr7e4`
## 2026-04-23 08:28:19
- finished targeted tuning `patent_cn_edge_org_codeIII_transductive__mrr_topk8_lr5e4__s42` rc=`0` elapsed_sec=`4324.0`
## 2026-04-23 09:11:36
- finished targeted tuning `patent_cn_edge_org_codeIII_transductive__mrr_topk4_lr7e4__s42` rc=`0` elapsed_sec=`3348.0`
## 2026-04-23 22:55:08
- finished `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive` rc=`-15` elapsed_sec=`84977.7`
## 2026-04-23 22:55:08
- launch `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-23 23:47:39
- launch `tgn__patent_cn_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-24 00:29:41
- finished `tgn__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`2522.3`
## 2026-04-24 00:29:41
- launch `tgn__patent_cn_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-24 01:06:43
- finished `tgn__patent_cn_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`2222.1`
## 2026-04-24 10:42:15
- launch `edgebank__patent_cn_edge_org_codeIII_inductive` on GPU `0`
## 2026-04-24 10:42:30
- finished `edgebank__patent_cn_edge_org_codeIII_inductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-24 10:42:30
- launch `edgebank__patent_cn_edge_org_codeIII_transductive` on GPU `0`
## 2026-04-24 10:42:45
- finished `edgebank__patent_cn_edge_org_codeIII_transductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-26 19:41:06
- launch `dygformer__patent_cn_edge_org_codeIII_inductive` on GPU `3`
## 2026-04-27 04:08:53
- finished `dygformer__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`30467.1`
## 2026-04-27 04:08:53
- launch `dygformer__patent_cn_edge_org_codeIII_transductive` on GPU `3`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-04-27 16:01:32
- launch `sthn__patent_cn_edge_org_codeIII_inductive` on GPU `2`
## 2026-04-27 16:51:35
- finished `sthn__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`3002.7`
## 2026-04-27 16:51:35
- launch `sthn__patent_cn_edge_org_codeIII_transductive` on GPU `2`
## 2026-04-27 18:05:23
- finished `sthn__patent_cn_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`4428.7`
## 2026-05-08 20:30:17
- TAMI baseline edge runs completed for `patent_cn_edge_org_codeIII_inductive` and `patent_cn_edge_org_codeIII_transductive`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/patent_cn_edge_org_codeIII_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/patent_cn_edge_org_codeIII_transductive.csv`
- Test metrics: inductive MRR `0.605196`, AUC `0.879264`, AP `0.394290`; transductive MRR `0.391302`, AUC `0.688073`, AP `0.154868`.

## 2026-05-08 22:12:13
- TAMI formal baseline rerun completed with `epochs=10`, `max_edges=50000`, `batch_size=256`, `eval_num_negs=20`, `early_stop_patience=5`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/patent_cn_edge_org_codeIII_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/patent_cn_edge_org_codeIII_transductive.csv`
- Test metrics: inductive MRR `0.803809`, AUC `0.962239`, AP `0.672295`; transductive MRR `0.589981`, AUC `0.857233`, AP `0.420703`.
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-05-20 21:49:21
- launch `tabular_ml__patent_cn_edge_org_codeIII_inductive` on GPU `0`
## 2026-05-20 21:51:51
- finished `tabular_ml__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`150.1`
## 2026-05-20 21:51:51
- launch `tabular_ml__patent_cn_edge_org_codeIII_transductive` on GPU `0`
## 2026-05-20 21:54:21
- finished `tabular_ml__patent_cn_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`150.1`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_inductive`
- relation_target: `org_codeIII`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `patent_cn_edge_org_codeIII_transductive`
- relation_target: `org_codeIII`
## 2026-05-20 22:31:25
- launch `edgebank__patent_cn_edge_org_codeIII_inductive` on GPU `0`
## 2026-05-20 22:31:40
- finished `edgebank__patent_cn_edge_org_codeIII_inductive` rc=`0` elapsed_sec=`15.1`
## 2026-05-20 22:31:40
- launch `edgebank__patent_cn_edge_org_codeIII_transductive` on GPU `0`
## 2026-05-20 22:31:55
- finished `edgebank__patent_cn_edge_org_codeIII_transductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 22:34:26
- launch `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive` on GPU `0`
