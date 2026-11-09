# Edge Benchmark Process Log

## 2026-04-20 00:33:32
- cleanup completed for non-main baseline/intermediate outputs
- removed path count: `7`
- manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/cleanup_manifest.json`
## 2026-04-20 00:38:24
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-04-20 07:41:32
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[-1]`
- seed: `42`
## 2026-04-20 09:51:54
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[-1]`
- seed: `42`
## 2026-04-20 10:02:57
- inspected live campaign status and anomalies
- confirmed main DHC line is still running on `thgl_software_edge_transductive`
- fixed `run_dyglib_model_zoo_smoke.py` compatibility for `raw_edge_ids`, model constructor signatures, `MergeLayer`, and CPU `--gpu-id -1` semantics
- fixed standalone CPU fallback in `run_repeatmixer_smoke.py` and `run_tgb2_custom_smoke.py`
- launched standalone repair for `repeatmixer_model_zoo__patent_cn_edge_org_codeIII_inductive`
- scheduled deferred DHC repair for `thgl_github_edge_inductive/transductive` after main GPU campaign exits using `batch-size=16` and `recent-neighbor-size=8`

## 2026-04-20 10:06:44
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[1]`
- seed: `42`
## 2026-04-20 14:43:49
- old mixed queue on GPU 0 was stopped after finishing its active baseline child
- confirmed GPU baseline outputs for `repeatmixer__patent_cn_edge_org_codeIII_inductive` and `tgn_edgetype__patent_cn_edge_org_codeIII_transductive`
- launched `thgl_github_edge_inductive` repair on GPU 3 with `batch-size=16` and `recent-neighbor-size=8`
- queued `thgl_github_edge_transductive` to run next on GPU 3 in the same repair session

## 2026-04-20 22:49:32
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-04-20 22:50:59
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[1]`
- seed: `42`
## 2026-04-20 22:51:47
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[1]`
- seed: `42`
## 2026-04-20 22:52:15
- reprioritized baseline execution for time-to-results
- kept DHC repair on GPU 3 unchanged
- activated fast baseline lane on GPU 0 for `tgn_edgetype`
- activated fast baseline lane on GPU 1 for `repeatmixer`
- deprioritized long-running `repeatmixer_model_zoo` job because it was dominating wall-clock time
- goal changed from strict original order to fastest continuous result delivery

## 2026-04-20 23:18:33
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[2]`
- seed: `42`
## 2026-04-20 23:18:59
- continued time optimization with multi-lane scheduling
- GPU 0 reserved for `tgn_edgetype` baseline lane
- GPU 1 reserved for `repeatmixer` baseline lane
- GPU 2 reserved for `sthn` baseline lane
- GPU 3 reserved for remaining DHC main repairs on `thgl_github`
- scheduling policy is now throughput-first instead of original family order

## 2026-04-21 11:59:36
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-04-21 12:00:06
- stopped `tgn_edgetype` baseline on `thgl_github` by request
- restarted GPU 0 baseline lane excluding `thgl_github`
- current GPU 0 task: `tgn_edgetype__thgl_software_edge_inductive`
- objective remains to finish all non-`thgl_github` baselines first

## 2026-04-21 13:20:07
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-04-21 13:22:15
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-04-21 18:00:10
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[3]`
- seed: `42`
## 2026-04-21 18:00:10
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[2]`
- seed: `42`
## 2026-04-21 18:00:10
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-04-21 18:04:38
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-04-21 22:40:08
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[1, 2]`
- seed: `42`
## 2026-04-21 22:47:43
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[1, 2]`
- seed: `42`
## 2026-04-21 23:11:51
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[1, 2]`
- seed: `42`
## 2026-04-21 23:14:38
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[2, 3]`
- seed: `42`
## 2026-04-22 09:43:00
- launched optimized `main` rerun after integrating explicit edge-type features into DHC and dataset-specific edge-task profiles
- command:
  - `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/experiments/run_edge_only_campaign.py --project-root /home/xuxj/2026/dyn_relation_memory --repo-root /home/xuxj/2026 --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --phase main --datasets patent_cn,patent_in,patent_us,aminer_paper,dblp,thgl_software --gpu-ids 0,2 --max-parallel 2 --no-skip-existing`
- active launch order at this checkpoint:
  - `patent_cn_edge_org_codeIII_inductive__seed42`
  - `patent_cn_edge_org_codeIII_transductive__seed42`
- run log:
  - `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/logs/patent_cn_edge_org_codeIII_inductive__seed42.log`
  - `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/logs/patent_cn_edge_org_codeIII_transductive__seed42.log`
## 2026-04-22 09:44:00
- launched focused edge-memory ablation to quantify the contribution of explicit edge-type features vs relation memory
- command:
  - `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/experiments/run_edge_memory_ablation.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --datasets patent_in,dblp,aminer_paper,thgl_software --task-types edge_transductive --epochs 6 --patience 4 --gpu-ids 3 --max-parallel 1 --no-skip-existing`
- first ablation run at this checkpoint:
  - `aminer_paper_citation_edge_transductive__full__s42`
- ablation log:
  - `/home/xuxj/2026/dyn_relation_memory/results/edge_memory_ablation/logs/aminer_paper_citation_edge_transductive__full__s42.log`
## 2026-04-22 09:47:08
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0, 2]`
- seed: `42`
## 2026-04-22 23:11:11
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-04-22 23:11:11
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[1, 2, 3]`
- seed: `42`
## 2026-04-22 23:18:16
- cancelled all active and queued `thgl_github` experiments by user request
- `thgl_github` is now excluded from new benchmark launches
- remaining baseline jobs will be relaunched on free GPUs without `thgl_github`

## 2026-04-22 23:18:34
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[1, 2, 3]`
- seed: `42`
## 2026-04-22 23:18:34
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-04-23 02:26:46
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-04-23 07:16:15
- targeted tuning campaign started for below-baseline edge tasks
- output root: `/home/xuxj/2026/dyn_relation_memory/results/targeted_edge_tuning_20260423`
- gpu_ids: `[1, 2, 3]`
- seed: `42`
## 2026-04-23 18:24:52
- targeted tuning campaign finished
- manifest: `/home/xuxj/2026/dyn_relation_memory/results/targeted_edge_tuning_20260423/run_manifest.csv`
- logs: `/home/xuxj/2026/dyn_relation_memory/results/targeted_edge_tuning_20260423/logs`
## 2026-04-23 23:02:12
- follow-up targeted tuning campaign started for 4 near-line tasks
- output root: `/home/xuxj/2026/dyn_relation_memory/results/targeted_edge_tuning_20260423_followup`
- gpu_ids: `[1, 3]`
- seed: `42`
## 2026-04-23 23:45:31
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-04-24 10:45:46
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-04-24 16:27:23
- follow-up targeted tuning campaign finished
- manifest: `/home/xuxj/2026/dyn_relation_memory/results/targeted_edge_tuning_20260423_followup/run_manifest.csv`
## 2026-04-26 19:41:06
- launch DyGFormer-only edge baseline campaign
- out_root: `/home/xuxj/2026/dyn_relation_memory/results/dygformer_edge_formal`
- datasets: `['patent_cn', 'patent_cn', 'patent_in', 'patent_in', 'patent_us', 'patent_us', 'thgl_github', 'thgl_github', 'thgl_software', 'thgl_software', 'amazon_movies', 'amazon_movies', 'aminer_paper', 'aminer_paper', 'dblp', 'dblp']`
- gpu_ids: `[3]`
## 2026-04-27 15:52:58
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[2]`
- seed: `42`

## 2026-05-08 20:30:17
- added TAMI custom edge runner for the new `/home/xuxj/2026/baselines/TAMI_temporal_graph-main.zip` baseline
- ran TAMI edge inductive/transductive tasks for `dblp`, `aminer_paper`, `patent_us`, `patent_in`, `patent_cn`, and `thgl_software`
- protocol: `epochs=1`, `max_edges=10000`, `batch_size=200`, `eval_num_negs=20`, `gamma=0.9`, `gpu_id=2`
- runner: `/home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_tami_custom_edge.py`
- summary: `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/summary.csv`
- per-task logs: `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/logs`

## 2026-05-08 22:12:13
- reran TAMI with formal baseline protocol from previous baseline experiments
- protocol: `epochs=10`, `max_edges=50000`, `batch_size=256`, `eval_num_negs=20`, `early_stop_patience=5`, `monitor_metric=val_mrr`, `gamma=0.9`, `gpu_id=2`
- completed datasets: `dblp`, `aminer_paper`, `patent_us`, `patent_in`, `patent_cn`, `thgl_software`
- runner: `/home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_tami_custom_edge.py`
- manifest: `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/run_manifest.csv`
- summary: `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/summary.csv`
- test summary: `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/test_summary.csv`
- per-task logs: `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/logs`
## 2026-05-11 18:00:55
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0, 2]`
- seed: `42`
## 2026-05-11 20:37:03
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-05-11 20:37:41
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0, 2]`
- seed: `42`
## 2026-05-12 01:36:57
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-05-12 01:37:21
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-05-12 06:27:34
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-05-20 17:40:35
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-05-20 17:44:05
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-05-20 17:45:07
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-05-20 18:10:51
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-05-20 21:46:46
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
## 2026-05-20 22:19:08
- edge-only benchmark campaign finished
- main manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/main/run_manifest.csv`
- baseline manifest: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/run_manifest.csv`
- baseline metrics long: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/metrics_long.csv`
- reports root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/reports`
## 2026-05-20 22:28:29
- edge-only benchmark campaign started
- output root: `/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal`
- gpu_ids: `[0]`
- seed: `42`
