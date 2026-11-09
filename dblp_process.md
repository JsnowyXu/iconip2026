# dblp Process Log

## Scope
- Project root: `/home/xuxj/2026/dyn_relation_memory`
- Dataset root: `/home/xuxj/2026/datasets`
- Baseline root: `/home/xuxj/2026/baselines/src`
- Python env: `/home/xuxj/.conda/envs/pytorch/bin/python`

## Commands
- `KMP_DUPLICATE_LIB_OK=TRUE /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/prepare_dataset.py --dataset dblp --archive /home/xuxj/2026/datasets/DBLP-Citation-network-V18.zip --max-edges 200000 --text-encoder hybrid_hash --text-dim 64 --text-batch-size 128 --out-pt /home/xuxj/2026/dyn_relation_memory/data/processed/dblp_unified.pt --export-baselines --baseline-out-dir /home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/dblp`
- `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py --input /home/xuxj/2026/dyn_relation_memory/data/processed/dblp_unified.pt --out-dir /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks --name-prefix dblp`
- `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/analyze_datasets.py --inputs /home/xuxj/2026/dyn_relation_memory/data/processed/dblp_unified.pt /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/dblp_edge_transductive.pt /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/dblp_edge_inductive.pt /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/dblp_node_multilabel.pt --out-dir /home/xuxj/2026/dyn_relation_memory/data/reports`
- queued DHC main experiment:
`CUDA_VISIBLE_DEVICES=1 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/experiments/run_full_benchmark_repeats.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --out-root /home/xuxj/2026/dyn_relation_memory/results/paper_dhc_formal --datasets dblp,amazon_movies --seeds 42 --epochs 20 --batch-size 512 --lr 1e-3 --early-stop-patience 5 --max-parallel 1 --poll-sec 20 --mode main --variants full --gpu-ids 1 --save-memory-artifacts`
- queued baseline suite:
`CUDA_VISIBLE_DEVICES=2 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_all_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --repo-root /home/xuxj/2026 --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --out-root /home/xuxj/2026/dyn_relation_memory/results/baseline_formal_dblp_amazon --datasets dblp,amazon_movies --gpu-ids 2 --max-parallel 1 --epochs 10 --batch-size 256 --eval-num-negs 20 --poll-sec 20`

## Outputs
- `/home/xuxj/2026/dyn_relation_memory/data/processed/dblp_unified.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/dblp_edge_transductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/dblp_edge_inductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/dblp_node_multilabel.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/dblp`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/dblp_v18`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/dblp_edge_transductive`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/dblp_edge_inductive`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/dblp_node_multilabel`

## Progress Update
- `2026-04-16 00:13 CST`
- Preprocessing completed successfully.
- Unified graph stats:
  - nodes: `184938`
  - edges: `200000`
  - relation types: `2`
  - node types: `2`
  - multilabel tensor: `(184938, 4)`
  - total positive multilabel entries: `430406`
  - average labels per node: `2.3273`
- Task-view split sizes:
  - `dblp_edge_transductive`: train `146099`, val `36810`, test `17091`
  - `dblp_edge_inductive`: train `147886`, val `33360`, test `18754`
  - `dblp_node_multilabel`: train `146099`, val `36810`, test `17091`
- Analysis outputs are present under `data/reports/dblp_*`.
- GPU-side DHC and baseline jobs are queued to start automatically after the current patent node reruns release GPU 1 and GPU 2.

## Runtime Checkpoint
- `2026-04-16 17:18 CST`
- The original waiting chain did not actually launch the formal runs, so the jobs were restarted manually.
- Active DHC command:
`CUDA_VISIBLE_DEVICES=2 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/experiments/run_full_benchmark_repeats.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --out-root /home/xuxj/2026/dyn_relation_memory/results/paper_dhc_formal --datasets dblp,amazon_movies --seeds 42 --epochs 20 --batch-size 512 --lr 1e-3 --early-stop-patience 5 --max-parallel 1 --poll-sec 20 --mode main --variants full --gpu-ids 2 --save-memory-artifacts`
- Active DHC process:
  - scheduler PID: `2228875`
  - current child: `dblp_edge_transductive__full__s42`
  - current child PID: `2229203`
- Current DHC artifacts already present:
  - `/home/xuxj/2026/dyn_relation_memory/results/paper_dhc_formal/dblp_edge_transductive__full__s42/best_model.pt`
  - `/home/xuxj/2026/dyn_relation_memory/results/paper_dhc_formal/logs/dblp_edge_transductive__full__s42.log`
- Latest visible training state from the log:
  - reached `epoch=8`
  - current best checkpoint is at least `epoch=4`
  - validation AP already improved to `0.285604`
- Active baseline command:
`CUDA_VISIBLE_DEVICES=1 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_all_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --repo-root /home/xuxj/2026 --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --out-root /home/xuxj/2026/dyn_relation_memory/results/baseline_formal_dblp_amazon --datasets dblp,amazon_movies --gpu-ids 0 --max-parallel 1 --epochs 10 --batch-size 256 --eval-num-negs 20 --poll-sec 20`
- Active baseline process:
  - scheduler PID: `2270903`
- Baseline output root is now initialized:
  - `/home/xuxj/2026/dyn_relation_memory/results/baseline_formal_dblp_amazon/logs`
  - `/home/xuxj/2026/dyn_relation_memory/results/baseline_formal_dblp_amazon/metrics`
- At this checkpoint the baseline scheduler is alive, but the first per-task metric CSV has not been written yet.

## Failure And Recovery Checkpoint
- `2026-04-16 17:24 CST`
- The first baseline task `tgb2_custom__dblp_edge_transductive` failed immediately with:
  - `RuntimeError: CUDA error: out of memory`
- Re-checking the traceback showed the root cause was not the model itself but the wrapper implementation:
  - `scripts/baselines/run_tgb2_custom_smoke.py` moved the full graph tensors `src/dst/ts/msg/rel` to GPU at startup.
- Fix applied:
  - patched `scripts/baselines/run_tgb2_custom_smoke.py`
  - the wrapper now keeps the full graph on CPU and only transfers active minibatches plus replay slices to GPU
- Validation status after the patch:
  - launched smoke command:
`CUDA_VISIBLE_DEVICES=3 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_tgb2_custom_smoke.py --data /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/dblp_edge_transductive.pt --epochs 1 --batch-size 256 --gpu-id 0 --eval-num-negs 20 --out /home/xuxj/2026/dyn_relation_memory/results/baseline_formal_dblp_amazon/metrics/_smoke_tgb2_custom_dblp.csv`
  - active smoke PID: `2279847`
  - unlike the old launcher, the patched process stays alive beyond startup and uses only about `784 MiB` on the selected GPU at this checkpoint
- Next action:
  - if the smoke run completes normally, restart the full `dblp,amazon_movies` baseline suite with the patched wrapper
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-20 12:59:04
- launch `dblp_edge_inductive__seed42` on GPU `0`
## 2026-04-20 13:10:19
- finished `dblp_edge_inductive__seed42` rc=`0` elapsed_sec=`675.6`
## 2026-04-20 13:10:19
- launch `dblp_edge_transductive__seed42` on GPU `0`
## 2026-04-20 13:19:20
- finished `dblp_edge_transductive__seed42` rc=`0` elapsed_sec=`540.5`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-21 12:52:51
- launch `sthn__dblp_edge_inductive` on GPU `2`
## 2026-04-21 12:57:14
- launch `repeatmixer__dblp_edge_inductive` on GPU `1`
## 2026-04-21 13:06:07
- finished `sthn__dblp_edge_inductive` rc=`0` elapsed_sec=`795.6`
## 2026-04-21 13:06:07
- launch `sthn__dblp_edge_transductive` on GPU `2`
## 2026-04-21 13:09:29
- finished `repeatmixer__dblp_edge_inductive` rc=`0` elapsed_sec=`735.6`
## 2026-04-21 13:09:29
- launch `repeatmixer__dblp_edge_transductive` on GPU `1`
## 2026-04-21 13:20:07
- finished `sthn__dblp_edge_transductive` rc=`0` elapsed_sec=`840.7`
## 2026-04-21 13:22:15
- finished `repeatmixer__dblp_edge_transductive` rc=`0` elapsed_sec=`765.6`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-21 23:12:03
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-21 23:12:03
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-21 23:14:51
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-21 23:14:51
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-23 00:31:55
- launch `tgn_edgetype__dblp_edge_inductive` on GPU `1`
## 2026-04-23 00:33:40
- launch `tgn_edgetype__dblp_edge_transductive` on GPU `2`
## 2026-04-23 00:44:26
- finished `tgn_edgetype__dblp_edge_inductive` rc=`0` elapsed_sec=`750.7`
## 2026-04-23 00:47:56
- finished `tgn_edgetype__dblp_edge_transductive` rc=`0` elapsed_sec=`855.8`
## 2026-04-23 09:44:08
- launch targeted tuning `dblp_edge_inductive__rank_topk8_lr5e4__s42` on GPU `3`
- variant: `rank_topk8_lr5e4`
## 2026-04-23 10:05:09
- finished targeted tuning `dblp_edge_inductive__rank_topk8_lr5e4__s42` rc=`0` elapsed_sec=`1261.1`
## 2026-04-23 10:05:09
- launch targeted tuning `dblp_edge_inductive__rank_topk4_lr1e3__s42` on GPU `1`
- variant: `rank_topk4_lr1e3`
## 2026-04-23 10:19:55
- launch targeted tuning `dblp_edge_inductive__rank_nocomm_lr5e4__s42` on GPU `2`
- variant: `rank_nocomm_lr5e4`
## 2026-04-23 10:21:40
- finished targeted tuning `dblp_edge_inductive__rank_topk4_lr1e3__s42` rc=`0` elapsed_sec=`990.9`
## 2026-04-23 10:21:40
- launch targeted tuning `dblp_edge_transductive__rank_topk8_lr5e4__s42` on GPU `3`
- variant: `rank_topk8_lr5e4`
## 2026-04-23 10:36:11
- launch targeted tuning `dblp_edge_transductive__rank_topk4_lr1e3__s42` on GPU `1`
- variant: `rank_topk4_lr1e3`
## 2026-04-23 10:39:41
- finished targeted tuning `dblp_edge_inductive__rank_nocomm_lr5e4__s42` rc=`0` elapsed_sec=`1186.1`
## 2026-04-23 10:39:41
- launch targeted tuning `dblp_edge_transductive__rank_nocomm_lr5e4__s42` on GPU `2`
- variant: `rank_nocomm_lr5e4`
## 2026-04-23 10:43:26
- finished targeted tuning `dblp_edge_transductive__rank_topk8_lr5e4__s42` rc=`0` elapsed_sec=`1306.2`
## 2026-04-23 10:52:57
- finished targeted tuning `dblp_edge_transductive__rank_nocomm_lr5e4__s42` rc=`0` elapsed_sec=`795.7`
## 2026-04-23 10:57:42
- finished targeted tuning `dblp_edge_transductive__rank_topk4_lr1e3__s42` rc=`0` elapsed_sec=`1291.2`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-24 10:16:44
- launch `tgn__dblp_edge_inductive` on GPU `0`
## 2026-04-24 10:29:14
- finished `tgn__dblp_edge_inductive` rc=`0` elapsed_sec=`750.6`
## 2026-04-24 10:29:14
- launch `tgn__dblp_edge_transductive` on GPU `0`
## 2026-04-24 10:42:15
- finished `tgn__dblp_edge_transductive` rc=`0` elapsed_sec=`780.6`
## 2026-04-24 10:45:16
- launch `edgebank__dblp_edge_inductive` on GPU `0`
## 2026-04-24 10:45:31
- finished `edgebank__dblp_edge_inductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-24 10:45:31
- launch `edgebank__dblp_edge_transductive` on GPU `0`
## 2026-04-24 10:45:46
- finished `edgebank__dblp_edge_transductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-05-08 20:30:17
- TAMI baseline edge runs completed for `dblp_edge_inductive` and `dblp_edge_transductive`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/dblp_edge_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/dblp_edge_transductive.csv`
- Test metrics: inductive MRR `0.490323`, AUC `0.806934`, AP `0.144308`; transductive MRR `0.781263`, AUC `0.964893`, AP `0.449397`.

## 2026-05-08 22:12:13
- TAMI formal baseline rerun completed with `epochs=10`, `max_edges=50000`, `batch_size=256`, `eval_num_negs=20`, `early_stop_patience=5`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/dblp_edge_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/dblp_edge_transductive.csv`
- Test metrics: inductive MRR `0.478494`, AUC `0.810183`, AP `0.139577`; transductive MRR `0.795211`, AUC `0.974176`, AP `0.556297`.
## 2026-05-20 17:42:05
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-05-20 17:42:05
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-05-20 17:43:35
- launch `edgebank__dblp_edge_inductive` on GPU `0`
## 2026-05-20 17:43:50
- finished `edgebank__dblp_edge_inductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 17:43:50
- launch `edgebank__dblp_edge_transductive` on GPU `0`
## 2026-05-20 17:44:05
- finished `edgebank__dblp_edge_transductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 17:46:35
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-05-20 17:46:35
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-05-20 18:07:21
- launch `tabular_ml__dblp_edge_inductive` on GPU `0`
## 2026-05-20 18:09:06
- finished `tabular_ml__dblp_edge_inductive` rc=`0` elapsed_sec=`105.1`
## 2026-05-20 18:09:06
- launch `tabular_ml__dblp_edge_transductive` on GPU `0`
## 2026-05-20 18:10:51
- finished `tabular_ml__dblp_edge_transductive` rc=`0` elapsed_sec=`105.1`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-05-20 22:16:08
- launch `tabular_ml__dblp_edge_inductive` on GPU `0`
## 2026-05-20 22:17:38
- finished `tabular_ml__dblp_edge_inductive` rc=`0` elapsed_sec=`90.1`
## 2026-05-20 22:17:38
- launch `tabular_ml__dblp_edge_transductive` on GPU `0`
## 2026-05-20 22:19:08
- finished `tabular_ml__dblp_edge_transductive` rc=`0` elapsed_sec=`90.1`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `dblp_edge_inductive`
- relation_target: `rel_1`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `dblp_edge_transductive`
- relation_target: `rel_1`
## 2026-05-20 22:33:56
- launch `edgebank__dblp_edge_inductive` on GPU `0`
## 2026-05-20 22:34:11
- finished `edgebank__dblp_edge_inductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 22:34:11
- launch `edgebank__dblp_edge_transductive` on GPU `0`
## 2026-05-20 22:34:26
- finished `edgebank__dblp_edge_transductive` rc=`0` elapsed_sec=`15.0`
