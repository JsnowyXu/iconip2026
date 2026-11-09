# amazon_movies Process Log

## Scope
- Project root: `/home/xuxj/2026/dyn_relation_memory`
- Dataset root: `/home/xuxj/2026/datasets`
- Baseline root: `/home/xuxj/2026/baselines/src`
- Python env: `/home/xuxj/.conda/envs/pytorch/bin/python`

## Commands
- `KMP_DUPLICATE_LIB_OK=TRUE /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/prepare_dataset.py --dataset amazon_movies --archive /home/xuxj/2026/datasets/Amazon_movies.zip --max-edges 200000 --text-encoder hybrid_hash --text-dim 64 --text-batch-size 128 --out-pt /home/xuxj/2026/dyn_relation_memory/data/processed/amazon_movies_unified.pt --export-baselines --baseline-out-dir /home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/amazon_movies`
- `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py --input /home/xuxj/2026/dyn_relation_memory/data/processed/amazon_movies_unified.pt --out-dir /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks --name-prefix amazon_movies`
- `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/analyze_datasets.py --inputs /home/xuxj/2026/dyn_relation_memory/data/processed/amazon_movies_unified.pt /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/amazon_movies_edge_transductive.pt /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/amazon_movies_edge_inductive.pt /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/amazon_movies_node_multilabel.pt --out-dir /home/xuxj/2026/dyn_relation_memory/data/reports`
- queued DHC main experiment:
`CUDA_VISIBLE_DEVICES=1 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/experiments/run_full_benchmark_repeats.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --out-root /home/xuxj/2026/dyn_relation_memory/results/paper_dhc_formal --datasets dblp,amazon_movies --seeds 42 --epochs 20 --batch-size 512 --lr 1e-3 --early-stop-patience 5 --max-parallel 1 --poll-sec 20 --mode main --variants full --gpu-ids 1 --save-memory-artifacts`
- queued baseline suite:
`CUDA_VISIBLE_DEVICES=2 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_all_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --repo-root /home/xuxj/2026 --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --out-root /home/xuxj/2026/dyn_relation_memory/results/baseline_formal_dblp_amazon --datasets dblp,amazon_movies --gpu-ids 2 --max-parallel 1 --epochs 10 --batch-size 256 --eval-num-negs 20 --poll-sec 20`

## Outputs
- `/home/xuxj/2026/dyn_relation_memory/data/processed/amazon_movies_unified.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/amazon_movies_edge_transductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/amazon_movies_edge_inductive.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/amazon_movies_node_multilabel.pt`
- `/home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/amazon_movies`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/amazon_movies`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/amazon_movies_edge_transductive`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/amazon_movies_edge_inductive`
- `/home/xuxj/2026/dyn_relation_memory/data/reports/amazon_movies_node_multilabel`

## Progress Update
- `2026-04-16 00:13 CST`
- Preprocessing completed successfully.
- Unified graph stats:
  - nodes: `293562`
  - edges: `200000`
  - relation types: `5`
  - node types: `2`
  - multilabel tensor: `(293562, 4)`
  - total positive multilabel entries: `28244`
  - average labels per node: `0.0962`
- Task-view split sizes:
  - `amazon_movies_edge_transductive`: train `139880`, val `29981`, test `30139`
  - `amazon_movies_edge_inductive`: train `144521`, val `36734`, test `18745`
  - `amazon_movies_node_multilabel`: train `140013`, val `30087`, test `29900`
- Analysis outputs are present under `data/reports/amazon_movies*`.
- GPU-side DHC and baseline jobs are queued to start automatically after the current patent node reruns release GPU 1 and GPU 2.

## Runtime Checkpoint
- `2026-04-16 17:18 CST`
- The original waiting chain did not actually launch the formal runs, so the shared `dblp,amazon_movies` suites were restarted manually.
- Shared DHC command now running:
`CUDA_VISIBLE_DEVICES=2 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/experiments/run_full_benchmark_repeats.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --out-root /home/xuxj/2026/dyn_relation_memory/results/paper_dhc_formal --datasets dblp,amazon_movies --seeds 42 --epochs 20 --batch-size 512 --lr 1e-3 --early-stop-patience 5 --max-parallel 1 --poll-sec 20 --mode main --variants full --gpu-ids 2 --save-memory-artifacts`
- Shared DHC scheduler PID:
  - `2228875`
- Current active child run:
  - `dblp_edge_transductive__full__s42`
- This means `amazon_movies` has not started training yet; it is queued behind the current `dblp` run in the same suite.
- Shared baseline command now running:
`CUDA_VISIBLE_DEVICES=1 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_all_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --repo-root /home/xuxj/2026 --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --out-root /home/xuxj/2026/dyn_relation_memory/results/baseline_formal_dblp_amazon --datasets dblp,amazon_movies --gpu-ids 0 --max-parallel 1 --epochs 10 --batch-size 256 --eval-num-negs 20 --poll-sec 20`
- Shared baseline scheduler PID:
  - `2270903`
- Baseline output root has been created successfully:
  - `/home/xuxj/2026/dyn_relation_memory/results/baseline_formal_dblp_amazon`
- At this checkpoint no `amazon_movies` metric CSV has been produced yet because the shared scheduler has only just been restored.

## Failure And Recovery Checkpoint
- `2026-04-16 17:24 CST`
- The shared baseline suite hit an immediate failure before reaching `amazon_movies`:
  - `tgb2_custom__dblp_edge_transductive` failed with `CUDA out of memory`
- Root cause analysis showed the issue is in the shared TGB2 wrapper rather than in the dataset-specific task files:
  - `scripts/baselines/run_tgb2_custom_smoke.py` copied the full temporal graph to GPU before batching
- Fix applied:
  - patched `scripts/baselines/run_tgb2_custom_smoke.py`
  - changed it to CPU-resident full tensors with batch-wise GPU transfer
- Validation now in progress on the shared `dblp` entry task:
  - smoke PID: `2279847`
  - selected GPU: physical `GPU 3`
- `amazon_movies` baseline tasks remain queued behind this validation because both datasets share the same scheduler and wrapper implementation.
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-20 10:26:56
- launch `amazon_movies_edge_inductive__seed42` on GPU `0`
## 2026-04-20 10:45:12
- finished `amazon_movies_edge_inductive__seed42` rc=`0` elapsed_sec=`1095.9`
## 2026-04-20 10:45:12
- launch `amazon_movies_edge_transductive__seed42` on GPU `0`
## 2026-04-20 11:03:58
- finished `amazon_movies_edge_transductive__seed42` rc=`0` elapsed_sec=`1126.0`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-21 10:01:27
- launch `sthn__amazon_movies_edge_inductive` on GPU `2`
## 2026-04-21 10:16:28
- finished `sthn__amazon_movies_edge_inductive` rc=`0` elapsed_sec=`900.8`
## 2026-04-21 10:16:28
- launch `sthn__amazon_movies_edge_transductive` on GPU `2`
## 2026-04-21 10:18:50
- launch `repeatmixer__amazon_movies_edge_inductive` on GPU `1`
## 2026-04-21 10:30:58
- finished `sthn__amazon_movies_edge_transductive` rc=`0` elapsed_sec=`870.7`
## 2026-04-21 10:40:07
- finished `repeatmixer__amazon_movies_edge_inductive` rc=`0` elapsed_sec=`1276.1`
## 2026-04-21 10:40:07
- launch `repeatmixer__amazon_movies_edge_transductive` on GPU `1`
## 2026-04-21 11:01:38
- finished `repeatmixer__amazon_movies_edge_transductive` rc=`0` elapsed_sec=`1291.0`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-22 23:18:51
- launch `tgn_edgetype__amazon_movies_edge_inductive` on GPU `3`
## 2026-04-22 23:18:51
- launch `tgn_edgetype__amazon_movies_edge_transductive` on GPU `1`
## 2026-04-22 23:34:22
- finished `tgn_edgetype__amazon_movies_edge_transductive` rc=`0` elapsed_sec=`930.8`
## 2026-04-22 23:36:22
- finished `tgn_edgetype__amazon_movies_edge_inductive` rc=`0` elapsed_sec=`1051.0`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-24 08:04:51
- launch `tgn__amazon_movies_edge_inductive` on GPU `0`
## 2026-04-24 08:19:37
- finished `tgn__amazon_movies_edge_inductive` rc=`0` elapsed_sec=`885.7`
## 2026-04-24 08:19:37
- launch `tgn__amazon_movies_edge_transductive` on GPU `0`
## 2026-04-24 08:33:53
- finished `tgn__amazon_movies_edge_transductive` rc=`0` elapsed_sec=`855.8`
## 2026-04-24 10:44:16
- launch `edgebank__amazon_movies_edge_inductive` on GPU `0`
## 2026-04-24 10:44:31
- finished `edgebank__amazon_movies_edge_inductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-24 10:44:31
- launch `edgebank__amazon_movies_edge_transductive` on GPU `0`
## 2026-04-24 10:44:46
- finished `edgebank__amazon_movies_edge_transductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_inductive`
- relation_target: `rel_4`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `amazon_movies_edge_transductive`
- relation_target: `rel_4`
