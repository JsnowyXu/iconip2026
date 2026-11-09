# thgl_github Process Log

## 2026-04-20 00:39:58
- built THGL unified dataset
- command: `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/import_tgb2_thg.py --dataset-name thgl_github --edgelist-csv /home/xuxj/2026/datasets/tgb2/thgl-github/thgl-github_edgelist.csv --nodetype-csv /home/xuxj/2026/datasets/tgb2/thgl-github/thgl-github_nodetype.csv --out-pt /home/xuxj/2026/dyn_relation_memory/data/processed/thgl_github_unified.pt --max-edges -1`
- output: `/home/xuxj/2026/dyn_relation_memory/data/processed/thgl_github_unified.pt`
## 2026-04-20 00:40:56
- built THGL task views
- command: `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py --input /home/xuxj/2026/dyn_relation_memory/data/processed/thgl_github_unified.pt --out-dir /home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks --name-prefix thgl_github`
- outputs: `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_github_edge_transductive.pt`, `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_github_edge_inductive.pt`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_inductive`
- relation_target: `rel_5`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_transductive`
- relation_target: `rel_5`
## 2026-04-20 03:47:20
- launch `thgl_github_edge_inductive__seed42` on GPU `0`
## 2026-04-20 04:32:37
- finished `thgl_github_edge_inductive__seed42` rc=`1` elapsed_sec=`2717.3`
## 2026-04-20 04:32:37
- launch `thgl_github_edge_transductive__seed42` on GPU `0`
## 2026-04-20 05:19:10
- finished `thgl_github_edge_transductive__seed42` rc=`1` elapsed_sec=`2792.5`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_inductive`
- relation_target: `rel_5`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_transductive`
- relation_target: `rel_5`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_inductive`
- relation_target: `rel_5`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_transductive`
- relation_target: `rel_5`
## 2026-04-20 10:02:57
- inspected previous DHC failures for `thgl_github_edge_inductive` and `thgl_github_edge_transductive`
- confirmed failure mode was CUDA OOM from the original run, not dataset corruption
- scheduled deferred repair after current main GPU campaign exits
- repair settings: `batch-size=16`, `recent-neighbor-size=8`, seed `42`

## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_inductive`
- relation_target: `rel_5`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_transductive`
- relation_target: `rel_5`
## 2026-04-20 14:43:49
- stopped delayed repair chain that was waiting on an old process id
- relaunched direct repair on GPU 3
- active run: `thgl_github_edge_inductive__seed42`
- next queued run: `thgl_github_edge_transductive__seed42`
- repair log: `results/edge_only_formal/main/logs/thgl_github_edge_inductive__seed42_repair_bs16_gpu3.log`

## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_inductive`
- relation_target: `rel_5`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_transductive`
- relation_target: `rel_5`
## 2026-04-21 02:28:50
- launch `tgn_edgetype__thgl_github_edge_inductive` on GPU `0`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_inductive`
- relation_target: `rel_5`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_transductive`
- relation_target: `rel_5`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_inductive`
- relation_target: `rel_5`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_transductive`
- relation_target: `rel_5`
## 2026-04-22 23:13:15
- launch `tgn_edgetype__thgl_github_edge_inductive` on GPU `1`
## 2026-04-22 23:13:16
- launch `tgn_edgetype__thgl_github_edge_transductive` on GPU `2`
## 2026-04-22 23:18:16
- cancelled all active and queued `thgl_github` experiments by user request
- do not relaunch new `thgl_github` jobs in the current benchmark scope

## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `thgl_github_edge_inductive`
- relation_target: `rel_5`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `thgl_github_edge_transductive`
- relation_target: `rel_5`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_inductive`
- relation_target: `rel_5`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `thgl_github_edge_transductive`
- relation_target: `rel_5`
## 2026-04-27 23:29:58
- launch `sthn__thgl_github_edge_inductive` on GPU `2`
