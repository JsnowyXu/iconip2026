# thgl_software Process Log

## 2026-04-20 00:41:32
- built THGL unified dataset
- command: `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/import_tgb2_thg.py --dataset-name thgl_software --edgelist-csv /home/xuxj/2026/datasets/tgb2/thgl-software/thgl-software_edgelist.csv --nodetype-csv /home/xuxj/2026/datasets/tgb2/thgl-software/thgl-software_nodetype.csv --out-pt /home/xuxj/2026/dyn_relation_memory/data/processed/thgl_software_unified.pt --max-edges -1`
- output: `/home/xuxj/2026/dyn_relation_memory/data/processed/thgl_software_unified.pt`
## 2026-04-20 00:41:37
- built THGL task views
- command: `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py --input /home/xuxj/2026/dyn_relation_memory/data/processed/thgl_software_unified.pt --out-dir /home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks --name-prefix thgl_software`
- outputs: `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_software_edge_transductive.pt`, `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_software_edge_inductive.pt`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-20 05:19:10
- launch `thgl_software_edge_inductive__seed42` on GPU `0`
## 2026-04-20 06:18:13
- finished `thgl_software_edge_inductive__seed42` rc=`0` elapsed_sec=`3543.1`
## 2026-04-20 06:18:13
- launch `thgl_software_edge_transductive__seed42` on GPU `0`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-20 10:26:56
- finished `thgl_software_edge_transductive__seed42` rc=`0` elapsed_sec=`14922.9`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-21 06:04:29
- launch `sthn__thgl_software_edge_inductive` on GPU `2`
## 2026-04-21 07:32:57
- launch `repeatmixer__thgl_software_edge_inductive` on GPU `1`
## 2026-04-21 07:39:27
- finished `repeatmixer__thgl_software_edge_inductive` rc=`1` elapsed_sec=`390.3`
## 2026-04-21 07:39:27
- launch `repeatmixer__thgl_software_edge_transductive` on GPU `1`
## 2026-04-21 08:03:51
- finished `sthn__thgl_software_edge_inductive` rc=`0` elapsed_sec=`7161.2`
## 2026-04-21 08:03:51
- launch `sthn__thgl_software_edge_transductive` on GPU `2`
## 2026-04-21 10:01:27
- finished `sthn__thgl_software_edge_transductive` rc=`0` elapsed_sec=`7056.1`
## 2026-04-21 10:18:50
- finished `repeatmixer__thgl_software_edge_transductive` rc=`0` elapsed_sec=`9563.2`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-21 11:59:51
- launch `tgn_edgetype__thgl_software_edge_inductive` on GPU `0`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-21 18:00:23
- launch `repeatmixer__thgl_software_edge_inductive` on GPU `2`
## 2026-04-21 18:00:23
- launch `tgn_edgetype__thgl_software_edge_inductive` on GPU `0`
## 2026-04-21 18:04:23
- finished `repeatmixer__thgl_software_edge_inductive` rc=`1` elapsed_sec=`240.2`
## 2026-04-21 20:40:46
- finished `tgn_edgetype__thgl_software_edge_inductive` rc=`0` elapsed_sec=`9623.5`
## 2026-04-21 20:40:46
- launch `tgn_edgetype__thgl_software_edge_transductive` on GPU `0`
## 2026-04-22 09:31:00
- rebuilt `thgl_software` task views after fixing the inductive split fallback in `build_paper_task_views.py`
- command: `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/datasets/build_paper_task_views.py --input /home/xuxj/2026/dyn_relation_memory/data/processed/thgl_software_unified.pt --out-dir /home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks --name-prefix thgl_software`
- regenerated files:
  - `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_software_edge_transductive.pt`
  - `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_software_edge_inductive.pt`
  - `/home/xuxj/2026/dyn_relation_memory/data/processed/tgb2_tasks/thgl_software_node_multilabel.pt`
- verified split sizes after rebuild:
  - `thgl_software_edge_inductive`: train `1108627`, val `203697`, test `177482`
  - `thgl_software_edge_transductive`: train `1108627`, val `203697`, test `177482`
- note:
  - the old inductive task view had an empty validation window
  - the new fallback keeps a stable temporal validation window instead of producing `val=0`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-21 23:12:03
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-21 23:12:03
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-21 23:12:03
- launch `thgl_software_edge_inductive__seed42` on GPU `1`
## 2026-04-21 23:12:03
- launch `thgl_software_edge_transductive__seed42` on GPU `2`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-22 11:40:57
- launch `thgl_software_edge_inductive__seed42` on GPU `0`
## 2026-04-22 11:42:57
- launch `thgl_software_edge_transductive__seed42` on GPU `2`
## 2026-04-22 14:28:06
- finished `thgl_software_edge_transductive__seed42` rc=`0` elapsed_sec=`9909.0`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-22 23:13:16
- launch `tgn_edgetype__thgl_software_edge_transductive` on GPU `1`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-22 23:18:51
- launch `tgn_edgetype__thgl_software_edge_transductive` on GPU `2`
## 2026-04-23 00:44:26
- launch `repeatmixer__thgl_software_edge_inductive` on GPU `3`
## 2026-04-23 01:40:14
- finished `tgn_edgetype__thgl_software_edge_transductive` rc=`0` elapsed_sec=`8482.9`
## 2026-04-23 02:26:46
- finished `repeatmixer__thgl_software_edge_inductive` rc=`0` elapsed_sec=`6140.6`
## 2026-04-23 12:04:16
- launch targeted tuning `thgl_software_edge_inductive__rank_topk8_lr5e4__s42` on GPU `1`
- variant: `rank_topk8_lr5e4`
## 2026-04-23 12:37:18
- launch targeted tuning `thgl_software_edge_inductive__rank_no_edge_type__s42` on GPU `2`
- variant: `rank_no_edge_type`
## 2026-04-23 13:32:06
- launch targeted tuning `thgl_software_edge_transductive__rank_topk8_lr5e4__s42` on GPU `3`
- variant: `rank_topk8_lr5e4`
## 2026-04-23 15:25:57
- finished targeted tuning `thgl_software_edge_inductive__rank_no_edge_type__s42` rc=`0` elapsed_sec=`10119.2`
## 2026-04-23 15:25:57
- launch targeted tuning `thgl_software_edge_transductive__rank_no_edge_type__s42` on GPU `1`
- variant: `rank_no_edge_type`
## 2026-04-23 17:27:18
- finished targeted tuning `thgl_software_edge_inductive__rank_topk8_lr5e4__s42` rc=`0` elapsed_sec=`19382.7`
## 2026-04-23 18:22:36
- finished targeted tuning `thgl_software_edge_transductive__rank_topk8_lr5e4__s42` rc=`0` elapsed_sec=`17430.9`
## 2026-04-23 18:24:52
- finished targeted tuning `thgl_software_edge_transductive__rank_no_edge_type__s42` rc=`0` elapsed_sec=`10734.8`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-24 03:02:13
- launch follow-up tuning `thgl_software_edge_inductive__noetype_topk4_lr5e4__s42` on GPU `1`
- variant: `noetype_topk4_lr5e4`
## 2026-04-24 03:08:43
- launch follow-up tuning `thgl_software_edge_inductive__noetype_topk0_lr7e4__s42` on GPU `3`
- variant: `noetype_topk0_lr7e4`
## 2026-04-24 03:59:23
- launch `tgn__thgl_software_edge_inductive` on GPU `0`
## 2026-04-24 05:49:08
- finished follow-up tuning `thgl_software_edge_inductive__noetype_topk4_lr5e4__s42` rc=`0` elapsed_sec=`10015.4`
## 2026-04-24 05:49:08
- launch follow-up tuning `thgl_software_edge_inductive__noetype_topk8_lr3e4__s42` on GPU `1`
- variant: `noetype_topk8_lr3e4`
## 2026-04-24 05:56:24
- finished follow-up tuning `thgl_software_edge_inductive__noetype_topk0_lr7e4__s42` rc=`0` elapsed_sec=`10060.5`
## 2026-04-24 05:56:24
- launch follow-up tuning `thgl_software_edge_transductive__noetype_topk4_lr5e4__s42` on GPU `3`
- variant: `noetype_topk4_lr5e4`
## 2026-04-24 06:02:00
- finished `tgn__thgl_software_edge_inductive` rc=`0` elapsed_sec=`7356.8`
## 2026-04-24 06:02:00
- launch `tgn__thgl_software_edge_transductive` on GPU `0`
## 2026-04-24 08:04:51
- finished `tgn__thgl_software_edge_transductive` rc=`0` elapsed_sec=`7371.7`
## 2026-04-24 08:46:34
- finished follow-up tuning `thgl_software_edge_transductive__noetype_topk4_lr5e4__s42` rc=`0` elapsed_sec=`10210.4`
## 2026-04-24 08:46:34
- launch follow-up tuning `thgl_software_edge_transductive__noetype_topk0_lr7e4__s42` on GPU `1`
- variant: `noetype_topk0_lr7e4`
## 2026-04-24 10:43:45
- launch `edgebank__thgl_software_edge_inductive` on GPU `0`
## 2026-04-24 10:44:00
- finished `edgebank__thgl_software_edge_inductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-24 10:44:00
- launch `edgebank__thgl_software_edge_transductive` on GPU `0`
## 2026-04-24 10:44:15
- finished `edgebank__thgl_software_edge_transductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-24 11:04:12
- finished follow-up tuning `thgl_software_edge_inductive__noetype_topk8_lr3e4__s42` rc=`0` elapsed_sec=`18904.4`
## 2026-04-24 11:04:12
- launch follow-up tuning `thgl_software_edge_transductive__noetype_topk8_lr3e4__s42` on GPU `3`
- variant: `noetype_topk8_lr3e4`
## 2026-04-24 12:02:31
- finished follow-up tuning `thgl_software_edge_transductive__noetype_topk0_lr7e4__s42` rc=`0` elapsed_sec=`11757.0`
## 2026-04-24 16:27:23
- finished follow-up tuning `thgl_software_edge_transductive__noetype_topk8_lr3e4__s42` rc=`0` elapsed_sec=`19390.1`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-05-08 20:30:17
- TAMI baseline edge runs completed for `thgl_software_edge_inductive` and `thgl_software_edge_transductive`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/thgl_software_edge_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/thgl_software_edge_transductive.csv`
- Test metrics: inductive MRR `0.552713`, AUC `0.917339`, AP `0.244943`; transductive MRR `0.528775`, AUC `0.914209`, AP `0.223156`.

## 2026-05-08 22:12:13
- TAMI formal baseline rerun completed with `epochs=10`, `max_edges=50000`, `batch_size=256`, `eval_num_negs=20`, `early_stop_patience=5`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/thgl_software_edge_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/thgl_software_edge_transductive.csv`
- Test metrics: inductive MRR `0.812781`, AUC `0.972152`, AP `0.636898`; transductive MRR `0.802127`, AUC `0.968779`, AP `0.606058`.
## 2026-05-11 18:02:25
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-05-11 18:02:25
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-05-11 18:02:25
- launch `thgl_software_edge_inductive__seed42` on GPU `0`
## 2026-05-11 18:02:25
- launch `thgl_software_edge_transductive__seed42` on GPU `2`
## 2026-05-11 20:25:17
- finished `thgl_software_edge_transductive__seed42` rc=`0` elapsed_sec=`8572.2`
## 2026-05-11 20:37:03
- finished `thgl_software_edge_inductive__seed42` rc=`0` elapsed_sec=`9277.8`
## 2026-05-11 20:39:40
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-05-11 20:39:40
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-05-11 20:39:40
- launch `tgn__thgl_software_edge_inductive` on GPU `0`
## 2026-05-11 20:39:40
- launch `tgn__thgl_software_edge_transductive` on GPU `2`
## 2026-05-11 22:50:47
- finished `tgn__thgl_software_edge_transductive` rc=`0` elapsed_sec=`7866.9`
## 2026-05-11 22:50:48
- launch `tgn_edgetype__thgl_software_edge_inductive` on GPU `0`
## 2026-05-11 23:18:04
- finished `tgn__thgl_software_edge_inductive` rc=`0` elapsed_sec=`9503.7`
## 2026-05-11 23:18:04
- launch `tgn_edgetype__thgl_software_edge_transductive` on GPU `2`
## 2026-05-12 01:32:41
- finished `tgn_edgetype__thgl_software_edge_transductive` rc=`0` elapsed_sec=`8077.1`
## 2026-05-12 01:36:56
- finished `tgn_edgetype__thgl_software_edge_inductive` rc=`0` elapsed_sec=`9968.8`
## 2026-05-12 01:39:19
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-05-12 01:39:19
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-05-12 01:39:19
- launch `sthn__thgl_software_edge_inductive` on GPU `0`
## 2026-05-12 03:29:55
- finished `sthn__thgl_software_edge_inductive` rc=`0` elapsed_sec=`6635.6`
## 2026-05-12 03:29:55
- launch `sthn__thgl_software_edge_transductive` on GPU `0`
## 2026-05-12 06:27:34
- finished `sthn__thgl_software_edge_transductive` rc=`0` elapsed_sec=`10658.9`
## 2026-05-20 17:42:05
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-05-20 17:42:05
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-05-20 17:42:35
- launch `edgebank__thgl_software_edge_inductive` on GPU `0`
## 2026-05-20 17:42:50
- finished `edgebank__thgl_software_edge_inductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 17:42:50
- launch `edgebank__thgl_software_edge_transductive` on GPU `0`
## 2026-05-20 17:43:05
- finished `edgebank__thgl_software_edge_transductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 17:46:35
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-05-20 17:46:35
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-05-20 17:54:35
- launch `tabular_ml__thgl_software_edge_inductive` on GPU `0`
## 2026-05-20 17:58:05
- finished `tabular_ml__thgl_software_edge_inductive` rc=`0` elapsed_sec=`210.2`
## 2026-05-20 17:58:05
- launch `tabular_ml__thgl_software_edge_transductive` on GPU `0`
## 2026-05-20 18:01:36
- finished `tabular_ml__thgl_software_edge_transductive` rc=`0` elapsed_sec=`210.3`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-05-20 22:08:37
- launch `tabular_ml__thgl_software_edge_inductive` on GPU `0`
## 2026-05-20 22:10:37
- finished `tabular_ml__thgl_software_edge_inductive` rc=`0` elapsed_sec=`120.1`
## 2026-05-20 22:10:37
- launch `tabular_ml__thgl_software_edge_transductive` on GPU `0`
## 2026-05-20 22:12:38
- finished `tabular_ml__thgl_software_edge_transductive` rc=`0` elapsed_sec=`120.1`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_inductive`
- relation_target: `rel_6`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `thgl_software_edge_transductive`
- relation_target: `rel_6`
## 2026-05-20 22:32:55
- launch `edgebank__thgl_software_edge_inductive` on GPU `0`
## 2026-05-20 22:33:11
- finished `edgebank__thgl_software_edge_inductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 22:33:11
- launch `edgebank__thgl_software_edge_transductive` on GPU `0`
## 2026-05-20 22:33:26
- finished `edgebank__thgl_software_edge_transductive` rc=`0` elapsed_sec=`15.0`
