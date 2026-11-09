# Dyn Relation Memory Process

- 代码目录：/home/xuxj/2026/dyn_relation_memory 
- 数据集目录：/home/xuxj/2026/datasets
- baseline 仓库目录：/home/xuxj/2026/baselines/src,tgb2.0 的tgn_edgetype,sthn,edgebank repeatmixer中的repeatmixer,tgn,dygformer
- 数据集：amazon_movie, dblp,patent_cn,patent_us,patent_in,thgl_software
实验环境用pytorch，所有实现都直接在服务器GPU上跑，先看占用情况，再分配GPU

## 2026-04-26

### Current Main Work

- Main experiments, ablations, sensitivity runs, and DyGFormer edge baselines are already running in `dyn_relation_memory`.
- Active long-running schedulers at the time of this note:
  - `scripts/experiments/run_best_config_module_ablations.py` on GPU `0,1`
  - `scripts/experiments/run_best_config_sensitivity.py` on GPU `2`
  - `scripts/baselines/run_dygformer_edge_formal.py` on GPU `3`

### STHN Baseline Audit

- User reported that `results/edge_only_formal/baselines/metrics/sthn__*inductive.csv` and `sthn__*transductive.csv` looked identical for some datasets.
- Audit conclusion:
  - This is a real baseline pipeline issue for `STHN`, not just a display/export issue.
  - Old `STHN` wrapper flow exported only edge CSV + node type CSV, then let the official TGB2 loader regenerate `train/val/test` by timestamp quantiles.
  - As a result, task-view `.pt` files with different explicit splits could still be evaluated with the same regenerated split.

### Confirmed Findings

- `patent_cn`, `patent_in`, `patent_us`:
  - `STHN` inductive/transductive outputs were already different, so these were not the suspicious cases.
- `amazon_movies`, `aminer_paper_citation`, `dblp`:
  - Task `.pt` files have different explicit `split` definitions.
  - Old `STHN` wrapper discarded those splits, so reruns were needed.
- `thgl_software`:
  - Upstream task-view data itself is currently abnormal.
  - `thgl_software_edge_inductive.pt` and `thgl_software_edge_transductive.pt` have the same edge content and the same split, so identical baseline results there are expected until task generation is fixed upstream.

### Code Fixes Applied

- Preserved task-view split overrides during official TGB2/STHN dataset export:
  - `scripts/baselines/prepare_tgb2_official_dataset.py`
- Modified local TGB2 loader to prefer explicit task split files when present instead of regenerating quantile splits:
  - `../baselines/src/TGB2-main/TGB2-main/tgb/linkproppred/dataset.py`

### Validation After Fix

- Verified with `dblp` that the loader now reads the intended task split counts instead of the old regenerated `70/15/15` split.
- This confirms the `STHN` rerun path is now correct for datasets whose task-view splits are valid.

### Rerun Plan

- Re-run `STHN` edge baselines from `scripts/experiments/run_edge_only_campaign.py` with:
  - `--phase baseline`
  - `--families sthn`
  - `--no-skip-existing`
- A GPU-idle watcher has been added so rerun starts automatically when any GPU becomes idle.
- Note:
  - `thgl_software` rerun will still stay identical until its task-view generation bug is fixed.
  - `amazon_movies`, `aminer_paper_citation`, and `dblp` are the main datasets expected to change after rerun.

### Next Turn Suggested Start

- Check whether the GPU-idle watcher has started the `STHN` rerun.
- After rerun finishes, compare old/new `STHN` metrics for:
  - `amazon_movies`
  - `aminer_paper_citation`
  - `dblp`
- Then inspect and fix upstream `thgl_software` task generation so inductive/transductive task views truly diverge.

## 2026-04-27

### Current Experiment Progress Check

- Checked the latest result directories under `results/`.
- `results/current_snapshot` was regenerated at `2026-04-27 04:08`.
- `results/best_config_module_ablation` has partial outputs:
  - planned runs: `32`
  - completed `metrics.json`: `30`
  - still missing:
    - `thgl_software_edge_transductive__no_edge_type__s42`
    - `thgl_software_edge_transductive__no_community__s42`
- `results/best_config_sensitivity` is still in an early partial state:
  - planned runs: `192`
  - completed `metrics.json`: `26`
  - completed scope is currently the full `patent_cn_edge_org_codeIII_inductive` sweep plus the first two `patent_cn_edge_org_codeIII_transductive` typed-negative runs.
- At the time of this check, no active scheduler/train process was found for:
  - `run_best_config_module_ablations.py`
  - `run_best_config_sensitivity.py`
  - `run_dygformer_edge_formal.py`
  - `run_edge_only_campaign.py`
- So the latest best-config ablation/sensitivity jobs are currently paused or stopped rather than still actively running.

### Draft Sync Audit

- Checked paper draft files:
  - `ICONIP/main.tex`
  - `ICONIP/tables/*.tex`
- All of those paper files were last updated around `2026-04-27 00:03`.
- The newest experiment artifacts were written later than that, for example:
  - `results/best_config_sensitivity/patent_cn_edge_org_codeIII_transductive__sens_typed_negative_ratio_0p05__s42/metrics.json` at `2026-04-27 04:27`
  - `results/best_config_module_ablation/thgl_software_edge_transductive__full__s42/metrics.json` at `2026-04-27 04:28`
  - `results/best_config_module_ablation/thgl_software_edge_transductive__no_relation_memory__s42/metrics.json` at `2026-04-27 05:08`
- Conclusion:
  - the newest best-config ablation and sensitivity results have **not** been updated into the current ICONIP draft yet.

### Concrete Mismatch Examples

- The draft still keeps the older TGB2 software numbers in `ICONIP/tables/tgb2_thg_main.tex`:
  - transductive DHC: `0.7654/0.9506/0.4340`
  - inductive DHC: `0.8260/0.9532/0.4643`
- But the latest checked software anchor run in `results/edge_only_formal/main/thgl_software_edge_transductive__seed42/metrics.json` is:
  - `MRR/AUC/AP = 0.8009/0.9736/0.5864`
- The newly rerun ablation full-config software transductive result in `results/best_config_module_ablation/thgl_software_edge_transductive__full__s42/metrics.json` is:
  - `MRR/AUC/AP = 0.7941/0.9726/0.5822`
- `results/current_snapshot/main_experiments_table.csv` also does not match the current draft tables one-to-one.
  - Example: snapshot `dblp_edge_transductive` is `0.5269/0.8634/0.2569`
  - But the draft compact/metric-major tables report `dblp` transductive around `0.9699/0.9952/0.8693`
- This means the paper draft, `current_snapshot`, and the newest best-config experiment outputs are currently based on different result sets and need a deliberate reconciliation before the next paper refresh.

### Suggested Next Step

- Finish or resume the missing `best_config_module_ablation` and the remaining `best_config_sensitivity` jobs first.
- Then regenerate the paper tables/figures from the intended authoritative result source before editing `main.tex` wording again.

### Resume Status After Audit

- After the audit, resumed the missing `best_config_module_ablation` jobs on GPU `1`.
- Resume command now running:
  - `python scripts/experiments/run_best_config_module_ablations.py --project-root /home/xuxj/2026/dyn_relation_memory --gpu-ids 1 --max-parallel 1 --poll-sec 20`
- Confirmed runtime behavior:
  - the script skipped the existing `30` completed runs
  - it started with `thgl_software_edge_transductive__no_edge_type__s42`
- The two previously missing runs look like external interruption rather than code failure:
  - `thgl_software_edge_transductive__no_edge_type__s42.log` had stopped at epoch `4`
  - `thgl_software_edge_transductive__no_community__s42.log` had stopped at epoch `2`
  - neither log showed a Python traceback or explicit training exception
- Runtime evidence:
  - live process on GPU `1`: `/home/xuxj/.conda/envs/pytorch/bin/python`
  - current log file: `results/best_config_module_ablation/logs/thgl_software_edge_transductive__no_edge_type__s42.log`
- `best_config_sensitivity` has not been resumed yet in this turn.
  - Reason: GPU `2` and `3` are occupied by other users' long-running jobs, and GPU `1` was used first to finish the smaller missing ablation tail safely.

### Second Recheck

- Rechecked the draft sync status again:
  - `ICONIP/main.tex` and `ICONIP/tables/*.tex` are still timestamped at `2026-04-27 00:03`
  - latest experiment outputs are still newer than the paper files
  - so the paper draft is still **not yet fully updated**
- Rechecked experiment progress:
  - `best_config_module_ablation`: still `30/32` completed on disk at check time, with the remaining two runs actively being recovered on GPU `1`
  - `best_config_sensitivity`: still `26/192` completed on disk at check time before the new resume started
- Rechecked GPU status:
  - GPU `0` had only a small resident process from another user (`~502 MiB`) and `0%` utilization at check time
  - from an available-memory / live-utilization perspective, GPU `0` was usable for continuing our jobs

### GPU0 Resume Action

- Resumed `best_config_sensitivity` on GPU `0`:
  - `python scripts/experiments/run_best_config_sensitivity.py --project-root /home/xuxj/2026/dyn_relation_memory --gpu-ids 0 --max-parallel 1 --poll-sec 20`
- Confirmed runtime behavior:
  - the script skipped the existing `26` completed runs
  - it started with `patent_cn_edge_org_codeIII_transductive__sens_typed_negative_ratio_0p1__s42`
- Concurrent running jobs after this resume:
  - GPU `1`: `best_config_module_ablation`
  - GPU `0`: `best_config_sensitivity`

### 2026-04-27 10:10 STHN Queue

- User requested to run `dyn_relation_memory` main experiment together with the `STHN` baseline rerun under the `pytorch` environment.
- Added launcher script:
  - `scripts/experiments/queue_sthn_main_and_baseline.sh`
- Queue behavior:
  - wait for an idle GPU among `{0,1}` with threshold `memory_used <= 1000 MiB` and `util <= 5%`
  - run `run_edge_only_campaign.py --phase main --no-skip-existing`
  - then run `run_edge_only_campaign.py --phase baseline --families sthn --no-skip-existing`
- Runtime status at launch:
  - queue started successfully
  - current log: `results/sthn_main_and_baseline_queue_20260427.log`
  - first status line after launch: `no idle gpu in {0,1}; sleeping 60s`
- This means the queue is already active in the `pytorch` environment, but the actual main/STHN jobs have not started yet because GPU `0` and `1` are still occupied by the ongoing resumed best-config jobs.

### 2026-04-27 15:36 STHN Fix And Rerun

- Found the failure reason in `scripts/experiments/queue_sthn_main_and_baseline.sh`:
  - the helper `log()` function wrote to stdout
  - `GPU="$(wait_for_idle_gpu)"` therefore captured both the log text and the GPU id
  - this polluted `--gpu-ids` and caused `ValueError: invalid literal for int()`
- Fixed the launcher script:
  - `log()` now writes the line to the log file and stderr instead of stdout
  - idle thresholds were made configurable via `IDLE_MAX_MEM_MIB` and `IDLE_MAX_UTIL`
  - current defaults set in the script: `1500 MiB` and `10%`
- To prioritize the blocked `STHN` rerun itself, launched the dedicated watcher again instead of waiting for the combined main+baseline queue:
  - `python scripts/experiments/watch_and_rerun_sthn_when_gpu_idle.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --poll-seconds 60 --idle-max-memory 1500 --idle-max-util 10 --log-path /home/xuxj/2026/dyn_relation_memory/results/edge_only_formal/baselines/logs/sthn_rerun_waiter_20260427_fix.log`
- Verified status:
  - watcher process is alive
  - log file `results/edge_only_formal/baselines/logs/sthn_rerun_waiter_20260427_fix.log` shows:
    - `watcher started`
    - `no idle gpu yet; sleeping`
- Current conclusion:
  - the bug in the queue script has been repaired
  - the `STHN` rerun is now correctly queued again and will start automatically once a GPU meets the idle threshold

### 2026-04-27 21:44 STHN Progress Check

- Rechecked the live `STHN` rerun status.
- Important update:
  - the dedicated watcher did successfully detect an idle GPU and already launched the full `STHN` rerun campaign
  - launch time recorded in `results/edge_only_formal/baselines/logs/sthn_rerun_waiter_20260427_fix.log`:
    - `2026-04-27 15:52:57`
  - launched command:
    - `python scripts/experiments/run_edge_only_campaign.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --phase baseline --families sthn --gpu-ids 2 --max-parallel 1 --no-skip-existing`
- So `STHN` is no longer in waiting state; it is actively running on GPU `2`.

### Live Progress Snapshot

- Active parent process:
  - `scripts/experiments/run_edge_only_campaign.py --phase baseline --families sthn --gpu-ids 2 --no-skip-existing`
- Active child process at check time:
  - `scripts/baselines/run_tgb2_official_sthn_smoke.py`
  - current task: `patent_us_edge_org_codeIII_inductive`
  - GPU: `2`
- Files updated by the current rerun so far:
  - `sthn__patent_cn_edge_org_codeIII_inductive.csv` at `2026-04-27 16:51`
  - `sthn__patent_cn_edge_org_codeIII_transductive.csv` at `2026-04-27 18:05`
  - `sthn__patent_in_edge_org_codeIII_inductive.csv` at `2026-04-27 19:33`
  - `sthn__patent_in_edge_org_codeIII_transductive.csv` at `2026-04-27 21:03`
  - `sthn__patent_us_edge_org_codeIII_inductive.log` was created at `2026-04-27 21:03`, confirming the next rerun step started

### Completion Status

- `edge_benchmark_common.py` currently defines `16` edge-task candidates for this campaign:
  - `amazon_movies` x2
  - `aminer_paper` x2
  - `dblp` x2
  - `patent_cn` x2
  - `patent_in` x2
  - `patent_us` x2
  - `thgl_github` x2
  - `thgl_software` x2
- Current `STHN` metric CSVs present on disk: `14`
- The two currently missing metric CSVs are:
  - `sthn__thgl_github_edge_inductive.csv`
  - `sthn__thgl_github_edge_transductive.csv`
- Therefore:
  - `STHN` all-task rerun is in progress
  - it has not finished yet
  - at least the current `patent_us` pair plus the two `thgl_github` tasks remain on the critical path, and later tasks may still be overwritten by the same `--no-skip-existing` campaign
