# aminer_paper Process Log

## Scope
- Project root: `/home/xuxj/2026/dyn_relation_memory`
- Dataset root: `/home/xuxj/2026/datasets`
- Baseline root: `/home/xuxj/2026/baselines/src`
- Date: `2026-04-15`
- Python env for this run: `/home/xuxj/.conda/envs/pytorch/bin/python`

## GPU Snapshot Before Running
- Command: `nvidia-smi`
- Preferred GPU for this round: `GPU 1`
- Reason: `GPU 1` was nearly idle at the start of this run.
- Allocation used:
  - `GPU 1`: DHC main experiments
  - `GPU 0`: baseline experiments
  - `GPU 2/3`: skipped because already busy

## Execution Log

### 2026-04-15 01:01 CST
- Checked repository layout, current processed data, and `aminer_paper` script entry points.
- Confirmed source archives exist:
  - `/home/xuxj/2026/datasets/Aminer/AMiner-Paper.rar`
  - `/home/xuxj/2026/datasets/Aminer/AMiner-Author.zip`
  - `/home/xuxj/2026/datasets/Aminer/AMiner-Author2Paper.zip`
  - `/home/xuxj/2026/datasets/Aminer/AMiner-Coauthor.zip`
- Confirmed existing processed file:
  - `/home/xuxj/2026/dyn_relation_memory/data/processed/aminer_paper_unified.pt`
- Located relevant scripts:
  - `scripts/datasets/prepare_dataset.py`
  - `scripts/datasets/analyze_datasets.py`
  - `scripts/datasets/build_paper_task_views.py`
  - `scripts/baselines/run_all_baselines_formal.py`
  - `scripts/experiments/run_full_benchmark_repeats.py`

### 2026-04-15 01:03-01:06 CST
- Re-ran preprocessing command:
  - `/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/prepare_dataset.py --dataset aminer_paper --archive /home/xuxj/2026/datasets/Aminer/AMiner-Paper.rar --out-pt /home/xuxj/2026/dyn_relation_memory/data/processed/aminer_paper_unified.pt --max-edges 200000 --aminer-min-year 2002 --aminer-max-year 2012 --export-baselines --baseline-out-dir /home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/aminer_paper`
- Output summary:
  - `saved unified dataset to /home/xuxj/2026/dyn_relation_memory/data/processed/aminer_paper_unified.pt`
  - `nodes=265234 edges=961282 rels=4`
  - `baseline adapters exported to /home/xuxj/2026/dyn_relation_memory/data/interim/baseline_adapters/aminer_paper`
- Runtime summary:
  - wall time: `3:04.51`
  - user CPU time: `170.75s`
  - peak RSS: `4,889,756 KB`

### 2026-04-15 01:07 CST
- Extracted unified-graph statistics from `data/processed/aminer_paper_unified.pt`.
- Confirmed relation counts:
  - `paper_cite_paper`: `200000`
  - `author_write_paper`: `49776`
  - `paper_publish_venue`: `17970`
  - `author_coauthor_author`: `693536`
- Confirmed node type counts:
  - `paper`: `127349`
  - `author`: `137450`
  - `venue`: `435`

### 2026-04-15 01:08 CST
- Built AMiner paper-task views for both target relations:
  - citation:
    - `/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/build_paper_task_views.py --input data/processed/aminer_paper_unified.pt --out-dir data/processed/paper_tasks --name-prefix aminer_paper_citation --target-rel 0`
  - coauthor:
    - `/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/build_paper_task_views.py --input data/processed/aminer_paper_unified.pt --out-dir data/processed/paper_tasks --name-prefix aminer_paper_coauthor --target-rel 3`
- Generated files:
  - `data/processed/paper_tasks/aminer_paper_citation_edge_transductive.pt`
  - `data/processed/paper_tasks/aminer_paper_citation_edge_inductive.pt`
  - `data/processed/paper_tasks/aminer_paper_coauthor_edge_transductive.pt`
  - `data/processed/paper_tasks/aminer_paper_coauthor_edge_inductive.pt`
  - `data/processed/paper_tasks/aminer_paper_node_multilabel.pt`
- The generic node task file was normalized from the citation-generated node view so that the metadata name is `aminer_paper_node_multilabel`.

### 2026-04-15 01:09 CST
- Ran dataset analysis command on unified + task views:
  - `/home/xuxj/.conda/envs/pytorch/bin/python scripts/datasets/analyze_datasets.py --inputs data/processed/aminer_paper_unified.pt data/processed/paper_tasks/aminer_paper_citation_edge_transductive.pt data/processed/paper_tasks/aminer_paper_citation_edge_inductive.pt data/processed/paper_tasks/aminer_paper_coauthor_edge_transductive.pt data/processed/paper_tasks/aminer_paper_coauthor_edge_inductive.pt data/processed/paper_tasks/aminer_paper_node_multilabel.pt --out-dir data/reports`
- Summary file generated:
  - `data/reports/dataset_stats_summary.csv`
- Figure directories generated:
  - `data/reports/aminer_paper`
  - `data/reports/aminer_paper_citation_edge_transductive`
  - `data/reports/aminer_paper_citation_edge_inductive`
  - `data/reports/aminer_paper_coauthor_edge_transductive`
  - `data/reports/aminer_paper_coauthor_edge_inductive`
  - `data/reports/aminer_paper_node_multilabel`

### 2026-04-15 01:1x CST
- Added AMiner-specific schedulers for reproducible reruns:
  - `scripts/experiments/run_aminer_dhc_suite.py`
  - `scripts/baselines/run_aminer_baselines_formal.py`
- Syntax checked with:
  - `/home/xuxj/.conda/envs/pytorch/bin/python -m py_compile scripts/experiments/run_aminer_dhc_suite.py`
  - `/home/xuxj/.conda/envs/pytorch/bin/python -m py_compile scripts/baselines/run_aminer_baselines_formal.py`

### 2026-04-15 01:1x CST
- Started DHC suite:
  - `/home/xuxj/.conda/envs/pytorch/bin/python scripts/experiments/run_aminer_dhc_suite.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 1 --seeds 42 --epochs 20 --batch-size 512 --lr 1e-3 --early-stop-patience 5 --max-parallel 1 --poll-sec 20 --save-memory-artifacts`
- Output root:
  - `results/aminer_dhc_formal`
- The first launched run is:
  - `aminer_paper_citation_edge_transductive__seed42`

### 2026-04-15 01:12 CST
- First DHC launch failed immediately for all queued tasks.
- Root cause:
  - the scheduler called `dhc/train/run_train.py` as a script without exposing the project root as an importable package root;
  - logs showed `ModuleNotFoundError: No module named 'dhc'`.
- Fix applied:
  - changed the scheduler to launch `python -m dhc.train.run_train`
  - injected `PYTHONPATH=/home/xuxj/2026/dyn_relation_memory`
- Re-run command:
  - `/home/xuxj/.conda/envs/pytorch/bin/python scripts/experiments/run_aminer_dhc_suite.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 1 --seeds 42 --epochs 20 --batch-size 512 --lr 1e-3 --early-stop-patience 5 --max-parallel 1 --poll-sec 20 --save-memory-artifacts --no-skip-existing`
- Current state after the fix:
  - `results/aminer_dhc_formal/aminer_paper_citation_edge_transductive__seed42/best_model.pt` has already appeared, so the first DHC task is training rather than failing at import time.

### 2026-04-15 01:1x CST
- Started baseline suite:
  - `/home/xuxj/.conda/envs/pytorch/bin/python scripts/baselines/run_aminer_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 0 --max-parallel 1 --epochs 10 --batch-size 256 --dyglib-max-edges 50000 --eval-num-negs 20 --poll-sec 20`
- Output root:
  - `results/baseline_formal_aminer_custom`
- Current state:
  - the first baseline job is `tgn__aminer_paper_citation_edge_transductive`
  - the scheduler has already exported per-task adapters under `data/interim/baseline_adapters/aminer_paper_*`

## Next Checkpoints
- Wait for DHC and baseline manifests plus summary CSVs.
- Build metric-major comparison tables after the first batch of runs finishes.
- Generate memory/community visualizations from saved `memory_trace.pt` and `memory_final_state.pt`.
- Write the DHC representation-usage check into the final dataset/experiment notes.

### 2026-04-15 baseline rerun diagnosis
- Re-checked `results/baseline_formal_aminer_custom/metrics` and confirmed AMiner baselines were not fully completed.
- Successful before rerun:
  - `tgn__*` all 4 edge tasks
  - `tgn_edges__*` all 4 edge tasks
  - `dyglib_custom__aminer_paper_citation_edge_transductive.csv`
  - `dyglib_custom__aminer_paper_citation_edge_inductive.csv`
  - `dyglib_custom__aminer_paper_coauthor_edge_transductive.csv`
- Failed before rerun:
  - `dyglib_model_zoo__aminer_paper_citation_edge_transductive.csv`
  - `dyglib_model_zoo__aminer_paper_citation_edge_inductive.csv`
  - `dyglib_model_zoo__aminer_paper_coauthor_edge_transductive.csv`
  - all existing `sthn__*.log` runs failed and produced no result CSV
- Missing before rerun:
  - `dyglib_custom__aminer_paper_coauthor_edge_inductive.csv`
  - `dyglib_model_zoo__aminer_paper_coauthor_edge_inductive.csv`
  - `dyglib_model_zoo__aminer_paper_node_multilabel.csv`
  - `sthn__aminer_paper_citation_edge_transductive.csv`
  - `sthn__aminer_paper_citation_edge_inductive.csv`
  - `sthn__aminer_paper_coauthor_edge_transductive.csv`
  - `sthn__aminer_paper_coauthor_edge_inductive.csv`

### Failure causes identified
- `dyglib_model_zoo`
  - existing logs showed CUDA BCE assertion failure:
    - `Assertion input_val >= zero && input_val <= one failed`
  - root cause at the AMiner wrapper layer:
    - the smoke script used `BCELoss` directly on probability outputs that could become `NaN/Inf`, causing device-side assertions
- `STHN`
  - existing logs failed in official TGB2 code path:
    - `ValueError: Found array with 0 sample(s) ... MinMaxScaler`
  - root cause:
    - `modules/sthn.py` tried to fit `MinMaxScaler` on empty `subgraph_edts`

### Fixes applied before rerun
- Patched `scripts/baselines/run_dyglib_model_zoo_smoke.py`
  - switched training loss to `BCEWithLogitsLoss`
  - added `sanitize_logits(...)` and safe probability conversion to guard `NaN/Inf`
- Patched official baseline file:
  - `/home/xuxj/2026/baselines/src/TGB2-main/TGB2-main/modules/sthn.py`
  - added empty-`subgraph_edts` guard before `MinMaxScaler.fit(...)`
- Patched `scripts/baselines/run_aminer_baselines_formal.py`
  - added `--families`
  - added `--skip-successful` logic
  - scheduler now reruns only failed/missing outputs instead of rerunning successful baselines

### Validation before full rerun
- `dyglib_model_zoo` smoke on `JODIE` for `aminer_paper_citation_edge_transductive` completed successfully and wrote:
  - `results/baseline_formal_aminer_custom/metrics/_smoke_dyglib_jodie.csv`

### Full rerun launched
- Command:
  - `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_aminer_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 1,2 --max-parallel 2 --epochs 10 --batch-size 256 --dyglib-max-edges 50000 --eval-num-negs 20 --poll-sec 20 --families dyglib_custom,dyglib_model_zoo,sthn`
- Scheduler output at launch:
  - total jobs in suite: `21`
  - filtered rerun jobs: `10`
- First confirmed healthy rerun signal:
  - `dyglib_model_zoo__aminer_paper_citation_edge_transductive` reached `JODIE epoch=1` with valid metrics instead of immediate CUDA failure

### 2026-04-15 15:3x CST RepeatMixer baseline integration
- New uploaded baseline source:
  - `/home/xuxj/2026/baselines/src/RepeatMixer-main`
- Compatibility issue found:
  - `models/RepeatMixer.py` imported `audtorch.metrics.functional.pearsonr`
  - the current `pytorch` environment does not have `audtorch`
- Fix applied:
  - patched `/home/xuxj/2026/baselines/src/RepeatMixer-main/models/RepeatMixer.py`
  - added a local torch-based `pearsonr(...)` fallback used only when `audtorch` is unavailable
- New AMiner wrapper added:
  - `scripts/baselines/run_repeatmixer_smoke.py`
  - wrapper reads our AMiner `.pt` task views directly
  - pads node and edge features to a common dimension compatible with RepeatMixer
  - uses the task-provided `train/val/test` split
  - writes the same split-level metric CSV schema already used by the other custom baselines
- Scheduler integration:
  - patched `scripts/baselines/run_aminer_baselines_formal.py`
  - added a new baseline family: `repeatmixer`
  - outputs will be written as:
    - `results/baseline_formal_aminer_custom/metrics/repeatmixer__aminer_paper_citation_edge_transductive.csv`
    - `results/baseline_formal_aminer_custom/metrics/repeatmixer__aminer_paper_citation_edge_inductive.csv`
    - `results/baseline_formal_aminer_custom/metrics/repeatmixer__aminer_paper_coauthor_edge_transductive.csv`
    - `results/baseline_formal_aminer_custom/metrics/repeatmixer__aminer_paper_coauthor_edge_inductive.csv`

### RepeatMixer smoke validation
- Command:
  - `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_repeatmixer_smoke.py --data /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/aminer_paper_citation_edge_transductive.pt --epochs 1 --batch-size 128 --gpu-id 2 --eval-num-negs 10 --out /home/xuxj/2026/dyn_relation_memory/results/baseline_formal_aminer_custom/metrics/_smoke_repeatmixer.csv`
- Result file:
  - `results/baseline_formal_aminer_custom/metrics/_smoke_repeatmixer.csv`
- Smoke metrics:
  - val: `MRR=0.500175`, `AUC=0.832405`, `AP=0.288900`
  - test: `MRR=0.650682`, `AUC=0.870344`, `AP=0.450367`
  - params: `1,292,861`
  - avg epoch sec: `188.92`
- Conclusion:
  - RepeatMixer is compatible with the current AMiner task view and can train/evaluate successfully on GPU

### RepeatMixer formal run launched
- Command:
  - `/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_aminer_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 2 --max-parallel 1 --epochs 10 --batch-size 256 --dyglib-max-edges 50000 --eval-num-negs 20 --poll-sec 20 --families repeatmixer`
- Scheduler state at launch:
  - total jobs in suite definition: `25`
  - filtered RepeatMixer jobs: `4`
  - first launched job: `repeatmixer__aminer_paper_citation_edge_transductive` on `GPU 2`

### Current baseline progress snapshot
- `dyglib_model_zoo__aminer_paper_citation_edge_transductive`
  - currently far past the former instant-failure point
  - confirmed progress includes:
    - `JODIE` finished and wrote valid test metrics in the log
    - `DyRep` finished and wrote valid test metrics in the log
    - `TGAT` finished and wrote valid test metrics in the log
    - `CAWN` is still running, already reached at least `epoch=5`
- `sthn__aminer_paper_citation_edge_transductive`
  - still running in the rerun batch
  - no final metric CSV yet at this checkpoint
- `repeatmixer__aminer_paper_citation_edge_transductive`
  - newly started as a separate formal batch on `GPU 2`

### Important status note
- As of this checkpoint, the AMiner baseline suite is still in progress.
- Existing failed `dyglib_model_zoo__*.csv` files in `results/baseline_formal_aminer_custom/metrics` are stale old outputs until the current rerun finishes and rewrites them.

### 2026-04-15 22:05 CST rerun checkpoint
- Re-checked all AMiner experiment artifacts.
- `aminer_dhc_formal` is complete:
  - `run_manifest.csv`: 5/5 finished with `return_code=0`
  - `summary.csv`: present
- `baseline_formal_aminer_custom` still had stale or failed baseline artifacts:
  - `dyglib_model_zoo__aminer_paper_citation_edge_inductive.csv`: all rows `failed:RuntimeError`

### 2026-04-16 17:18 CST runtime checkpoint
- Re-checked active AMiner experiment processes.
- `aminer_dhc_formal` remains complete and unchanged.
- `baseline_formal_aminer_custom` is still actively running its final node multilabel rerun:
  - process PID: `1491012`
  - command:
`/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_dyglib_model_zoo_smoke.py --data /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/aminer_paper_node_multilabel.pt --models JODIE,DyRep,TGAT,CAWN,TCL,GraphMixer,DyGFormer --epochs 10 --batch-size 256 --max-edges 50000 --gpu-id 0 --eval-num-negs 20 --out /home/xuxj/2026/dyn_relation_memory/results/baseline_formal_aminer_custom/metrics/dyglib_model_zoo__aminer_paper_node_multilabel.csv`
- Latest confirmed model progress from the log:
  - `JODIE`, `DyRep`, and `TGAT` finished and wrote test metrics
  - `CAWN` is still running and has already reached `epoch=8`
- No new failure signature was observed in this checkpoint; the rerun is slow but healthy.
  - `dyglib_model_zoo__aminer_paper_coauthor_edge_transductive.csv`: all rows `failed:RuntimeError`
  - `repeatmixer__aminer_paper_coauthor_edge_inductive.csv`: training log exists but final metric CSV missing
  - `dyglib_custom__aminer_paper_coauthor_edge_inductive.csv`: missing
  - `dyglib_model_zoo__aminer_paper_coauthor_edge_inductive.csv`: missing
  - `dyglib_model_zoo__aminer_paper_node_multilabel.csv`: missing
  - all `STHN` tasks remained stale or broken
- Old hanging AMiner baseline scheduler and stale `STHN` subprocesses were terminated.
- Filtered rerun launched for the non-`STHN` failed or missing AMiner baselines:
`CUDA_VISIBLE_DEVICES=0 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_aminer_baselines_formal.py --project-root /home/xuxj/2026/dyn_relation_memory --python-bin /home/xuxj/.conda/envs/pytorch/bin/python --gpu-ids 0 --max-parallel 1 --epochs 10 --batch-size 256 --dyglib-max-edges 50000 --eval-num-negs 20 --poll-sec 20 --families dyglib_custom,dyglib_model_zoo,repeatmixer`
- New rerun progress confirmed:
  - scheduler process is active
  - `dyglib_model_zoo__aminer_paper_citation_edge_inductive.log` has restarted from `JODIE`

### 2026-04-15 22:10 CST progress recheck
- Active rerun scheduler:
  - PID `264180`
- Active child worker:
  - PID `265332`
  - command:
`/home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_dyglib_model_zoo_smoke.py --data /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/aminer_paper_citation_edge_inductive.pt --models JODIE,DyRep,TGAT,CAWN,TCL,GraphMixer,DyGFormer --epochs 10 --batch-size 256 --max-edges 50000 --gpu-id 0 --eval-num-negs 20 --out /home/xuxj/2026/dyn_relation_memory/results/baseline_formal_aminer_custom/metrics/dyglib_model_zoo__aminer_paper_citation_edge_inductive.csv`
- Runtime check:
  - log tail is still advancing and currently shows `JODIE` through `epoch=4`
  - no evidence of hang or crash in the current rerun
- Important interpretation:
  - existing stale `failed:RuntimeError` CSVs in `results/baseline_formal_aminer_custom/metrics` should still be treated as old outputs until the active rerun rewrites them
- Pending after this checkpoint:
  - finish the active non-`STHN` rerun batch
  - re-evaluate whether `STHN` should be relaunched separately after the current batch settles

### 2026-04-15 22:12 CST STHN verification
- Rechecked the patched official file:
  - `/home/xuxj/2026/baselines/src/TGB2-main/TGB2-main/modules/sthn.py`
  - empty `subgraph_edts` protection is still present before `MinMaxScaler.fit(...)`
- New isolated smoke rerun launched to verify the official `STHN` path on our AMiner task view:
`CUDA_VISIBLE_DEVICES=0 /home/xuxj/.conda/envs/pytorch/bin/python /home/xuxj/2026/dyn_relation_memory/scripts/baselines/run_tgb2_official_sthn_smoke.py --data /home/xuxj/2026/dyn_relation_memory/data/processed/paper_tasks/aminer_paper_citation_edge_transductive.pt --epochs 1 --batch-size 128 --gpu-id 0 --dataset-name thgl-github --out /home/xuxj/2026/dyn_relation_memory/results/baseline_formal_aminer_custom/metrics/_smoke_sthn_aminer_citation_transductive.csv`
- Live process check:
  - parent PID `274762`
  - currently inside dataset preparation / official runner setup
- Purpose:
  - if this isolated smoke succeeds, `STHN` can be relaunched as a separate formal rerun family without blocking the other baseline schedulers

### 2026-04-16 00:05 CST running-status note
- AMiner non-`STHN` rerun scheduler is still active:
  - PID `264180`
- Current active child remains a `dyglib_model_zoo` rerun on `GPU 0`.
- Patent node-multilabel repairs were launched on `GPU 1` and `GPU 2` in parallel; AMiner reruns were left untouched and continue in the background.
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 00:43:25
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 07:42:24
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 09:52:08
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 10:06:58
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 11:03:58
- launch `aminer_paper_citation_edge_inductive__seed42` on GPU `0`
## 2026-04-20 11:51:45
- finished `aminer_paper_citation_edge_inductive__seed42` rc=`0` elapsed_sec=`2867.4`
## 2026-04-20 11:51:45
- launch `aminer_paper_citation_edge_transductive__seed42` on GPU `0`
## 2026-04-20 12:59:04
- finished `aminer_paper_citation_edge_transductive__seed42` rc=`0` elapsed_sec=`4038.4`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 22:50:39
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 22:51:12
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 22:52:00
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-20 23:19:38
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 10:30:58
- launch `sthn__aminer_paper_citation_edge_inductive` on GPU `2`
## 2026-04-21 11:01:38
- launch `repeatmixer__aminer_paper_citation_edge_inductive` on GPU `1`
## 2026-04-21 11:43:32
- finished `sthn__aminer_paper_citation_edge_inductive` rc=`0` elapsed_sec=`4353.8`
## 2026-04-21 11:43:32
- launch `sthn__aminer_paper_citation_edge_transductive` on GPU `2`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 11:59:51
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 12:10:41
- finished `repeatmixer__aminer_paper_citation_edge_inductive` rc=`0` elapsed_sec=`4143.6`
## 2026-04-21 12:10:41
- launch `repeatmixer__aminer_paper_citation_edge_transductive` on GPU `1`
## 2026-04-21 12:52:51
- finished `sthn__aminer_paper_citation_edge_transductive` rc=`0` elapsed_sec=`4158.6`
## 2026-04-21 12:57:14
- finished `repeatmixer__aminer_paper_citation_edge_transductive` rc=`0` elapsed_sec=`2792.5`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 18:00:22
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 18:00:23
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 22:40:20
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 22:47:55
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 23:12:03
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 23:12:03
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 23:14:51
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 23:14:51
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-21 23:14:51
- launch `aminer_paper_citation_edge_inductive__seed42` on GPU `2`
## 2026-04-21 23:14:51
- launch `aminer_paper_citation_edge_transductive__seed42` on GPU `3`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 09:47:20
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 14:28:06
- launch `aminer_paper_citation_edge_inductive__seed42` on GPU `0`
## 2026-04-22 15:25:24
- finished `aminer_paper_citation_edge_inductive__seed42` rc=`0` elapsed_sec=`3438.2`
## 2026-04-22 15:25:24
- launch `aminer_paper_citation_edge_transductive__seed42` on GPU `2`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 23:13:15
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 23:18:51
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-22 23:34:22
- launch `tgn_edgetype__aminer_paper_citation_edge_inductive` on GPU `2`
## 2026-04-22 23:36:22
- launch `tgn_edgetype__aminer_paper_citation_edge_transductive` on GPU `3`
## 2026-04-23 00:31:55
- finished `tgn_edgetype__aminer_paper_citation_edge_transductive` rc=`0` elapsed_sec=`3333.0`
## 2026-04-23 00:33:40
- finished `tgn_edgetype__aminer_paper_citation_edge_inductive` rc=`0` elapsed_sec=`3558.3`
## 2026-04-23 10:43:26
- launch targeted tuning `aminer_paper_citation_edge_inductive__rank_norel_lr5e4__s42` on GPU `3`
- variant: `rank_norel_lr5e4`
## 2026-04-23 10:52:57
- launch targeted tuning `aminer_paper_citation_edge_inductive__rank_norel_topk8__s42` on GPU `1`
- variant: `rank_norel_topk8`
## 2026-04-23 10:57:42
- launch targeted tuning `aminer_paper_citation_edge_transductive__rank_norel_lr5e4__s42` on GPU `2`
- variant: `rank_norel_lr5e4`
## 2026-04-23 11:50:15
- finished targeted tuning `aminer_paper_citation_edge_inductive__rank_norel_lr5e4__s42` rc=`0` elapsed_sec=`4008.7`
## 2026-04-23 11:50:15
- launch targeted tuning `aminer_paper_citation_edge_transductive__rank_norel_topk8__s42` on GPU `3`
- variant: `rank_norel_topk8`
## 2026-04-23 12:04:16
- finished targeted tuning `aminer_paper_citation_edge_inductive__rank_norel_topk8__s42` rc=`0` elapsed_sec=`4278.9`
## 2026-04-23 12:37:17
- finished targeted tuning `aminer_paper_citation_edge_transductive__rank_norel_lr5e4__s42` rc=`0` elapsed_sec=`5975.4`
## 2026-04-23 13:32:05
- finished targeted tuning `aminer_paper_citation_edge_transductive__rank_norel_topk8__s42` rc=`0` elapsed_sec=`6110.6`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-23 23:47:39
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-24 08:33:53
- launch `tgn__aminer_paper_citation_edge_inductive` on GPU `0`
## 2026-04-24 09:24:11
- finished `tgn__aminer_paper_citation_edge_inductive` rc=`0` elapsed_sec=`3017.9`
## 2026-04-24 09:24:11
- launch `tgn__aminer_paper_citation_edge_transductive` on GPU `0`
## 2026-04-24 10:16:44
- finished `tgn__aminer_paper_citation_edge_transductive` rc=`0` elapsed_sec=`3152.8`
## 2026-04-24 10:44:46
- launch `edgebank__aminer_paper_citation_edge_inductive` on GPU `0`
## 2026-04-24 10:45:01
- finished `edgebank__aminer_paper_citation_edge_inductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-24 10:45:01
- launch `edgebank__aminer_paper_citation_edge_transductive` on GPU `0`
## 2026-04-24 10:45:16
- finished `edgebank__aminer_paper_citation_edge_transductive` rc=`1` elapsed_sec=`15.0`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-26 19:41:06
- scheduled DyGFormer edge baseline
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-04-27 16:01:32
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-05-08 20:30:17
- TAMI baseline edge runs completed for `aminer_paper_citation_edge_inductive` and `aminer_paper_citation_edge_transductive`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/aminer_paper_citation_edge_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_edge_20260508/aminer_paper_citation_edge_transductive.csv`
- Test metrics: inductive MRR `0.705282`, AUC `0.943122`, AP `0.388429`; transductive MRR `0.796995`, AUC `0.965180`, AP `0.526093`.

## 2026-05-08 22:12:13
- TAMI formal baseline rerun completed with `epochs=10`, `max_edges=50000`, `batch_size=256`, `eval_num_negs=20`, `early_stop_patience=5`.
- Output CSVs:
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/aminer_paper_citation_edge_inductive.csv`
  - `/home/xuxj/2026/dyn_relation_memory/results/tami_baseline_formal_20260508/aminer_paper_citation_edge_transductive.csv`
- Test metrics: inductive MRR `0.602626`, AUC `0.865511`, AP `0.219894`; transductive MRR `0.829720`, AUC `0.962749`, AP `0.708484`.
## 2026-05-20 17:42:05
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-05-20 17:42:05
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-05-20 17:43:05
- launch `edgebank__aminer_paper_citation_edge_inductive` on GPU `0`
## 2026-05-20 17:43:20
- finished `edgebank__aminer_paper_citation_edge_inductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 17:43:20
- launch `edgebank__aminer_paper_citation_edge_transductive` on GPU `0`
## 2026-05-20 17:43:35
- finished `edgebank__aminer_paper_citation_edge_transductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 17:46:35
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-05-20 17:46:35
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-05-20 18:01:36
- launch `tabular_ml__aminer_paper_citation_edge_inductive` on GPU `0`
## 2026-05-20 18:04:21
- finished `tabular_ml__aminer_paper_citation_edge_inductive` rc=`0` elapsed_sec=`165.2`
## 2026-05-20 18:04:21
- launch `tabular_ml__aminer_paper_citation_edge_transductive` on GPU `0`
## 2026-05-20 18:07:21
- finished `tabular_ml__aminer_paper_citation_edge_transductive` rc=`0` elapsed_sec=`180.2`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-05-20 21:49:21
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-05-20 22:12:38
- launch `tabular_ml__aminer_paper_citation_edge_inductive` on GPU `0`
## 2026-05-20 22:14:23
- finished `tabular_ml__aminer_paper_citation_edge_inductive` rc=`0` elapsed_sec=`105.1`
## 2026-05-20 22:14:23
- launch `tabular_ml__aminer_paper_citation_edge_transductive` on GPU `0`
## 2026-05-20 22:16:08
- finished `tabular_ml__aminer_paper_citation_edge_transductive` rc=`0` elapsed_sec=`105.1`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_inductive`
- relation_target: `paper_cite_paper`
## 2026-05-20 22:31:25
- scheduled edge-only benchmark tasks
- task: `aminer_paper_citation_edge_transductive`
- relation_target: `paper_cite_paper`
## 2026-05-20 22:33:26
- launch `edgebank__aminer_paper_citation_edge_inductive` on GPU `0`
## 2026-05-20 22:33:41
- finished `edgebank__aminer_paper_citation_edge_inductive` rc=`0` elapsed_sec=`15.0`
## 2026-05-20 22:33:41
- launch `edgebank__aminer_paper_citation_edge_transductive` on GPU `0`
## 2026-05-20 22:33:56
- finished `edgebank__aminer_paper_citation_edge_transductive` rc=`0` elapsed_sec=`15.0`
