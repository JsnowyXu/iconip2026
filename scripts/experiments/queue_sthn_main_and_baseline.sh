#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/xuxj/2026/dyn_relation_memory"
PY="/home/xuxj/.conda/envs/pytorch/bin/python"
LOG="$PROJECT/results/sthn_main_and_baseline_queue_20260427.log"
IDLE_MAX_MEM_MIB="${IDLE_MAX_MEM_MIB:-1500}"
IDLE_MAX_UTIL="${IDLE_MAX_UTIL:-10}"

mkdir -p "$PROJECT/results"
cd "$PROJECT"

log() {
  local line
  line="$(printf '[%s] %s' "$(date '+%F %T')" "$1")"
  printf '%s\n' "$line" >> "$LOG"
  printf '%s\n' "$line" >&2
}

wait_for_idle_gpu() {
  while true; do
    local gpu
    gpu="$("$PY" - <<'PYCODE'
import os
import subprocess

allowed = {0, 1}
max_mem = int(os.environ.get("IDLE_MAX_MEM_MIB", "1500"))
max_util = int(os.environ.get("IDLE_MAX_UTIL", "10"))
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
for line in out.strip().splitlines():
    idx_s, mem_s, util_s = [x.strip() for x in line.split(",")]
    idx = int(idx_s)
    mem = int(mem_s)
    util = int(util_s)
    if idx in allowed and mem <= max_mem and util <= max_util:
        print(idx)
        break
PYCODE
)"
    if [[ -n "$gpu" ]]; then
      log "idle gpu detected: $gpu"
      printf '%s\n' "$gpu"
      return 0
    fi
    log "no idle gpu in {0,1} under mem<=${IDLE_MAX_MEM_MIB}MiB util<=${IDLE_MAX_UTIL}%; sleeping 60s"
    sleep 60
  done
}

log "queue started"
GPU="$(wait_for_idle_gpu)"
log "launching main phase on gpu $GPU"
"$PY" scripts/experiments/run_edge_only_campaign.py \
  --project-root "$PROJECT" \
  --python-bin "$PY" \
  --phase main \
  --gpu-ids "$GPU" \
  --max-parallel 1 \
  --no-skip-existing 2>&1 | tee -a "$LOG"
log "main phase finished"

GPU="$(wait_for_idle_gpu)"
log "launching baseline sthn phase on gpu $GPU"
"$PY" scripts/experiments/run_edge_only_campaign.py \
  --project-root "$PROJECT" \
  --python-bin "$PY" \
  --phase baseline \
  --families sthn \
  --gpu-ids "$GPU" \
  --max-parallel 1 \
  --no-skip-existing 2>&1 | tee -a "$LOG"
log "baseline sthn phase finished"
