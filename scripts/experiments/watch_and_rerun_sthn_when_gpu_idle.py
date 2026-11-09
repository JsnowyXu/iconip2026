#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def query_gpus() -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = []
    for line in proc.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used": int(parts[1]),
                "utilization": int(parts[2]),
            }
        )
    return rows


def choose_idle_gpu(max_memory_used: int, max_utilization: int) -> int | None:
    for row in query_gpus():
        if row["memory_used"] <= max_memory_used and row["utilization"] <= max_utilization:
            return int(row["index"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True, type=Path)
    ap.add_argument("--python-bin", required=True, type=str)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--idle-max-memory", type=int, default=1000)
    ap.add_argument("--idle-max-util", type=int, default=5)
    ap.add_argument("--log-path", required=True, type=Path)
    args = ap.parse_args()

    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with args.log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    log("watcher started")
    gpu_id = None
    while gpu_id is None:
        try:
            gpu_id = choose_idle_gpu(args.idle_max_memory, args.idle_max_util)
        except Exception as exc:
            log(f"gpu query failed: {exc}")
        if gpu_id is None:
            log("no idle gpu yet; sleeping")
            time.sleep(max(30, int(args.poll_seconds)))

    log(f"idle gpu detected: {gpu_id}")

    cmd = [
        args.python_bin,
        str(args.project_root / "scripts" / "experiments" / "run_edge_only_campaign.py"),
        "--project-root",
        str(args.project_root),
        "--python-bin",
        args.python_bin,
        "--phase",
        "baseline",
        "--families",
        "sthn",
        "--gpu-ids",
        str(gpu_id),
        "--max-parallel",
        "1",
        "--no-skip-existing",
    ]
    log(f"launching rerun: {' '.join(cmd)}")
    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(args.project_root), env=env)
    log(f"rerun finished with code={proc.returncode}")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
