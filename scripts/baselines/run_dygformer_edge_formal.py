#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "scripts" / "experiments"
for _path in [PROJECT_ROOT, EXPERIMENTS_ROOT]:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from edge_benchmark_common import DATASET_ORDER, TaskSpec, append_dataset_log, append_master_log, build_edge_task_specs, ensure_thgl_tasks


def tail_last_line(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return ""
            off = min(size, 4096)
            f.seek(-off, os.SEEK_END)
            txt = f.read().decode("utf-8", errors="ignore")
        lines = [x.strip() for x in txt.splitlines() if x.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""


def dygformer_success(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty:
        return False
    if "model" not in df.columns or "status" not in df.columns:
        return False
    sub = df[df["model"].astype(str) == "DyGFormer"].copy()
    if sub.empty:
        return False
    return any(str(x).strip().lower() == "ok" for x in sub["status"].tolist())


def parse_dygformer_csv(path: Path, task: TaskSpec) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    df = df[df["model"].astype(str) == "DyGFormer"].copy()
    if df.empty:
        return []
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": task.dataset,
                "task_name": task.task_name,
                "task_type": task.task_type,
                "relation_target": task.relation_target,
                "family": "dyglib_model_zoo",
                "model": "DyGFormer",
                "status": r.get("status", "unknown"),
                "val_mrr": r.get("val_mrr"),
                "val_auc": r.get("val_auc"),
                "val_ap": r.get("val_ap"),
                "test_mrr": r.get("test_mrr"),
                "test_auc": r.get("test_auc"),
                "test_ap": r.get("test_ap"),
                "avg_epoch_sec": r.get("avg_epoch_sec"),
                "total_sec": r.get("total_sec"),
                "total_params": r.get("total_params"),
                "trainable_params": r.get("trainable_params"),
                "device": r.get("device"),
                "source_path": str(path),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser("Run DyGFormer on all dyn_relation_memory edge tasks")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument("--repo-root", type=Path, default=Path("/home/xuxj/2026"))
    ap.add_argument("--python-bin", type=str, default="/home/xuxj/.conda/envs/pytorch/bin/python")
    ap.add_argument("--datasets", type=str, default="")
    ap.add_argument("--gpu-ids", type=str, default="3")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-edges", type=int, default=0)
    ap.add_argument("--eval-num-negs", type=int, default=20)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--skip-successful", action="store_true", default=True)
    ap.add_argument("--no-skip-successful", dest="skip_successful", action="store_false")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/xuxj/2026/dyn_relation_memory/results/dygformer_edge_formal"),
    )
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    repo_root = args.repo_root.resolve()
    out_root = args.out_root.resolve()
    metric_root = out_root / "metrics"
    log_root = out_root / "logs"
    metric_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = [3]

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(project_root) + (os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else "")
    base_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["MKL_THREADING_LAYER"] = "GNU"
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    ensure_thgl_tasks(project_root, repo_root, args.python_bin, base_env)
    task_specs = build_edge_task_specs(project_root)
    selected_datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    if selected_datasets:
        task_specs = [spec for spec in task_specs if spec.dataset in selected_datasets]
    task_specs = sorted(
        task_specs,
        key=lambda spec: (
            DATASET_ORDER.index(spec.dataset) if spec.dataset in DATASET_ORDER else 99,
            spec.task_type,
            spec.task_name,
        ),
    )

    jobs: List[Dict[str, object]] = []
    for idx, task in enumerate(task_specs):
        out_path = metric_root / f"dygformer__{task.task_name}.csv"
        jobs.append(
            {
                "name": f"dygformer__{task.task_name}",
                "task": task,
                "out_path": out_path,
                "log_path": log_root / f"dygformer__{task.task_name}.log",
                "gpu": gpu_ids[idx % len(gpu_ids)],
                "cmd": [
                    args.python_bin,
                    str(project_root / "scripts" / "baselines" / "run_dyglib_model_zoo_smoke.py"),
                    "--data",
                    str(task.data_path),
                    "--models",
                    "DyGFormer",
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--max-edges",
                    str(args.max_edges),
                    "--gpu-id",
                    "{gpu}",
                    "--eval-num-negs",
                    str(args.eval_num_negs),
                    "--out",
                    str(out_path),
                ],
            }
        )

    append_master_log(
        project_root,
        [
            "- launch DyGFormer-only edge baseline campaign",
            f"- out_root: `{out_root}`",
            f"- datasets: `{[task.dataset for task in task_specs]}`",
            f"- gpu_ids: `{gpu_ids}`",
        ],
    )
    for task in task_specs:
        append_dataset_log(
            project_root,
            task.dataset,
            [
                "- scheduled DyGFormer edge baseline",
                f"- task: `{task.task_name}`",
                f"- relation_target: `{task.relation_target}`",
            ],
        )

    queue = list(jobs)
    running: List[Dict[str, object]] = []
    finished: List[Dict[str, object]] = []

    while queue or running:
        while queue and len(running) < max(1, int(args.max_parallel)):
            job = queue.pop(0)
            if args.skip_successful and dygformer_success(Path(job["out_path"])):
                finished.append(
                    {
                        "name": job["name"],
                        "dataset": job["task"].dataset,
                        "task_name": job["task"].task_name,
                        "task_type": job["task"].task_type,
                        "relation_target": job["task"].relation_target,
                        "gpu": job["gpu"],
                        "return_code": 0,
                        "elapsed_sec": 0.0,
                        "out_path": str(job["out_path"]),
                        "log_path": str(job["log_path"]),
                        "cmd": " ".join(str(x) for x in job["cmd"]),
                    }
                )
                continue
            cmd = [str(x).replace("{gpu}", str(job["gpu"])) for x in job["cmd"]]
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            log_f = Path(job["log_path"]).open("w", encoding="utf-8")
            print(f"[dygformer-edge] launch {job['name']} gpu={job['gpu']}")
            append_dataset_log(project_root, job["task"].dataset, [f"- launch `{job['name']}` on GPU `{job['gpu']}`"])
            proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append({**job, "proc": proc, "log_f": log_f, "start": time.time(), "cmd_resolved": cmd})

        time.sleep(max(1, int(args.poll_sec)))
        keep_running: List[Dict[str, object]] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                keep_running.append(item)
                tail = tail_last_line(Path(item["log_path"]))
                if tail:
                    print(f"[dygformer-edge] {item['name']}: {tail}")
                continue
            item["log_f"].close()
            elapsed = float(time.time() - item["start"])
            print(f"[dygformer-edge] done {item['name']} rc={rc} elapsed={elapsed:.1f}s")
            append_dataset_log(
                project_root,
                item["task"].dataset,
                [f"- finished `{item['name']}` rc=`{rc}` elapsed_sec=`{elapsed:.1f}`"],
            )
            finished.append(
                {
                    "name": item["name"],
                    "dataset": item["task"].dataset,
                    "task_name": item["task"].task_name,
                    "task_type": item["task"].task_type,
                    "relation_target": item["task"].relation_target,
                    "gpu": item["gpu"],
                    "return_code": int(rc),
                    "elapsed_sec": elapsed,
                    "out_path": str(item["out_path"]),
                    "log_path": str(item["log_path"]),
                    "cmd": " ".join(item["cmd_resolved"]),
                }
            )
        running = keep_running

    write_csv(
        out_root / "run_manifest.csv",
        finished,
        [
            "name",
            "dataset",
            "task_name",
            "task_type",
            "relation_target",
            "gpu",
            "return_code",
            "elapsed_sec",
            "out_path",
            "log_path",
            "cmd",
        ],
    )

    metric_rows: List[Dict[str, object]] = []
    for job in jobs:
        metric_rows.extend(parse_dygformer_csv(Path(job["out_path"]), job["task"]))
    write_csv(
        out_root / "metrics_long.csv",
        metric_rows,
        [
            "dataset",
            "task_name",
            "task_type",
            "relation_target",
            "family",
            "model",
            "status",
            "val_mrr",
            "val_auc",
            "val_ap",
            "test_mrr",
            "test_auc",
            "test_ap",
            "avg_epoch_sec",
            "total_sec",
            "total_params",
            "trainable_params",
            "device",
            "source_path",
        ],
    )
    if metric_rows:
        df = pd.DataFrame(metric_rows)
        summary = (
            df.sort_values(["dataset", "task_type", "task_name"])
            [["dataset", "task_name", "task_type", "relation_target", "status", "val_mrr", "val_auc", "val_ap", "test_mrr", "test_auc", "test_ap", "avg_epoch_sec", "total_sec", "total_params", "trainable_params", "device", "source_path"]]
        )
        summary.to_csv(out_root / "summary.csv", index=False)

    append_master_log(
        project_root,
        [
            "- DyGFormer-only edge baseline campaign finished",
            f"- manifest: `{out_root / 'run_manifest.csv'}`",
            f"- metrics_long: `{out_root / 'metrics_long.csv'}`",
            f"- summary: `{out_root / 'summary.csv'}`",
        ],
    )
    print(out_root)


if __name__ == "__main__":
    main()
