#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from edge_benchmark_common import TaskSpec, build_edge_task_specs


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


def task_profile(task: TaskSpec) -> Dict[str, object]:
    batch_size = "512" if task.dataset.startswith("patent_") else "256"
    extra = ["--lambda-cl", "0.05", "--recent-neighbor-size", "12"]

    if task.dataset.startswith("patent_"):
        extra = [
            "--lambda-cl",
            "0.10",
            "--recent-neighbor-size",
            "10",
            "--community-assign-temperature",
            "0.7",
            "--community-assign-topk",
            "8",
            "--lambda-community-entropy",
            "0.01",
        ]
    elif task.dataset in {"aminer_paper", "dblp"}:
        extra = [
            "--lambda-cl",
            "0.05",
            "--recent-neighbor-size",
            "16",
            "--monitor-metric",
            "val_mrr",
            "--typed-negative-ratio",
            "0.0",
            "--popular-negative-ratio",
            "0.0",
        ]
    elif task.dataset == "thgl_software":
        batch_size = "128"
        extra = [
            "--lambda-cl",
            "0.05",
            "--recent-neighbor-size",
            "24",
            "--monitor-metric",
            "val_mrr",
            "--typed-negative-ratio",
            "0.2",
            "--popular-negative-ratio",
            "0.1",
        ]
    elif task.dataset == "thgl_github":
        batch_size = "32"
        extra = [
            "--lambda-cl",
            "0.05",
            "--recent-neighbor-size",
            "16",
            "--monitor-metric",
            "val_mrr",
            "--typed-negative-ratio",
            "0.2",
            "--popular-negative-ratio",
            "0.1",
        ]
    return {"batch_size": batch_size, "extra": extra}


def build_variants() -> List[Dict[str, object]]:
    return [
        {"name": "full", "extra": []},
        {"name": "no_edge_type", "extra": ["--no-edge-type-features"]},
        {"name": "no_rel", "extra": ["--no-relation-memory"]},
        {
            "name": "no_edge_type_no_rel",
            "extra": ["--no-edge-type-features", "--no-relation-memory"],
        },
    ]


def load_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    test = obj.get("test", {}) if isinstance(obj, dict) else {}
    best = obj.get("best_val", {}) if isinstance(obj, dict) else {}
    return {
        "best_epoch": obj.get("best_epoch"),
        "stopped_epoch": obj.get("stopped_epoch"),
        "val_mrr": best.get("val_mrr"),
        "val_auc": best.get("val_auc"),
        "val_ap": best.get("val_ap"),
        "test_mrr": test.get("mrr"),
        "test_auc": test.get("auc"),
        "test_ap": test.get("ap"),
        "avg_epoch_time_sec": obj.get("avg_epoch_time_sec"),
    }


def main() -> None:
    ap = argparse.ArgumentParser("Run edge-type vs relation-memory ablations on edge-only tasks")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument("--python-bin", type=str, default="/home/xuxj/.conda/envs/pytorch/bin/python")
    ap.add_argument("--datasets", type=str, default="patent_in,dblp,aminer_paper,thgl_software")
    ap.add_argument("--task-types", type=str, default="edge_transductive")
    ap.add_argument("--variants", type=str, default="full,no_edge_type,no_rel,no_edge_type_no_rel")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--gpu-ids", type=str, default="3")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--out-root", type=str, default="")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root.strip()
        else project_root / "results" / "edge_memory_ablation"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    selected_datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    selected_task_types = {x.strip() for x in args.task_types.split(",") if x.strip()}
    selected_variants = {x.strip() for x in args.variants.split(",") if x.strip()}
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]

    tasks = [
        task
        for task in build_edge_task_specs(project_root)
        if task.dataset in selected_datasets and task.task_type in selected_task_types
    ]
    variants = [v for v in build_variants() if v["name"] in selected_variants]
    jobs: List[Dict[str, object]] = []
    for task in tasks:
        prof = task_profile(task)
        for variant in variants:
            run_name = f"{task.task_name}__{variant['name']}__s{args.seed}"
            jobs.append(
                {
                    "run_name": run_name,
                    "task": task,
                    "gpu": gpu_ids[len(jobs) % len(gpu_ids)] if gpu_ids else None,
                    "batch_size": prof["batch_size"],
                    "extra": list(prof["extra"]) + list(variant["extra"]),
                }
            )

    print(f"[edge-memory-ablation] jobs={len(jobs)} max_parallel={args.max_parallel}")
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(project_root) + (os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else "")
    base_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["PYTHONUNBUFFERED"] = "1"

    queue = jobs[:]
    running: List[Dict[str, object]] = []
    finished: List[Dict[str, object]] = []
    while queue or running:
        while queue and len(running) < args.max_parallel:
            job = queue.pop(0)
            task: TaskSpec = job["task"]
            out_dir = out_root / str(job["run_name"])
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = out_dir / "metrics.json"
            log_path = log_root / f"{job['run_name']}.log"
            if args.skip_existing and metrics_path.exists():
                finished.append(
                    {
                        "run_name": str(job["run_name"]),
                        "dataset": task.dataset,
                        "task_name": task.task_name,
                        "task_type": task.task_type,
                        "gpu": job["gpu"],
                        "return_code": 0,
                        "elapsed_sec": 0.0,
                        "out_dir": str(out_dir),
                        "log_path": str(log_path),
                        "extra": " ".join(job["extra"]),
                    }
                )
                print(f"[edge-memory-ablation] skip existing {job['run_name']}")
                continue

            env = base_env.copy()
            cmd = [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(task.data_path),
                "--out-dir",
                str(out_dir),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(job["batch_size"]),
                "--lr",
                str(args.lr),
                "--early-stop-patience",
                str(args.patience),
                "--task-type",
                task.task_type,
                "--gpu-id",
                "0" if job["gpu"] is not None else "-1",
            ] + list(job["extra"])
            if job["gpu"] is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(int(job["gpu"]))
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""

            print(f"[edge-memory-ablation] launch {job['run_name']} gpu={job['gpu']}")
            log_f = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            running.append(
                {
                    "job": job,
                    "proc": proc,
                    "log_f": log_f,
                    "log_path": log_path,
                    "out_dir": out_dir,
                    "start": time.time(),
                }
            )

        time.sleep(max(1, args.poll_sec))
        new_running: List[Dict[str, object]] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                new_running.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - float(item["start"])
            job = item["job"]
            task = job["task"]
            print(f"[edge-memory-ablation] done {job['run_name']} rc={rc} elapsed={elapsed:.1f}s")
            finished.append(
                {
                    "run_name": str(job["run_name"]),
                    "dataset": task.dataset,
                    "task_name": task.task_name,
                    "task_type": task.task_type,
                    "gpu": job["gpu"],
                    "return_code": int(rc),
                    "elapsed_sec": float(elapsed),
                    "out_dir": str(item["out_dir"]),
                    "log_path": str(item["log_path"]),
                    "extra": " ".join(job["extra"]),
                }
            )
        running = new_running
        for item in running:
            tail = tail_last_line(Path(item["log_path"]))
            if tail:
                print(f"[edge-memory-ablation] {item['job']['run_name']}: {tail}")

    manifest_path = out_root / "run_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "dataset",
                "task_name",
                "task_type",
                "gpu",
                "return_code",
                "elapsed_sec",
                "out_dir",
                "log_path",
                "extra",
            ],
        )
        writer.writeheader()
        for row in finished:
            writer.writerow(row)

    summary_rows = []
    for row in finished:
        metrics = load_metrics(Path(row["out_dir"]) / "metrics.json")
        summary_rows.append({**row, **metrics})
    summary_path = out_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "dataset",
                "task_name",
                "task_type",
                "gpu",
                "return_code",
                "elapsed_sec",
                "val_mrr",
                "val_auc",
                "val_ap",
                "test_mrr",
                "test_auc",
                "test_ap",
                "best_epoch",
                "stopped_epoch",
                "avg_epoch_time_sec",
                "out_dir",
                "log_path",
                "extra",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"[edge-memory-ablation] manifest={manifest_path}")
    print(f"[edge-memory-ablation] summary={summary_path}")


if __name__ == "__main__":
    main()
