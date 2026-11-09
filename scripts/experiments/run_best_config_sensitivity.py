#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence

from best_edge_experiment_utils import (
    FOCUS_DATASETS,
    FOCUS_TASK_TYPES,
    anchor_run_args,
    best_runs_to_rows,
    build_train_command,
    load_best_runs,
)


SENSITIVITY_SPECS: Dict[str, Sequence[object]] = {
    "typed_negative_ratio": [0.0, 0.05, 0.1, 0.2],
    "popular_negative_ratio": [0.0, 0.05, 0.1, 0.2],
    "recent_neighbor_size": [8, 12, 16, 24, 32],
    "neighbor_aggregation": ["attn", "mean"],
    "community_assign_topk": [0, 4, 8, 12],
    "num_communities": [16, 32, 64, 96, 128],
}


def load_metrics(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_readme(
    out_root: Path,
    parameters: Sequence[str],
    best_rows: List[Dict[str, object]],
    plan_only: bool,
) -> None:
    report_lines = [
        "# Best-Config Sensitivity",
        "",
        "This sweep anchors each task at its current best full-module main-run config,",
        "then varies one parameter at a time across negative sampling, local-relation aggregation,",
        "and community-memory hyperparameters.",
        "",
        f"- mode: `{'plan_only' if plan_only else 'executed'}`",
        "",
        "Tracked parameters:",
    ]
    for name in parameters:
        report_lines.append(f"- `{name}`: {list(SENSITIVITY_SPECS[name])}")
    report_lines.append("")
    report_lines.append("Anchor runs:")
    for row in best_rows:
        report_lines.append(
            f"- `{row['task_name']}`: best MRR `{row['best_test_mrr']:.6f}` from `{row['source_group']}` -> `{row['metrics_path']}`"
        )
    (out_root / "README.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser("Run one-factor-at-a-time sensitivity sweeps from best edge-task configs")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument("--python-bin", type=str, default="/home/xuxj/.conda/envs/pytorch/bin/python")
    ap.add_argument("--datasets", type=str, default=",".join(FOCUS_DATASETS))
    ap.add_argument("--task-types", type=str, default=",".join(FOCUS_TASK_TYPES))
    ap.add_argument("--parameters", type=str, default=",".join(SENSITIVITY_SPECS.keys()))
    ap.add_argument("--gpu-ids", type=str, default="0")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--print-only", action="store_true", default=False)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/xuxj/2026/dyn_relation_memory/results/best_config_sensitivity"),
    )
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    task_types = [x.strip() for x in args.task_types.split(",") if x.strip()]
    parameters = [x.strip() for x in args.parameters.split(",") if x.strip()]
    for name in parameters:
        if name not in SENSITIVITY_SPECS:
            raise SystemExit(f"Unsupported sensitivity parameter: {name}")

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = [-1]

    best_runs = load_best_runs(
        project_root=project_root,
        datasets=datasets,
        task_types=task_types,
        require_full_modules=True,
    )
    if not best_runs:
        raise SystemExit("No best full-module runs found for the requested tasks.")

    write_csv(
        out_root / "best_run_catalog.csv",
        best_runs_to_rows(best_runs),
        ["dataset", "task_name", "task_type", "best_test_mrr", "source_group", "run_name", "metrics_path", "run_dir"],
    )

    jobs: List[Dict[str, object]] = []
    for best_run in best_runs:
        anchor_args = anchor_run_args(best_run, force_full_modules=False)
        for parameter in parameters:
            anchor_value = anchor_args.get(parameter)
            for value in SENSITIVITY_SPECS[parameter]:
                run_args = dict(anchor_args)
                run_args[parameter] = value
                run_name = f"{best_run.task.task_name}__sens_{parameter}_{str(value).replace('.', 'p')}__s{run_args.get('seed', 42)}"
                out_dir = out_root / run_name
                metrics_path = out_dir / "metrics.json"
                log_path = log_root / f"{run_name}.log"
                jobs.append(
                    {
                        "dataset": best_run.task.dataset,
                        "task_name": best_run.task.task_name,
                        "task_type": best_run.task.task_type,
                        "parameter": parameter,
                        "value": value,
                        "anchor_value": anchor_value,
                        "source_group": best_run.source_group,
                        "source_metrics_path": str(best_run.metrics_path),
                        "source_test_mrr": best_run.test_mrr,
                        "run_name": run_name,
                        "out_dir": out_dir,
                        "metrics_path": metrics_path,
                        "log_path": log_path,
                        "gpu_id": gpu_ids[len(jobs) % len(gpu_ids)],
                        "run_args": run_args,
                        "task": best_run.task,
                    }
                )

    plan_rows: List[Dict[str, object]] = []
    for row in jobs:
        cmd = build_train_command(args.python_bin, row["task"], row["out_dir"], row["run_args"], row["gpu_id"])
        plan_rows.append(
            {
                "dataset": row["dataset"],
                "task_name": row["task_name"],
                "task_type": row["task_type"],
                "parameter": row["parameter"],
                "value": row["value"],
                "anchor_value": row["anchor_value"],
                "run_name": row["run_name"],
                "gpu_id": row["gpu_id"],
                "metrics_path": str(row["metrics_path"]),
                "log_path": str(row["log_path"]),
                "source_group": row["source_group"],
                "source_metrics_path": row["source_metrics_path"],
                "source_test_mrr": row["source_test_mrr"],
                "cmd": " ".join(cmd),
            }
        )
    write_csv(
        out_root / "job_plan.csv",
        plan_rows,
        [
            "dataset",
            "task_name",
            "task_type",
            "parameter",
            "value",
            "anchor_value",
            "run_name",
            "gpu_id",
            "metrics_path",
            "log_path",
            "source_group",
            "source_metrics_path",
            "source_test_mrr",
            "cmd",
        ],
    )
    best_rows = best_runs_to_rows(best_runs)
    write_readme(out_root, parameters, best_rows, plan_only=bool(args.print_only))
    if args.print_only:
        print(out_root / "job_plan.csv")
        return

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
        while queue and len(running) < max(1, int(args.max_parallel)):
            job = queue.pop(0)
            if args.skip_existing and Path(job["metrics_path"]).exists():
                finished.append({**job, "return_code": 0, "elapsed_sec": 0.0})
                print(f"[sensitivity] skip existing {job['run_name']}")
                continue
            cmd = build_train_command(args.python_bin, job["task"], job["out_dir"], job["run_args"], job["gpu_id"])
            env = base_env.copy()
            log_f = Path(job["log_path"]).open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            print(f"[sensitivity] launch {job['run_name']} gpu={job['gpu_id']}")
            running.append({**job, "cmd": " ".join(cmd), "proc": proc, "log_f": log_f, "start": time.time()})

        time.sleep(max(1, int(args.poll_sec)))
        keep_running: List[Dict[str, object]] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                keep_running.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - float(item["start"])
            finished.append({**item, "return_code": int(rc), "elapsed_sec": float(elapsed)})
            print(f"[sensitivity] done {item['run_name']} rc={rc} elapsed={elapsed:.1f}s")
        running = keep_running

    summary_rows: List[Dict[str, object]] = []
    for row in finished:
        metrics = load_metrics(Path(row["metrics_path"]))
        best_val = metrics.get("best_val", {}) if isinstance(metrics, dict) else {}
        test = metrics.get("test", {}) if isinstance(metrics, dict) else {}
        summary_rows.append(
            {
                "dataset": row["dataset"],
                "task_name": row["task_name"],
                "task_type": row["task_type"],
                "parameter": row["parameter"],
                "value": row["value"],
                "anchor_value": row["anchor_value"],
                "run_name": row["run_name"],
                "source_group": row["source_group"],
                "source_metrics_path": row["source_metrics_path"],
                "source_test_mrr": row["source_test_mrr"],
                "return_code": row["return_code"],
                "elapsed_sec": row["elapsed_sec"],
                "val_mrr": best_val.get("val_mrr"),
                "val_auc": best_val.get("val_auc"),
                "val_ap": best_val.get("val_ap"),
                "test_mrr": test.get("mrr"),
                "test_auc": test.get("auc"),
                "test_ap": test.get("ap"),
                "metrics_path": str(row["metrics_path"]),
            }
        )

    write_csv(
        out_root / "summary.csv",
        summary_rows,
        [
            "dataset",
            "task_name",
            "task_type",
            "parameter",
            "value",
            "anchor_value",
            "run_name",
            "source_group",
            "source_metrics_path",
            "source_test_mrr",
            "return_code",
            "elapsed_sec",
            "val_mrr",
            "val_auc",
            "val_ap",
            "test_mrr",
            "test_auc",
            "test_ap",
            "metrics_path",
        ],
    )
    write_readme(out_root, parameters, best_rows, plan_only=False)
    print(out_root)


if __name__ == "__main__":
    main()
