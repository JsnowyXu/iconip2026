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

from best_edge_experiment_utils import (
    FOCUS_DATASETS,
    FOCUS_TASK_TYPES,
    anchor_run_args,
    best_runs_to_rows,
    build_train_command,
    load_best_runs,
    render_module_locations_markdown,
)


ABLATION_VARIANTS: List[Dict[str, object]] = [
    {"name": "full", "overrides": {}},
    {"name": "no_relation_memory", "overrides": {"no_relation_memory": True}},
    {"name": "no_edge_type", "overrides": {"no_edge_type_features": True}},
    {"name": "no_community", "overrides": {"no_community": True}},
]


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


def write_readme(out_root: Path, best_rows: List[Dict[str, object]], plan_only: bool) -> None:
    report_lines = [
        "# Best-Config Module Ablation",
        "",
        "This experiment uses the current best full-module main-run config for each target task as the anchor,",
        "then re-runs `full / no_relation_memory / no_edge_type / no_community`.",
        "",
        f"- mode: `{'plan_only' if plan_only else 'executed'}`",
        "",
        "Anchor runs:",
    ]
    for row in best_rows:
        report_lines.append(
            f"- `{row['task_name']}`: best MRR `{row['best_test_mrr']:.6f}` from `{row['source_group']}` -> `{row['metrics_path']}`"
        )
    report_lines.append("")
    report_lines.append("## Module Code Locations")
    report_lines.append("")
    report_lines.append(render_module_locations_markdown().strip())
    (out_root / "README.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser("Run module ablations from the current best edge-task configs")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument("--python-bin", type=str, default="/home/xuxj/.conda/envs/pytorch/bin/python")
    ap.add_argument("--datasets", type=str, default=",".join(FOCUS_DATASETS))
    ap.add_argument("--task-types", type=str, default=",".join(FOCUS_TASK_TYPES))
    ap.add_argument("--gpu-ids", type=str, default="0")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--print-only", action="store_true", default=False)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/xuxj/2026/dyn_relation_memory/results/best_config_module_ablation"),
    )
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    task_types = [x.strip() for x in args.task_types.split(",") if x.strip()]
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

    (out_root / "module_locations.md").write_text(render_module_locations_markdown(), encoding="utf-8")
    write_csv(
        out_root / "best_run_catalog.csv",
        best_runs_to_rows(best_runs),
        ["dataset", "task_name", "task_type", "best_test_mrr", "source_group", "run_name", "metrics_path", "run_dir"],
    )

    jobs: List[Dict[str, object]] = []
    for idx, best_run in enumerate(best_runs):
        base_args = anchor_run_args(best_run, force_full_modules=True)
        for variant in ABLATION_VARIANTS:
            run_args = dict(base_args)
            run_args.update(variant["overrides"])
            run_name = f"{best_run.task.task_name}__{variant['name']}__s{run_args.get('seed', 42)}"
            out_dir = out_root / run_name
            metrics_path = out_dir / "metrics.json"
            log_path = log_root / f"{run_name}.log"
            jobs.append(
                {
                    "dataset": best_run.task.dataset,
                    "task_name": best_run.task.task_name,
                    "task_type": best_run.task.task_type,
                    "variant": variant["name"],
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
                "variant": row["variant"],
                "run_name": row["run_name"],
                "gpu_id": row["gpu_id"],
                "out_dir": str(row["out_dir"]),
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
            "variant",
            "run_name",
            "gpu_id",
            "out_dir",
            "metrics_path",
            "log_path",
            "source_group",
            "source_metrics_path",
            "source_test_mrr",
            "cmd",
        ],
    )
    best_rows = best_runs_to_rows(best_runs)
    write_readme(out_root, best_rows, plan_only=bool(args.print_only))
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
                print(f"[module-ablation] skip existing {job['run_name']}")
                continue
            cmd = build_train_command(args.python_bin, job["task"], job["out_dir"], job["run_args"], job["gpu_id"])
            env = base_env.copy()
            log_f = Path(job["log_path"]).open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            print(f"[module-ablation] launch {job['run_name']} gpu={job['gpu_id']}")
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
            print(f"[module-ablation] done {item['run_name']} rc={rc} elapsed={elapsed:.1f}s")
        running = keep_running

    manifest_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    full_scores: Dict[str, float] = {}
    for row in finished:
        manifest_rows.append(
            {
                "dataset": row["dataset"],
                "task_name": row["task_name"],
                "task_type": row["task_type"],
                "variant": row["variant"],
                "run_name": row["run_name"],
                "gpu_id": row["gpu_id"],
                "return_code": row["return_code"],
                "elapsed_sec": row["elapsed_sec"],
                "out_dir": str(row["out_dir"]),
                "metrics_path": str(row["metrics_path"]),
                "log_path": str(row["log_path"]),
                "source_group": row["source_group"],
                "source_metrics_path": row["source_metrics_path"],
                "source_test_mrr": row["source_test_mrr"],
            }
        )
        metrics = load_metrics(Path(row["metrics_path"]))
        best_val = metrics.get("best_val", {}) if isinstance(metrics, dict) else {}
        test = metrics.get("test", {}) if isinstance(metrics, dict) else {}
        mrr = test.get("mrr")
        if row["variant"] == "full" and mrr is not None:
            full_scores[row["task_name"]] = float(mrr)
        summary_rows.append(
            {
                "dataset": row["dataset"],
                "task_name": row["task_name"],
                "task_type": row["task_type"],
                "variant": row["variant"],
                "run_name": row["run_name"],
                "source_group": row["source_group"],
                "source_metrics_path": row["source_metrics_path"],
                "source_test_mrr": row["source_test_mrr"],
                "return_code": row["return_code"],
                "elapsed_sec": row["elapsed_sec"],
                "val_mrr": best_val.get("val_mrr"),
                "val_auc": best_val.get("val_auc"),
                "val_ap": best_val.get("val_ap"),
                "test_mrr": mrr,
                "test_auc": test.get("auc"),
                "test_ap": test.get("ap"),
                "metrics_path": str(row["metrics_path"]),
            }
        )

    for row in summary_rows:
        full_mrr = full_scores.get(str(row["task_name"]))
        row["delta_vs_full_mrr"] = (float(full_mrr) - float(row["test_mrr"])) if (full_mrr is not None and row.get("test_mrr") is not None) else ""
        row["module_positive"] = (
            str(float(full_mrr) >= float(row["test_mrr"])) if (row["variant"] != "full" and full_mrr is not None and row.get("test_mrr") is not None) else ""
        )

    write_csv(
        out_root / "run_manifest.csv",
        manifest_rows,
        [
            "dataset",
            "task_name",
            "task_type",
            "variant",
            "run_name",
            "gpu_id",
            "return_code",
            "elapsed_sec",
            "out_dir",
            "metrics_path",
            "log_path",
            "source_group",
            "source_metrics_path",
            "source_test_mrr",
        ],
    )
    write_csv(
        out_root / "summary.csv",
        summary_rows,
        [
            "dataset",
            "task_name",
            "task_type",
            "variant",
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
            "delta_vs_full_mrr",
            "module_positive",
            "metrics_path",
        ],
    )

    write_readme(out_root, best_rows, plan_only=False)

    print(out_root)


if __name__ == "__main__":
    main()
