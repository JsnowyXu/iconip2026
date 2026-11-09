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

import pandas as pd

from edge_benchmark_common import (
    DATASET_ORDER,
    TaskSpec,
    append_dataset_log,
    append_master_log,
    build_edge_task_specs,
    ensure_thgl_tasks,
)


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


def parse_simple_split_csv(path: Path, family: str, task: TaskSpec, elapsed_sec: float) -> List[Dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": task.dataset,
                "family": family,
                "model": r.get("model", family),
                "task_name": task.task_name,
                "task_type": task.task_type,
                "relation_target": task.relation_target,
                "split": str(r.get("split", "")),
                "status": "ok",
                "val_mrr": r.get("mrr") if str(r.get("split", "")) == "val" else None,
                "val_auc": r.get("auc") if str(r.get("split", "")) == "val" else None,
                "val_ap": r.get("ap") if str(r.get("split", "")) == "val" else None,
                "test_mrr": r.get("mrr") if str(r.get("split", "")) == "test" else None,
                "test_auc": r.get("auc") if str(r.get("split", "")) == "test" else None,
                "test_ap": r.get("ap") if str(r.get("split", "")) == "test" else None,
                "avg_epoch_sec": r.get("avg_epoch_sec"),
                "total_sec": elapsed_sec,
                "total_params": r.get("total_params"),
                "trainable_params": r.get("trainable_params"),
                "device": r.get("device"),
                "source_path": str(path),
            }
        )
    return rows


def parse_model_zoo_csv(path: Path, family: str, task: TaskSpec) -> List[Dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": task.dataset,
                "family": family,
                "model": r.get("model", ""),
                "task_name": task.task_name,
                "task_type": task.task_type,
                "relation_target": task.relation_target,
                "split": "test",
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


def main_job_profile(args, task: TaskSpec) -> Dict[str, object]:
    batch_size = "512" if task.dataset.startswith("patent_") else "256"
    patience = int(args.main_patience)
    extra = [
        "--lambda-cl",
        "0.05",
        "--recent-neighbor-size",
        "12",
    ]

    if task.dataset.startswith("patent_"):
        extra = [
            "--lambda-cl",
            "0.05",
            "--recent-neighbor-size",
            "12",
        ]
        patience = max(patience, 7)
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
        patience = max(patience, 8)
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
        patience = max(patience, 8)
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
        patience = max(patience, 8)

    return {
        "batch_size": batch_size,
        "patience": patience,
        "extra": extra,
    }


def main_job_cmd(args, out_dir: Path, task: TaskSpec, gpu: int) -> List[str]:
    profile = main_job_profile(args, task)
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
        str(args.main_epochs),
        "--batch-size",
        str(profile["batch_size"]),
        "--lr",
        str(args.main_lr),
        "--gpu-id",
        str(gpu),
        "--early-stop-patience",
        str(profile["patience"]),
        "--task-type",
        str(task.task_type),
        "--save-memory-artifacts",
    ]
    cmd.extend(profile["extra"])
    return cmd


def baseline_jobs_for_task(args, out_root: Path, task: TaskSpec) -> List[Dict]:
    metric_root = out_root / "metrics"
    jobs = []
    jobs.append(
        {
            "name": f"tgn__{task.task_name}",
            "family": "tgn",
            "task": task,
            "out_path": metric_root / f"tgn__{task.task_name}.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(task.data_path),
                "--model",
                "tgn",
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(args.baseline_batch_size),
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--gpu-id",
                "{gpu}",
                "--out",
                str(metric_root / f"tgn__{task.task_name}.csv"),
            ],
        }
    )
    jobs.append(
        {
            "name": f"tgn_edgetype__{task.task_name}",
            "family": "tgn_edgetype",
            "task": task,
            "out_path": metric_root / f"tgn_edgetype__{task.task_name}.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(task.data_path),
                "--model",
                "tgn_edgetype",
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(args.baseline_batch_size),
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--gpu-id",
                "{gpu}",
                "--out",
                str(metric_root / f"tgn_edgetype__{task.task_name}.csv"),
            ],
        }
    )
    jobs.append(
        {
            "name": f"edgebank__{task.task_name}",
            "family": "edgebank",
            "task": task,
            "out_path": metric_root / f"edgebank__{task.task_name}.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_edgebank_smoke.py"),
                "--data",
                str(task.data_path),
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(metric_root / f"edgebank__{task.task_name}.csv"),
            ],
        }
    )
    jobs.append(
        {
            "name": f"sthn__{task.task_name}",
            "family": "sthn",
            "task": task,
            "out_path": metric_root / f"sthn__{task.task_name}.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tgb2_official_sthn_smoke.py"),
                "--data",
                str(task.data_path),
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(args.baseline_batch_size),
                "--gpu-id",
                "{gpu}",
                "--dataset-name",
                task.sthn_dataset_slot,
                "--out",
                str(metric_root / f"sthn__{task.task_name}.csv"),
            ],
        }
    )
    jobs.append(
        {
            "name": f"repeatmixer__{task.task_name}",
            "family": "repeatmixer",
            "task": task,
            "out_path": metric_root / f"repeatmixer__{task.task_name}.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_repeatmixer_smoke.py"),
                "--data",
                str(task.data_path),
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(args.baseline_batch_size),
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--gpu-id",
                "{gpu}",
                "--out",
                str(metric_root / f"repeatmixer__{task.task_name}.csv"),
            ],
        }
    )
    jobs.append(
        {
            "name": f"repeatmixer_model_zoo__{task.task_name}",
            "family": "repeatmixer_model_zoo",
            "task": task,
            "out_path": metric_root / f"repeatmixer_model_zoo__{task.task_name}.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_dyglib_model_zoo_smoke.py"),
                "--data",
                str(task.data_path),
                "--models",
                "TGN,TGAT,CAWN,DyGFormer",
                "--epochs",
                str(args.baseline_epochs),
                "--batch-size",
                str(args.baseline_batch_size),
                "--max-edges",
                "0",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--gpu-id",
                "{gpu}",
                "--out",
                str(metric_root / f"repeatmixer_model_zoo__{task.task_name}.csv"),
            ],
        }
    )
    jobs.append(
        {
            "name": f"tabular_ml__{task.task_name}",
            "family": "tabular_ml",
            "task": task,
            "out_path": metric_root / f"tabular_ml__{task.task_name}.csv",
            "cmd": [
                args.python_bin,
                str(args.project_root / "scripts" / "baselines" / "run_tabular_temporal_baselines.py"),
                "--data",
                str(task.data_path),
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--gpu-id",
                "{gpu}",
                "--out",
                str(metric_root / f"tabular_ml__{task.task_name}.csv"),
            ],
        }
    )
    return jobs


def filter_and_sort_baseline_jobs(jobs: List[Dict], families_filter: List[str]) -> List[Dict]:
    family_priority = {
        "tgn": 0,
        "tgn_edgetype": 1,
        "edgebank": 2,
        "sthn": 3,
        "repeatmixer": 4,
        "repeatmixer_model_zoo": 5,
        "tabular_ml": 6,
    }
    filtered = jobs
    if families_filter:
        wanted = set(families_filter)
        filtered = [job for job in jobs if str(job.get("family", "")) in wanted]
    return sorted(
        filtered,
        key=lambda job: (
            family_priority.get(str(job.get("family", "")), 99),
            DATASET_ORDER.index(job["task"].dataset) if job["task"].dataset in DATASET_ORDER else 99,
            str(job["task"].task_name),
        ),
    )


def run_job_queue(
    jobs: List[Dict],
    log_root: Path,
    cwd: Path,
    base_env: Dict[str, str],
    gpu_ids: List[int],
    max_parallel: int,
    skip_existing: bool,
    process_prefix: str,
    project_root: Path,
) -> List[Dict]:
    running = []
    finished = []
    queue = []
    for idx, job in enumerate(jobs):
        item = dict(job)
        item["gpu"] = gpu_ids[idx % len(gpu_ids)]
        item["log_path"] = log_root / f"{job['name']}.log"
        queue.append(item)

    while queue or running:
        while queue and len(running) < max_parallel:
            job = queue.pop(0)
            if skip_existing and job.get("out_path") and Path(job["out_path"]).exists():
                finished.append(
                    {
                        "name": job["name"],
                        "family": job["family"],
                        "task_name": job["task"].task_name,
                        "task_type": job["task"].task_type,
                        "relation_target": job["task"].relation_target,
                        "dataset": job["task"].dataset,
                        "gpu": job["gpu"],
                        "return_code": 0,
                        "elapsed_sec": 0.0,
                        "out_path": str(job.get("out_path", "")),
                        "log_path": str(job["log_path"]),
                        "cmd": " ".join(str(x) for x in job["cmd"]),
                    }
                )
                continue
            cmd = [str(x).replace("{gpu}", str(job["gpu"])) for x in job["cmd"]]
            job["log_path"].parent.mkdir(parents=True, exist_ok=True)
            log_f = job["log_path"].open("w", encoding="utf-8")
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            print(f"[{process_prefix}] launch {job['name']} gpu={job['gpu']}")
            proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            append_dataset_log(project_root, job["task"].dataset, [f"- launch `{job['name']}` on GPU `{job['gpu']}`"])
            running.append({"job": job, "proc": proc, "log_f": log_f, "start": time.time(), "cmd": cmd})

        time.sleep(max(1, 15))
        new_running = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                new_running.append(item)
                continue
            item["log_f"].close()
            elapsed = float(time.time() - item["start"])
            job = item["job"]
            print(f"[{process_prefix}] done {job['name']} rc={rc} elapsed={elapsed:.1f}s")
            append_dataset_log(
                project_root,
                job["task"].dataset,
                [f"- finished `{job['name']}` rc=`{rc}` elapsed_sec=`{elapsed:.1f}`"],
            )
            finished.append(
                {
                    "name": job["name"],
                    "family": job["family"],
                    "task_name": job["task"].task_name,
                    "task_type": job["task"].task_type,
                    "relation_target": job["task"].relation_target,
                    "dataset": job["task"].dataset,
                    "gpu": job["gpu"],
                    "return_code": int(rc),
                    "elapsed_sec": elapsed,
                    "out_path": str(job.get("out_path", "")),
                    "log_path": str(job["log_path"]),
                    "cmd": " ".join(item["cmd"]),
                }
            )
        running = new_running
        for item in running:
            tail = tail_last_line(item["job"]["log_path"])
            if tail:
                print(f"[{process_prefix}] {item['job']['name']}: {tail}")
    return finished


def write_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "family",
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
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def export_baseline_adapters(args, task_specs: List[TaskSpec], base_env: Dict[str, str]) -> None:
    for task in task_specs:
        adapter_dir = args.project_root / "data" / "interim" / "baseline_adapters" / task.task_name
        cmd = [
            args.python_bin,
            str(args.project_root / "scripts" / "datasets" / "export_baseline_formats.py"),
            "--input",
            str(task.data_path),
            "--name",
            task.task_name,
            "--out-dir",
            str(adapter_dir),
        ]
        rc = subprocess.run(cmd, cwd=str(args.project_root), env=base_env).returncode
        if rc != 0:
            raise RuntimeError(f"failed exporting baseline adapters for {task.task_name}")


def main() -> None:
    ap = argparse.ArgumentParser("Run the unified edge-only benchmark campaign")
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--repo-root", type=Path, default=Path("/home/xuxj/2026"))
    ap.add_argument("--python-bin", type=str, default=__import__("sys").executable)
    ap.add_argument("--gpu-ids", type=str, default="0")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--main-epochs", type=int, default=20)
    ap.add_argument("--main-lr", type=float, default=1e-3)
    ap.add_argument("--main-patience", type=int, default=5)
    ap.add_argument("--baseline-epochs", type=int, default=20)
    ap.add_argument("--baseline-batch-size", type=int, default=256)
    ap.add_argument("--eval-num-negs", type=int, default=20)
    ap.add_argument("--datasets", type=str, default="")
    ap.add_argument("--families", type=str, default="")
    ap.add_argument("--phase", type=str, default="both", choices=["both", "main", "baseline"])
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    args.project_root = args.project_root.resolve()
    args.repo_root = args.repo_root.resolve()
    out_root = args.project_root / "results" / "edge_only_formal"
    main_root = out_root / "main"
    baseline_root = out_root / "baselines"
    main_log_root = main_root / "logs"
    baseline_log_root = baseline_root / "logs"
    main_log_root.mkdir(parents=True, exist_ok=True)
    (baseline_root / "metrics").mkdir(parents=True, exist_ok=True)
    baseline_log_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()] or [0]
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(args.project_root) + (os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else "")
    base_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cpu_only = all(gpu_id < 0 for gpu_id in gpu_ids)
    cpu_threads = "8" if cpu_only else "1"
    base_env["OMP_NUM_THREADS"] = cpu_threads
    base_env["MKL_NUM_THREADS"] = cpu_threads
    base_env["OPENBLAS_NUM_THREADS"] = cpu_threads
    base_env["NUMEXPR_NUM_THREADS"] = cpu_threads
    base_env["MKL_THREADING_LAYER"] = "GNU"
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    append_master_log(
        args.project_root,
        [
            "- edge-only benchmark campaign started",
            f"- output root: `{out_root}`",
            f"- gpu_ids: `{gpu_ids}`",
            f"- seed: `{args.seed}`",
        ],
    )

    ensure_thgl_tasks(args.project_root, args.repo_root, args.python_bin, base_env)
    task_specs = build_edge_task_specs(args.project_root)
    selected_datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    selected_families = [x.strip() for x in args.families.split(",") if x.strip()]
    if selected_datasets:
        task_specs = [spec for spec in task_specs if spec.dataset in selected_datasets]
    task_lookup: Dict[str, List[TaskSpec]] = {dataset: [] for dataset in DATASET_ORDER}
    for spec in task_specs:
        task_lookup.setdefault(spec.dataset, []).append(spec)
    if args.phase in {"both", "baseline"}:
        export_baseline_adapters(args, task_specs, base_env)

    main_jobs = []
    baseline_jobs = []
    for dataset in DATASET_ORDER:
        for task in sorted(task_lookup.get(dataset, []), key=lambda x: x.task_name):
            run_name = f"{task.task_name}__seed{args.seed}"
            out_dir = main_root / run_name
            main_jobs.append(
                {
                    "name": run_name,
                    "family": "dhc",
                    "task": task,
                    "out_path": out_dir / "metrics.json",
                    "cmd": main_job_cmd(args, out_dir=out_dir, task=task, gpu=0),
                }
            )
            baseline_jobs.extend(baseline_jobs_for_task(args, out_root=baseline_root, task=task))
            append_dataset_log(
                args.project_root,
                dataset,
                [
                    "- scheduled edge-only benchmark tasks",
                    f"- task: `{task.task_name}`",
                    f"- relation_target: `{task.relation_target}`",
                ],
            )

    main_finished: List[Dict] = []
    if args.phase in {"both", "main"}:
        main_parallel = args.max_parallel if cpu_only else min(args.max_parallel, len(gpu_ids))
        main_finished = run_job_queue(
            jobs=main_jobs,
            log_root=main_log_root,
            cwd=args.project_root,
            base_env=base_env,
            gpu_ids=gpu_ids,
            max_parallel=max(1, main_parallel),
            skip_existing=args.skip_existing,
            process_prefix="edge-main",
            project_root=args.project_root,
        )
        write_manifest(main_root / "run_manifest.csv", main_finished)

    baseline_finished: List[Dict] = []
    if args.phase in {"both", "baseline"}:
        baseline_jobs = filter_and_sort_baseline_jobs(baseline_jobs, selected_families)
        baseline_parallel = args.max_parallel if cpu_only else min(args.max_parallel, len(gpu_ids))
        baseline_finished = run_job_queue(
            jobs=baseline_jobs,
            log_root=baseline_log_root,
            cwd=args.project_root,
            base_env=base_env,
            gpu_ids=gpu_ids,
            max_parallel=max(1, baseline_parallel),
            skip_existing=args.skip_existing,
            process_prefix="edge-baseline",
            project_root=args.project_root,
        )
        write_manifest(baseline_root / "run_manifest.csv", baseline_finished)

    metric_rows = []
    for row in baseline_finished:
        task = TaskSpec(
            dataset=row["dataset"],
            task_name=row["task_name"],
            task_type=row["task_type"],
            relation_target=row["relation_target"],
            data_path=Path("/nonexistent"),
            sthn_dataset_slot="thgl-github",
        )
        out_path = Path(row["out_path"]) if row.get("out_path") else None
        if out_path is None or not out_path.exists():
            continue
        if row["family"] in {"tgn", "tgn_edgetype", "edgebank", "sthn", "repeatmixer", "tabular_ml"}:
            metric_rows.extend(parse_simple_split_csv(out_path, row["family"], task, float(row["elapsed_sec"])))
        elif row["family"] == "repeatmixer_model_zoo":
            metric_rows.extend(parse_model_zoo_csv(out_path, row["family"], task))

    baseline_long = baseline_root / "metrics_long.csv"
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(baseline_long, index=False)
    if args.phase == "both":
        subprocess.run(
            [
                args.python_bin,
                str(args.project_root / "scripts" / "experiments" / "build_edge_only_reports.py"),
                "--project-root",
                str(args.project_root),
                "--campaign-root",
                str(out_root),
            ],
            cwd=str(args.project_root),
            env=base_env,
            check=False,
        )

    append_master_log(
        args.project_root,
        [
            "- edge-only benchmark campaign finished",
            f"- main manifest: `{main_root / 'run_manifest.csv'}`",
            f"- baseline manifest: `{baseline_root / 'run_manifest.csv'}`",
            f"- baseline metrics long: `{baseline_long}`",
            f"- reports root: `{out_root / 'reports'}`",
        ],
    )
    print(out_root)


if __name__ == "__main__":
    main()
