#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from edge_benchmark_common import append_dataset_log, append_master_log


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    task_name: str
    task_type: str
    relation_target: str
    data_path: Path


def build_target_specs(project_root: Path) -> Dict[str, TaskSpec]:
    processed_root = project_root / "data" / "processed"
    patent_root = processed_root / "tasks"
    thgl_root = processed_root / "tgb2_tasks"
    specs = [
        TaskSpec(
            dataset="patent_in",
            task_name="patent_in_edge_org_codeIII_transductive",
            task_type="edge_transductive",
            relation_target="org_codeIII",
            data_path=patent_root / "patent_in_edge_org_codeIII_transductive.pt",
        ),
        TaskSpec(
            dataset="patent_us",
            task_name="patent_us_edge_org_codeIII_transductive",
            task_type="edge_transductive",
            relation_target="org_codeIII",
            data_path=patent_root / "patent_us_edge_org_codeIII_transductive.pt",
        ),
        TaskSpec(
            dataset="thgl_software",
            task_name="thgl_software_edge_inductive",
            task_type="edge_inductive",
            relation_target="rel_6",
            data_path=thgl_root / "thgl_software_edge_inductive.pt",
        ),
        TaskSpec(
            dataset="thgl_software",
            task_name="thgl_software_edge_transductive",
            task_type="edge_transductive",
            relation_target="rel_6",
            data_path=thgl_root / "thgl_software_edge_transductive.pt",
        ),
    ]
    return {spec.task_name: spec for spec in specs if spec.data_path.exists()}


def build_jobs(project_root: Path, python_bin: str, seed: int) -> List[Dict]:
    task_specs = build_target_specs(project_root)
    out_root = project_root / "results" / "targeted_edge_tuning_20260423_followup"
    jobs: List[Dict] = []

    def add(task_name: str, variant: str, extra: List[str], priority: int) -> None:
        spec = task_specs[task_name]
        out_dir = out_root / f"{task_name}__{variant}__s{seed}"
        cmd = [
            python_bin,
            "-m",
            "dhc.train.run_train",
            "--data",
            str(spec.data_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--task-type",
            str(spec.task_type),
            "--save-memory-artifacts",
        ]
        cmd.extend(extra)
        jobs.append(
            {
                "name": f"{task_name}__{variant}__s{seed}",
                "task_name": task_name,
                "dataset": spec.dataset,
                "task_type": spec.task_type,
                "relation_target": spec.relation_target,
                "variant": variant,
                "priority": int(priority),
                "out_path": out_dir / "metrics.json",
                "cmd": cmd,
            }
        )

    patent_profiles = [
        (
            "close_topk8_lr5e4_ap",
            [
                "--epochs",
                "20",
                "--batch-size",
                "256",
                "--lr",
                "0.0005",
                "--early-stop-patience",
                "12",
                "--monitor-metric",
                "val_ap",
                "--recent-neighbor-size",
                "16",
                "--typed-negative-ratio",
                "0.0",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--community-assign-topk",
                "8",
            ],
        ),
        (
            "close_topk8_lr4e4_ap",
            [
                "--epochs",
                "20",
                "--batch-size",
                "256",
                "--lr",
                "0.0004",
                "--early-stop-patience",
                "12",
                "--monitor-metric",
                "val_ap",
                "--recent-neighbor-size",
                "24",
                "--typed-negative-ratio",
                "0.0",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--community-assign-topk",
                "8",
            ],
        ),
        (
            "close_topk4_lr7e4_ap",
            [
                "--epochs",
                "20",
                "--batch-size",
                "384",
                "--lr",
                "0.0007",
                "--early-stop-patience",
                "12",
                "--monitor-metric",
                "val_ap",
                "--recent-neighbor-size",
                "16",
                "--typed-negative-ratio",
                "0.1",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--community-assign-topk",
                "4",
            ],
        ),
    ]
    for idx, (variant, extra) in enumerate(patent_profiles):
        add("patent_in_edge_org_codeIII_transductive", variant, extra, priority=10 + idx)
        add("patent_us_edge_org_codeIII_transductive", variant, extra, priority=20 + idx)

    thgl_profiles = [
        (
            "noetype_topk4_lr5e4",
            [
                "--epochs",
                "20",
                "--batch-size",
                "128",
                "--lr",
                "0.0005",
                "--early-stop-patience",
                "10",
                "--monitor-metric",
                "val_mrr",
                "--recent-neighbor-size",
                "32",
                "--typed-negative-ratio",
                "0.1",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--community-assign-topk",
                "4",
                "--no-edge-type-features",
            ],
        ),
        (
            "noetype_topk0_lr7e4",
            [
                "--epochs",
                "20",
                "--batch-size",
                "128",
                "--lr",
                "0.0007",
                "--early-stop-patience",
                "10",
                "--monitor-metric",
                "val_mrr",
                "--recent-neighbor-size",
                "24",
                "--typed-negative-ratio",
                "0.1",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--no-edge-type-features",
            ],
        ),
        (
            "noetype_topk8_lr3e4",
            [
                "--epochs",
                "20",
                "--batch-size",
                "128",
                "--lr",
                "0.0003",
                "--early-stop-patience",
                "10",
                "--monitor-metric",
                "val_mrr",
                "--recent-neighbor-size",
                "32",
                "--typed-negative-ratio",
                "0.05",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--community-assign-topk",
                "8",
                "--no-edge-type-features",
            ],
        ),
    ]
    for idx, (variant, extra) in enumerate(thgl_profiles):
        add("thgl_software_edge_inductive", variant, extra, priority=30 + idx)
        add("thgl_software_edge_transductive", variant, extra, priority=40 + idx)

    return sorted(jobs, key=lambda row: (row["priority"], row["task_name"], row["variant"]))


def write_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "dataset",
        "task_name",
        "task_type",
        "relation_target",
        "variant",
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


def main() -> None:
    ap = argparse.ArgumentParser("Follow-up targeted edge tuning for the closest tasks")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument("--python-bin", type=str, default="/home/xuxj/.conda/envs/pytorch/bin/python")
    ap.add_argument("--gpu-ids", type=str, default="1,3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_root = project_root / "results" / "targeted_edge_tuning_20260423_followup"
    log_root = out_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "run_manifest.csv"
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    jobs = build_jobs(project_root=project_root, python_bin=args.python_bin, seed=args.seed)

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

    append_master_log(
        project_root,
        [
            "- follow-up targeted tuning campaign started for 4 near-line tasks",
            f"- output root: `{out_root}`",
            f"- gpu_ids: `{gpu_ids}`",
            f"- seed: `{args.seed}`",
        ],
    )

    running: List[Dict] = []
    queue: List[Dict] = []
    finished: List[Dict] = []

    for idx, job in enumerate(jobs):
        job = dict(job)
        job["gpu"] = gpu_ids[idx % len(gpu_ids)]
        job["log_path"] = log_root / f"{job['name']}.log"
        queue.append(job)

    while queue or running:
        while queue and len(running) < len(gpu_ids):
            job = queue.pop(0)
            if args.skip_existing and Path(job["out_path"]).exists():
                finished.append(
                    {
                        **job,
                        "return_code": 0,
                        "elapsed_sec": 0.0,
                        "cmd": " ".join(job["cmd"]),
                    }
                )
                continue
            cmd = list(job["cmd"]) + ["--gpu-id", str(job["gpu"])]
            env = base_env.copy()
            log_f = job["log_path"].open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            append_dataset_log(
                project_root,
                job["dataset"],
                [
                    f"- launch follow-up tuning `{job['name']}` on GPU `{job['gpu']}`",
                    f"- variant: `{job['variant']}`",
                ],
            )
            running.append({"job": job, "proc": proc, "log_f": log_f, "start": time.time(), "cmd": cmd})
            print(f"[edge-tuning-followup] launch {job['name']} gpu={job['gpu']}")

        time.sleep(15)
        keep_running: List[Dict] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                keep_running.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - item["start"]
            job = item["job"]
            finished.append(
                {
                    **job,
                    "return_code": int(rc),
                    "elapsed_sec": float(elapsed),
                    "cmd": " ".join(item["cmd"]),
                }
            )
            append_dataset_log(
                project_root,
                job["dataset"],
                [
                    f"- finished follow-up tuning `{job['name']}` rc=`{rc}` elapsed_sec=`{elapsed:.1f}`",
                ],
            )
            print(f"[edge-tuning-followup] done {job['name']} rc={rc} elapsed={elapsed:.1f}s")
        running = keep_running
        write_manifest(manifest_path, finished + [item["job"] for item in running] + queue)

    write_manifest(manifest_path, finished)
    append_master_log(
        project_root,
        [
            "- follow-up targeted tuning campaign finished",
            f"- manifest: `{manifest_path}`",
        ],
    )


if __name__ == "__main__":
    main()
