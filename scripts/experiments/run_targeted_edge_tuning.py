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

from edge_benchmark_common import append_dataset_log, append_master_log, build_edge_task_specs


def tune_jobs(project_root: Path, python_bin: str, seed: int) -> List[Dict]:
    task_specs = {spec.task_name: spec for spec in build_edge_task_specs(project_root)}
    out_root = project_root / "results" / "targeted_edge_tuning_20260423"
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

    # Tier 1: close-loss tasks, tuned to directly optimize MRR.
    patent_profiles = [
        (
            "mrr_topk8_lr5e4",
            [
                "--epochs",
                "20",
                "--batch-size",
                "256",
                "--lr",
                "0.0005",
                "--early-stop-patience",
                "10",
                "--monitor-metric",
                "val_mrr",
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
            "mrr_topk4_lr7e4",
            [
                "--epochs",
                "20",
                "--batch-size",
                "384",
                "--lr",
                "0.0007",
                "--early-stop-patience",
                "10",
                "--monitor-metric",
                "val_mrr",
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
    for task_name in [
        "patent_cn_edge_org_codeIII_transductive",
        "patent_in_edge_org_codeIII_inductive",
        "patent_in_edge_org_codeIII_transductive",
        "patent_us_edge_org_codeIII_transductive",
    ]:
        for idx, (variant, extra) in enumerate(patent_profiles):
            add(task_name, variant, extra, priority=10 + idx)

    dblp_profiles = [
        (
            "rank_topk8_lr5e4",
            [
                "--epochs",
                "20",
                "--batch-size",
                "256",
                "--lr",
                "0.0005",
                "--early-stop-patience",
                "10",
                "--monitor-metric",
                "val_mrr",
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
            "rank_topk4_lr1e3",
            [
                "--epochs",
                "20",
                "--batch-size",
                "256",
                "--lr",
                "0.001",
                "--early-stop-patience",
                "10",
                "--monitor-metric",
                "val_mrr",
                "--recent-neighbor-size",
                "24",
                "--typed-negative-ratio",
                "0.0",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--community-assign-topk",
                "4",
            ],
        ),
        (
            "rank_nocomm_lr5e4",
            [
                "--epochs",
                "20",
                "--batch-size",
                "256",
                "--lr",
                "0.0005",
                "--early-stop-patience",
                "10",
                "--monitor-metric",
                "val_mrr",
                "--recent-neighbor-size",
                "16",
                "--typed-negative-ratio",
                "0.0",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--no-community",
            ],
        ),
    ]
    for idx, (variant, extra) in enumerate(dblp_profiles):
        add("dblp_edge_inductive", variant, extra, priority=20 + idx)
        add("dblp_edge_transductive", variant, extra, priority=30 + idx)

    # Tier 2: higher-gap tasks with ablation-guided variants.
    aminer_profiles = [
        (
            "rank_norel_lr5e4",
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
                "val_mrr",
                "--recent-neighbor-size",
                "16",
                "--typed-negative-ratio",
                "0.0",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--no-relation-memory",
                "--community-assign-topk",
                "4",
            ],
        ),
        (
            "rank_norel_topk8",
            [
                "--epochs",
                "20",
                "--batch-size",
                "256",
                "--lr",
                "0.001",
                "--early-stop-patience",
                "12",
                "--monitor-metric",
                "val_mrr",
                "--recent-neighbor-size",
                "24",
                "--typed-negative-ratio",
                "0.0",
                "--popular-negative-ratio",
                "0.0",
                "--lambda-cl",
                "0.05",
                "--no-relation-memory",
                "--community-assign-topk",
                "8",
            ],
        ),
    ]
    for idx, (variant, extra) in enumerate(aminer_profiles):
        add("aminer_paper_citation_edge_inductive", variant, extra, priority=40 + idx)
        add("aminer_paper_citation_edge_transductive", variant, extra, priority=50 + idx)

    thgl_profiles = [
        (
            "rank_topk8_lr5e4",
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
                "8",
            ],
        ),
        (
            "rank_no_edge_type",
            [
                "--epochs",
                "20",
                "--batch-size",
                "128",
                "--lr",
                "0.001",
                "--early-stop-patience",
                "10",
                "--monitor-metric",
                "val_mrr",
                "--recent-neighbor-size",
                "24",
                "--typed-negative-ratio",
                "0.2",
                "--popular-negative-ratio",
                "0.1",
                "--lambda-cl",
                "0.05",
                "--no-edge-type-features",
            ],
        ),
    ]
    for idx, (variant, extra) in enumerate(thgl_profiles):
        add("thgl_software_edge_inductive", variant, extra, priority=60 + idx)
        add("thgl_software_edge_transductive", variant, extra, priority=70 + idx)

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
    ap = argparse.ArgumentParser("Targeted edge tuning for the gap-to-baseline tasks")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument("--python-bin", type=str, default="/home/xuxj/.conda/envs/pytorch/bin/python")
    ap.add_argument("--gpu-ids", type=str, default="1,2,3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_root = project_root / "results" / "targeted_edge_tuning_20260423"
    log_root = out_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "run_manifest.csv"
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    jobs = tune_jobs(project_root=project_root, python_bin=args.python_bin, seed=args.seed)

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
            "- targeted tuning campaign started for below-baseline edge tasks",
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
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            log_f = job["log_path"].open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            append_dataset_log(
                project_root,
                job["dataset"],
                [
                    f"- launch targeted tuning `{job['name']}` on GPU `{job['gpu']}`",
                    f"- variant: `{job['variant']}`",
                ],
            )
            running.append({"job": job, "proc": proc, "log_f": log_f, "start": time.time(), "cmd": cmd})
            print(f"[edge-tuning] launch {job['name']} gpu={job['gpu']}")

        time.sleep(15)
        keep_running: List[Dict] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                keep_running.append(item)
                continue
            item["log_f"].close()
            elapsed = float(time.time() - item["start"])
            job = item["job"]
            print(f"[edge-tuning] done {job['name']} rc={rc} elapsed={elapsed:.1f}s")
            append_dataset_log(
                project_root,
                job["dataset"],
                [f"- finished targeted tuning `{job['name']}` rc=`{rc}` elapsed_sec=`{elapsed:.1f}`"],
            )
            finished.append(
                {
                    **job,
                    "return_code": int(rc),
                    "elapsed_sec": elapsed,
                    "cmd": " ".join(item["cmd"]),
                }
            )
            write_manifest(manifest_path, finished + [dict(x["job"], return_code="", elapsed_sec="", cmd=" ".join(x["cmd"])) for x in keep_running])
        running = keep_running

    write_manifest(manifest_path, finished)
    append_master_log(
        project_root,
        [
            "- targeted tuning campaign finished",
            f"- manifest: `{manifest_path}`",
            f"- logs: `{log_root}`",
        ],
    )


if __name__ == "__main__":
    main()
