#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import torch


def resolve_processed_root(project_root: Path) -> Path:
    primary = project_root / "data" / "processed"
    fallback = project_root / "data" / "processed-auto"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def build_task_specs(project_root: Path) -> List[Dict]:
    processed_root = resolve_processed_root(project_root)
    paper = processed_root / "paper_tasks"
    patent = processed_root / "tasks"
    tasks = [
        {"dataset": "arxiv", "task": "arxiv_edge_transductive", "task_type": "edge_transductive", "data": paper / "arxiv_edge_transductive.pt"},
        {"dataset": "arxiv", "task": "arxiv_edge_inductive", "task_type": "edge_inductive", "data": paper / "arxiv_edge_inductive.pt"},
        {"dataset": "arxiv", "task": "arxiv_node_multilabel", "task_type": "node_multilabel", "data": paper / "arxiv_node_multilabel.pt"},
        {"dataset": "dblp", "task": "dblp_edge_transductive", "task_type": "edge_transductive", "data": paper / "dblp_edge_transductive.pt"},
        {"dataset": "dblp", "task": "dblp_edge_inductive", "task_type": "edge_inductive", "data": paper / "dblp_edge_inductive.pt"},
        {"dataset": "dblp", "task": "dblp_node_multilabel", "task_type": "node_multilabel", "data": paper / "dblp_node_multilabel.pt"},
        {"dataset": "amazon_movies", "task": "amazon_movies_edge_transductive", "task_type": "edge_transductive", "data": paper / "amazon_movies_edge_transductive.pt"},
        {"dataset": "amazon_movies", "task": "amazon_movies_edge_inductive", "task_type": "edge_inductive", "data": paper / "amazon_movies_edge_inductive.pt"},
        {"dataset": "amazon_movies", "task": "amazon_movies_node_multilabel", "task_type": "node_multilabel", "data": paper / "amazon_movies_node_multilabel.pt"},
        {"dataset": "raw_cn", "task": "raw_cn_edge_patent_ipc", "task_type": "edge_patent_ipc", "data": patent / "raw_cn_edge_patent_ipc.pt"},
        {"dataset": "raw_cn", "task": "raw_cn_edge_assignee_ipc_transductive", "task_type": "edge_transductive", "data": patent / "raw_cn_edge_assignee_ipc_transductive.pt"},
        {"dataset": "raw_cn", "task": "raw_cn_edge_assignee_ipc_inductive", "task_type": "edge_inductive", "data": patent / "raw_cn_edge_assignee_ipc_inductive.pt"},
        {"dataset": "raw_cn", "task": "raw_cn_node_patent_multilabel", "task_type": "node_multilabel", "data": patent / "raw_cn_node_patent_multilabel.pt"},
        {"dataset": "raw_us", "task": "raw_us_edge_patent_ipc", "task_type": "edge_patent_ipc", "data": patent / "raw_us_edge_patent_ipc.pt"},
        {"dataset": "raw_us", "task": "raw_us_edge_assignee_ipc_transductive", "task_type": "edge_transductive", "data": patent / "raw_us_edge_assignee_ipc_transductive.pt"},
        {"dataset": "raw_us", "task": "raw_us_edge_assignee_ipc_inductive", "task_type": "edge_inductive", "data": patent / "raw_us_edge_assignee_ipc_inductive.pt"},
        {"dataset": "raw_us", "task": "raw_us_node_patent_multilabel", "task_type": "node_multilabel", "data": patent / "raw_us_node_patent_multilabel.pt"},
    ]
    aminer_tasks = [
        {"dataset": "aminer_paper", "task": "aminer_paper_edge_transductive", "task_type": "edge_transductive", "data": paper / "aminer_paper_edge_transductive.pt"},
        {"dataset": "aminer_paper", "task": "aminer_paper_edge_inductive", "task_type": "edge_inductive", "data": paper / "aminer_paper_edge_inductive.pt"},
        {"dataset": "aminer_paper", "task": "aminer_paper_node_multilabel", "task_type": "node_multilabel", "data": paper / "aminer_paper_node_multilabel.pt"},
    ]
    if all(Path(t["data"]).exists() for t in aminer_tasks):
        tasks.extend(aminer_tasks)
    return tasks


def build_variants() -> List[Dict]:
    return [
        {"name": "full", "extra": []},
        {"name": "no_recent", "extra": ["--no-recent-neighbors", "--typed-negative-ratio", "0.0", "--popular-negative-ratio", "0.0"]},
        {"name": "no_rel", "extra": ["--no-relation-memory"]},
        {"name": "no_comm", "extra": ["--no-community", "--lambda-cl", "0.0"]},
        {"name": "uniform_neg", "extra": ["--typed-negative-ratio", "0.0", "--popular-negative-ratio", "0.0"]},
    ]


def task_is_hetero_main_ready(task_path: Path) -> bool:
    if not task_path.exists():
        return False
    try:
        obj = torch.load(task_path, map_location="cpu")
        meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
        extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
        return bool(extra.get("hetero_main_benchmark_ok", True))
    except Exception:
        return True


def task_default_args(dataset: str, task: str, task_type: str) -> List[str]:
    extra: List[str] = ["--ml-threshold-search"]
    if task_type == "node_multilabel":
        # Use per-dataset settings validated in results/grid_search_all_node_multilabel.
        extra += ["--node-eval-max-nodes", "0"]
        if dataset == "arxiv":
            extra += ["--ml-pos-weight", "--lambda-ml", "0.3"]
    if task_type.startswith("edge"):
        extra += ["--lambda-cl", "0.05", "--recent-neighbor-size", "12"]
    return extra


def replace_cli_arg(args: List[str], flag: str, value: str) -> List[str]:
    out = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        out.append(args[i])
        i += 1
    out += [flag, value]
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--python-bin", type=str, default="python")
    ap.add_argument("--out-root", type=str, default="")
    ap.add_argument("--datasets", type=str, default="arxiv,dblp,amazon_movies,raw_cn,raw_us")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--mode", type=str, default="both", choices=["main", "ablation", "both"])
    ap.add_argument("--ablation-datasets", type=str, default="arxiv,dblp,raw_cn")
    ap.add_argument("--task-types", type=str, default="", help="Optional comma-separated task_type filter.")
    ap.add_argument("--variants", type=str, default="", help="Optional comma-separated variant filter, e.g. full,no_rel,no_comm")
    ap.add_argument("--cuda-visible-devices", type=str, default="")
    ap.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU ids for multi-GPU scheduling.")
    ap.add_argument("--node-eval-max-nodes", type=int, default=-1)
    ap.add_argument("--ml-calib-max-nodes", type=int, default=-1)
    ap.add_argument("--include-single-type-datasets", action="store_true", default=False)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--save-memory-artifacts", action="store_true", default=False)
    ap.add_argument("--save-memory-variants", type=str, default="full", help="Comma-separated variants that should keep memory_trace/final_state.")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root.strip() else project_root / "results" / "full_benchmark_repeats"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    selected_datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    ablation_datasets = {x.strip() for x in args.ablation_datasets.split(",") if x.strip()}
    selected_task_types = {x.strip() for x in args.task_types.split(",") if x.strip()}
    selected_variants = {x.strip() for x in args.variants.split(",") if x.strip()}
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    save_memory_variants = {x.strip() for x in args.save_memory_variants.split(",") if x.strip()}

    task_specs = [t for t in build_task_specs(project_root) if t["dataset"] in selected_datasets]
    if selected_task_types:
        task_specs = [t for t in task_specs if t["task_type"] in selected_task_types]
    if not args.include_single_type_datasets:
        skipped = [t for t in task_specs if not task_is_hetero_main_ready(Path(t["data"]))]
        task_specs = [t for t in task_specs if task_is_hetero_main_ready(Path(t["data"]))]
        if skipped:
            print(
                "[full-benchmark] skip single-type datasets: "
                + ", ".join(sorted({f"{t['dataset']}::{t['task']}" for t in skipped}))
            )
    variants = build_variants()
    if selected_variants:
        variants = [v for v in variants if v["name"] in selected_variants]
    jobs: List[Dict] = []
    for task in task_specs:
        use_variants = [variants[0]]
        if args.mode in {"ablation", "both"} and task["dataset"] in ablation_datasets:
            use_variants = variants if args.mode == "both" else variants[1:]
        elif args.mode == "ablation":
            continue
        for seed in seeds:
            for variant in use_variants:
                extra = task_default_args(task["dataset"], task["task"], task["task_type"]) + variant["extra"]
                if args.node_eval_max_nodes >= 0:
                    extra = replace_cli_arg(extra, "--node-eval-max-nodes", str(args.node_eval_max_nodes))
                if args.ml_calib_max_nodes >= 0:
                    extra = replace_cli_arg(extra, "--ml-calib-max-nodes", str(args.ml_calib_max_nodes))
                jobs.append(
                    {
                        **task,
                        "seed": int(seed),
                        "variant": variant["name"],
                        "extra": extra,
                        "gpu": gpu_ids[len(jobs) % len(gpu_ids)] if gpu_ids else None,
                    }
                )

    running: List[Dict] = []
    finished: List[Dict] = []
    queue = jobs[:]
    print(f"[full-benchmark] jobs={len(queue)} max_parallel={args.max_parallel}")

    while queue or running:
        while queue and len(running) < args.max_parallel:
            job = queue.pop(0)
            data_path = Path(job["data"])
            if not data_path.exists():
                finished.append(
                    {
                        "run_name": f"{job['task']}__{job['variant']}__s{job['seed']}",
                        "dataset": job["dataset"],
                        "task": job["task"],
                        "task_type": job["task_type"],
                        "variant": job["variant"],
                        "seed": int(job["seed"]),
                        "data": str(job["data"]),
                        "return_code": 99,
                        "elapsed_sec": 0.0,
                        "out_dir": "",
                        "log_path": "",
                        "extra": " ".join(job["extra"]),
                    }
                )
                continue
            run_name = f"{job['task']}__{job['variant']}__s{job['seed']}"
            out_dir = out_root / run_name
            metrics_path = out_dir / "metrics.json"
            if args.skip_existing and metrics_path.exists():
                finished.append(
                    {
                        "run_name": run_name,
                        "dataset": job["dataset"],
                        "task": job["task"],
                        "task_type": job["task_type"],
                        "variant": job["variant"],
                        "seed": int(job["seed"]),
                        "data": str(job["data"]),
                        "return_code": 0,
                        "elapsed_sec": 0.0,
                        "out_dir": str(out_dir),
                        "log_path": "",
                        "extra": " ".join(job["extra"]),
                    }
                )
                print(f"[full-benchmark] skip existing {run_name}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_root / f"{run_name}.log"
            log_f = log_path.open("w", encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{project_root / 'src'}:{env.get('PYTHONPATH', '')}"
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            if args.cuda_visible_devices.strip():
                env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()
            elif job.get("gpu") is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(int(job["gpu"]))

            cmd = [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(data_path),
                "--out-dir",
                str(out_dir),
                "--seed",
                str(job["seed"]),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--early-stop-patience",
                str(args.early_stop_patience),
            ] + job["extra"]
            if job.get("gpu") is not None:
                cmd += ["--gpu-id", "0"]
            if args.save_memory_artifacts and job["variant"] in save_memory_variants:
                cmd += ["--save-memory-artifacts"]

            print(f"[full-benchmark] launch {run_name}: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append(
                {
                    "job": job,
                    "run_name": run_name,
                    "proc": proc,
                    "out_dir": out_dir,
                    "log_path": log_path,
                    "log_f": log_f,
                    "start": time.time(),
                }
            )

        time.sleep(max(1, args.poll_sec))
        still_running: List[Dict] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - item["start"]
            job = item["job"]
            finished.append(
                {
                    "run_name": item["run_name"],
                    "dataset": job["dataset"],
                    "task": job["task"],
                    "task_type": job["task_type"],
                    "variant": job["variant"],
                    "seed": int(job["seed"]),
                    "data": str(job["data"]),
                    "return_code": int(rc),
                    "elapsed_sec": float(elapsed),
                    "out_dir": str(item["out_dir"]),
                    "log_path": str(item["log_path"]),
                    "extra": " ".join(job["extra"]),
                }
            )
            print(f"[full-benchmark] done {item['run_name']} rc={rc} elapsed={elapsed:.1f}s")
        running = still_running

        if running:
            print(f"[full-benchmark] active={len(running)} queued={len(queue)}")
            for item in running:
                tail = tail_last_line(item["log_path"])
                if tail:
                    print(f"  - {item['run_name']}: {tail}")

    manifest = out_root / "run_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_name", "dataset", "task", "task_type", "variant", "seed", "data", "return_code", "elapsed_sec", "out_dir", "log_path", "extra"],
        )
        writer.writeheader()
        for row in finished:
            writer.writerow(row)
    print(manifest)


if __name__ == "__main__":
    main()
