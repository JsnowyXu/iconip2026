#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


def resolve_processed_root(project_root: Path) -> Path:
    primary = project_root / "data" / "processed"
    fallback = project_root / "data" / "processed-auto"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def resolve_results_root(project_root: Path) -> Path:
    primary = project_root / "results"
    fallback = project_root / "results-auto"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def run_one(cmd, cwd: Path, env: dict, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
        t0 = time.time()
        rc = p.wait()
        elapsed = time.time() - t0
    return rc, elapsed


def parse_metrics(metrics_json: Path):
    if not metrics_json.exists():
        return {}
    try:
        obj = json.loads(metrics_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    test = obj.get("test", {})
    node = obj.get("node_eval", {})
    return {
        "test_mrr": test.get("mrr"),
        "test_auc": test.get("auc"),
        "test_ap": test.get("ap"),
        "node_ml_micro_f1": node.get("node_ml_micro_f1"),
        "avg_epoch_time_sec": obj.get("avg_epoch_time_sec"),
        "best_epoch": obj.get("best_epoch"),
        "stopped_epoch": obj.get("stopped_epoch"),
    }


def main() -> None:
    ap = argparse.ArgumentParser("Run cross-relation boost experiments for patent->IPC")
    ap.add_argument("--project-root", type=str, default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--python-bin", type=str, default="python")
    ap.add_argument("--datasets", type=str, default="raw_cn,raw_us")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--save-memory-artifacts", action="store_true", default=False)
    args = ap.parse_args()

    proj = Path(args.project_root).resolve()
    processed_root = resolve_processed_root(proj)
    results_root = resolve_results_root(proj)
    out_root = results_root / "iter_opt" / "cross_relation_boost"
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{proj / 'src'}:{env.get('PYTHONPATH', '')}"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MKL_THREADING_LAYER"] = "GNU"
    env["PYTHONUNBUFFERED"] = "1"
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]

    jobs = []
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        base_data = processed_root / "tasks" / f"{ds}_edge_patent_ipc.pt"
        boost_data = processed_root / "tasks" / f"{ds}_edge_patent_ipc_plus_assignee_ipc.pt"
        if not base_data.exists() or not boost_data.exists():
            print(f"[skip] dataset={ds}, missing base/boost data")
            continue

        plans = [
            ("base", base_data, out_root / f"{ds}_patent_ipc_base"),
            ("boost", boost_data, out_root / f"{ds}_patent_ipc_boost"),
        ]
        for idx, (tag, data_pt, run_dir) in enumerate(plans):
            cmd = [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(data_pt),
                "--out-dir",
                str(run_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--ml-threshold-search",
            ]
            if args.save_memory_artifacts:
                cmd.append("--save-memory-artifacts")
            jobs.append(
                {
                    "dataset": ds,
                    "variant": tag,
                    "data": str(data_pt),
                    "run_dir": str(run_dir),
                    "gpu": gpu_ids[(len(jobs) + idx) % len(gpu_ids)] if gpu_ids else None,
                    "cmd": cmd,
                    "log_path": out_root / "logs" / f"{ds}_{tag}.log",
                }
            )

    rows = []
    queue = list(jobs)
    running = []
    while queue or running:
        while queue and len(running) < int(max(1, args.max_parallel)):
            job = queue.pop(0)
            run_dir = Path(job["run_dir"])
            metrics_json = run_dir / "metrics.json"
            if metrics_json.exists():
                met = parse_metrics(metrics_json)
                rows.append(
                    {
                        "dataset": job["dataset"],
                        "variant": job["variant"],
                        "data": str(job["data"]),
                        "run_dir": str(run_dir),
                        "gpu": job["gpu"],
                        "return_code": 0,
                        "elapsed_sec": 0.0,
                        **met,
                    }
                )
                print(f"[skip] existing dataset={job['dataset']} variant={job['variant']}")
                continue

            job_env = env.copy()
            if job["gpu"] is None:
                job_env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                job_env["CUDA_VISIBLE_DEVICES"] = str(int(job["gpu"]))
                job["cmd"].extend(["--gpu-id", "0"])
            log_path = Path(job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("w", encoding="utf-8")
            print("[run]", " ".join(job["cmd"]), f"gpu={job['gpu']}")
            proc = subprocess.Popen(job["cmd"], cwd=str(proj), env=job_env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append({"job": job, "proc": proc, "log_f": log_f, "start": time.time()})

        time.sleep(10)
        still_running = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - item["start"]
            job = item["job"]
            met = parse_metrics(Path(job["run_dir"]) / "metrics.json")
            rows.append(
                {
                    "dataset": job["dataset"],
                    "variant": job["variant"],
                    "data": str(job["data"]),
                    "run_dir": str(job["run_dir"]),
                    "gpu": job["gpu"],
                    "return_code": int(rc),
                    "elapsed_sec": float(elapsed),
                    **met,
                }
            )
        running = still_running

    df = pd.DataFrame(rows)
    summary_csv = out_root / "summary.csv"
    df.to_csv(summary_csv, index=False)

    md_lines = ["# Cross-Relation Boost Summary", ""]
    if not df.empty:
        md_lines.append(df.to_markdown(index=False))
        md_lines.append("")
        piv = df.pivot_table(index="dataset", columns="variant", values=["test_mrr", "test_auc", "test_ap"], aggfunc="first")
        if not piv.empty and ("test_mrr", "base") in piv.columns and ("test_mrr", "boost") in piv.columns:
            deltas = []
            for ds in piv.index:
                row = {"dataset": ds}
                for m in ["test_mrr", "test_auc", "test_ap"]:
                    try:
                        b = float(piv.loc[ds, (m, "base")])
                        x = float(piv.loc[ds, (m, "boost")])
                        row[f"delta_{m}"] = x - b
                    except Exception:
                        row[f"delta_{m}"] = None
                deltas.append(row)
            ddf = pd.DataFrame(deltas)
            ddf.to_csv(out_root / "delta.csv", index=False)
            md_lines.append("## Boost - Base Delta")
            md_lines.append("")
            md_lines.append(ddf.to_markdown(index=False))
            md_lines.append("")

    summary_md = out_root / "summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(summary_csv)
    print(summary_md)


if __name__ == "__main__":
    main()
