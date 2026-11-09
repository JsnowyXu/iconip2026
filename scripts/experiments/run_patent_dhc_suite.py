#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List


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


def build_tasks(project_root: Path) -> List[Dict]:
    task_root = project_root / "data" / "processed" / "tasks"
    tasks = []
    for ds in ["patent_us", "patent_cn", "patent_in"]:
        tasks.extend(
            [
                {
                    "dataset": ds,
                    "task_name": f"{ds}_edge_org_codeIII_transductive",
                    "task_type": "edge_transductive",
                    "relation_target": "org_codeIII",
                    "data": task_root / f"{ds}_edge_org_codeIII_transductive.pt",
                },
                {
                    "dataset": ds,
                    "task_name": f"{ds}_edge_org_codeIII_inductive",
                    "task_type": "edge_inductive",
                    "relation_target": "org_codeIII",
                    "data": task_root / f"{ds}_edge_org_codeIII_inductive.pt",
                },
                {
                    "dataset": ds,
                    "task_name": f"{ds}_node_patent_multilabel",
                    "task_type": "node_multilabel",
                    "relation_target": "all_relations",
                    "data": task_root / f"{ds}_node_patent_multilabel.pt",
                },
            ]
        )
    return [t for t in tasks if t["data"].exists()]


def main() -> None:
    ap = argparse.ArgumentParser("Run patent DHC suite")
    ap.add_argument("--project-root", default="/home/xuxj/2026/dyn_relation_memory")
    ap.add_argument("--python-bin", default="/home/xuxj/.conda/envs/pytorch/bin/python")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--gpu-ids", default="1")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--task-types", default="")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--save-memory-artifacts", action="store_true", default=True)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root.strip() else (project_root / "results" / "patent_dhc_formal")
    log_root = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
    selected_task_types = {x.strip() for x in args.task_types.split(",") if x.strip()}
    tasks = [
        t
        for t in build_tasks(project_root)
        if (not selected or t["dataset"] in selected)
        and (not selected_task_types or t["task_type"] in selected_task_types)
    ]
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()] or [1]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()] or [42]

    base_env = os.environ.copy()
    base_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    base_env["PYTHONUNBUFFERED"] = "1"

    jobs = []
    for task in tasks:
        for seed in seeds:
            run_name = f"{task['task_name']}__seed{seed}"
            out_dir = out_root / run_name
            gpu = gpu_ids[len(jobs) % len(gpu_ids)]
            cmd = [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(task["data"]),
                "--out-dir",
                str(out_dir),
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--gpu-id",
                str(gpu),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--ml-threshold-search",
                "--task-type",
                str(task["task_type"]),
            ]
            if args.save_memory_artifacts:
                cmd.append("--save-memory-artifacts")
            jobs.append(
                {
                    **task,
                    "seed": seed,
                    "run_name": run_name,
                    "out_dir": out_dir,
                    "log_path": log_root / f"{run_name}.log",
                    "gpu": gpu,
                    "cmd": cmd,
                }
            )

    print(f"[patent-dhc] total_jobs={len(jobs)} max_parallel={args.max_parallel}")
    queue = jobs[:]
    running = []
    finished = []
    while queue or running:
        while queue and len(running) < args.max_parallel:
            job = queue.pop(0)
            metrics_path = job["out_dir"] / "metrics.json"
            if args.skip_existing and metrics_path.exists():
                finished.append(
                    {
                        "dataset": job["dataset"],
                        "run_name": job["run_name"],
                        "task_name": job["task_name"],
                        "task_type": job["task_type"],
                        "relation_target": job["relation_target"],
                        "seed": job["seed"],
                        "gpu": job["gpu"],
                        "return_code": 0,
                        "elapsed_sec": 0.0,
                        "out_dir": str(job["out_dir"]),
                        "log_path": str(job["log_path"]),
                        "cmd": " ".join(job["cmd"]),
                        "status": "skipped_existing",
                    }
                )
                continue
            job["out_dir"].mkdir(parents=True, exist_ok=True)
            log_f = job["log_path"].open("w", encoding="utf-8")
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            env["PYTHONPATH"] = str(project_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            print(f"[patent-dhc] launch {job['run_name']} gpu={job['gpu']}")
            proc = subprocess.Popen(job["cmd"], cwd=str(project_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append({"job": job, "proc": proc, "log_f": log_f, "start": time.time()})

        time.sleep(max(1, args.poll_sec))
        new_running = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                new_running.append(item)
                continue
            item["log_f"].close()
            elapsed = float(time.time() - item["start"])
            job = item["job"]
            print(f"[patent-dhc] done {job['run_name']} rc={rc} elapsed={elapsed:.1f}s")
            finished.append(
                {
                    "dataset": job["dataset"],
                    "run_name": job["run_name"],
                    "task_name": job["task_name"],
                    "task_type": job["task_type"],
                    "relation_target": job["relation_target"],
                    "seed": job["seed"],
                    "gpu": job["gpu"],
                    "return_code": int(rc),
                    "elapsed_sec": elapsed,
                    "out_dir": str(job["out_dir"]),
                    "log_path": str(job["log_path"]),
                    "cmd": " ".join(job["cmd"]),
                    "status": "finished",
                }
            )
        running = new_running
        if running:
            print(f"[patent-dhc] active={len(running)} queued={len(queue)}")
            for item in running:
                tail = tail_last_line(item["job"]["log_path"])
                if tail:
                    print(f"  - {item['job']['run_name']}: {tail}")

    manifest_path = out_root / "run_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(finished[0].keys()) if finished else ["dataset"])
        writer.writeheader()
        for row in finished:
            writer.writerow(row)
    print(f"[patent-dhc] manifest={manifest_path}")

    summary_rows = []
    for row in finished:
        metrics_path = Path(row["out_dir"]) / "metrics.json"
        model_stats_path = Path(row["out_dir"]) / "model_stats.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_stats = json.loads(model_stats_path.read_text(encoding="utf-8")) if model_stats_path.exists() else {}
        best_val = metrics.get("best_val", {}) or {}
        test = metrics.get("test", {}) or {}
        node_eval = metrics.get("node_eval", {}) or {}
        ml_calib = metrics.get("multilabel_threshold_calibration", {}) or {}
        summary_rows.append(
            {
                "dataset": row["dataset"],
                "model": "DHC",
                "run_name": row["run_name"],
                "task_name": row["task_name"],
                "task_type": row["task_type"],
                "relation_target": row["relation_target"],
                "seed": row["seed"],
                "gpu": row["gpu"],
                "return_code": row["return_code"],
                "best_epoch": metrics.get("best_epoch"),
                "stopped_epoch": metrics.get("stopped_epoch"),
                "val_mrr": best_val.get("val_mrr"),
                "val_auc": best_val.get("val_auc"),
                "val_ap": best_val.get("val_ap"),
                "test_mrr": test.get("mrr"),
                "test_auc": test.get("auc"),
                "test_ap": test.get("ap"),
                "test_node_ml_micro_f1": node_eval.get("node_ml_micro_f1"),
                "test_node_ml_macro_f1": node_eval.get("node_ml_macro_f1"),
                "test_node_ml_map": node_eval.get("node_ml_map"),
                "test_node_ml_precision_at_1": node_eval.get("node_ml_precision_at_1"),
                "test_node_ml_precision_at_3": node_eval.get("node_ml_precision_at_3"),
                "test_node_ml_precision_at_5": node_eval.get("node_ml_precision_at_5"),
                "test_node_ml_recall_at_1": node_eval.get("node_ml_recall_at_1"),
                "test_node_ml_recall_at_3": node_eval.get("node_ml_recall_at_3"),
                "test_node_ml_recall_at_5": node_eval.get("node_ml_recall_at_5"),
                "test_node_ml_ndcg_at_1": node_eval.get("node_ml_ndcg_at_1"),
                "test_node_ml_ndcg_at_3": node_eval.get("node_ml_ndcg_at_3"),
                "test_node_ml_ndcg_at_5": node_eval.get("node_ml_ndcg_at_5"),
                "ml_threshold": ml_calib.get("threshold"),
                "avg_epoch_time_sec": metrics.get("avg_epoch_time_sec"),
                "elapsed_sec": row["elapsed_sec"],
                "memory_trace_file": metrics.get("memory_trace_file", ""),
                "memory_final_state_file": metrics.get("memory_final_state_file", ""),
                "total_params": model_stats.get("total_params"),
                "trainable_params": model_stats.get("trainable_params"),
                "device": model_stats.get("device"),
                "out_dir": row["out_dir"],
            }
        )
    if summary_rows:
        summary_path = out_root / "summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"[patent-dhc] summary={summary_path}")


if __name__ == "__main__":
    main()
