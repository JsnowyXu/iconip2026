#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


def resolve_processed_root(project_root: Path) -> Path:
    auto = project_root / "data" / "processed-auto"
    primary = project_root / "data" / "processed"
    if auto.exists():
        return auto
    return primary


def build_tasks(project_root: Path) -> List[Dict]:
    root = resolve_processed_root(project_root) / "tasks"
    return [
        {
            "task": "raw_cn_edge_assignee_ipc_transductive",
            "data": root / "raw_cn_edge_assignee_ipc_transductive.pt",
            "task_type": "edge_transductive",
        },
        {
            "task": "raw_cn_edge_assignee_ipc_inductive",
            "data": root / "raw_cn_edge_assignee_ipc_inductive.pt",
            "task_type": "edge_inductive",
        },
        {
            "task": "raw_cn_edge_patent_ipc",
            "data": root / "raw_cn_edge_patent_ipc.pt",
            "task_type": "edge_patent_ipc",
        },
    ]


def main_configs() -> List[Dict]:
    return [
        {
            "name": "main_fullbench",
            "extra": ["--ml-threshold-search", "--lambda-cl", "0.05", "--recent-neighbor-size", "12"],
        },
    ]


def tune_configs() -> List[Dict]:
    return [
        {
            "name": "main_fullbench_mrr",
            "extra": [
                "--ml-threshold-search",
                "--lambda-cl",
                "0.05",
                "--recent-neighbor-size",
                "12",
                "--monitor-metric",
                "val_mrr",
            ],
        },
        {
            "name": "patent_strong",
            "extra": ["--ml-threshold-search", "--lambda-cl", "0.10", "--recent-neighbor-size", "10"],
        },
        {
            "name": "patent_strong_mrr",
            "extra": [
                "--ml-threshold-search",
                "--lambda-cl",
                "0.10",
                "--recent-neighbor-size",
                "10",
                "--monitor-metric",
                "val_mrr",
            ],
        },
        {
            "name": "patent_strong_sparse",
            "extra": [
                "--ml-threshold-search",
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
            ],
        },
        {
            "name": "patent_strong_sparse_mrr",
            "extra": [
                "--ml-threshold-search",
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
                "--monitor-metric",
                "val_mrr",
            ],
        },
        {
            "name": "patent_stronger_cl",
            "extra": ["--ml-threshold-search", "--lambda-cl", "0.15", "--recent-neighbor-size", "12"],
        },
        {
            "name": "patent_stronger_cl_mrr",
            "extra": [
                "--ml-threshold-search",
                "--lambda-cl",
                "0.15",
                "--recent-neighbor-size",
                "12",
                "--monitor-metric",
                "val_mrr",
            ],
        },
    ]


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


def load_metrics(metrics_path: Path) -> Dict:
    if not metrics_path.exists():
        return {}
    try:
        obj = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    test = obj.get("test", {}) if isinstance(obj, dict) else {}
    return {
        "test_mrr": float(test.get("mrr")) if test.get("mrr") is not None else None,
        "test_auc": float(test.get("auc")) if test.get("auc") is not None else None,
        "test_ap": float(test.get("ap")) if test.get("ap") is not None else None,
        "best_epoch": obj.get("best_epoch"),
        "stopped_epoch": obj.get("stopped_epoch"),
        "avg_epoch_time_sec": obj.get("avg_epoch_time_sec"),
    }


def load_best_baselines(metrics_dir: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not metrics_dir.exists():
        return out
    for csv_path in sorted(metrics_dir.glob("*.csv")):
        name = csv_path.name
        task_name = None
        for cand in (
            "raw_cn_edge_assignee_ipc_transductive",
            "raw_cn_edge_assignee_ipc_inductive",
            "raw_cn_edge_patent_ipc",
        ):
            if cand in name:
                task_name = cand
                break
        if task_name is None:
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != "test":
                    continue
                cur = out.setdefault(
                    task_name,
                    {
                        "baseline_best_mrr": float("-inf"),
                        "baseline_best_auc": float("-inf"),
                        "baseline_best_ap": float("-inf"),
                    },
                )
                try:
                    cur["baseline_best_mrr"] = max(cur["baseline_best_mrr"], float(row.get("mrr", "nan")))
                except Exception:
                    pass
                try:
                    cur["baseline_best_auc"] = max(cur["baseline_best_auc"], float(row.get("auc", "nan")))
                except Exception:
                    pass
                try:
                    cur["baseline_best_ap"] = max(cur["baseline_best_ap"], float(row.get("ap", "nan")))
                except Exception:
                    pass
    return out


def compare_to_baseline(task: str, metrics: Dict, baselines: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    base = baselines.get(task, {})
    row = {
        "delta_mrr": None,
        "delta_auc": None,
        "delta_ap": None,
        "beats_mrr": 0,
        "beats_auc": 0,
        "beats_ap": 0,
        "beats_majority": 0,
        "meets_strong_gate": 0,
    }
    if not base:
        return row
    if metrics.get("test_mrr") is not None and base.get("baseline_best_mrr") is not None:
        row["delta_mrr"] = float(metrics["test_mrr"] - base["baseline_best_mrr"])
        row["beats_mrr"] = int(row["delta_mrr"] >= 0.0)
    if metrics.get("test_auc") is not None and base.get("baseline_best_auc") is not None:
        row["delta_auc"] = float(metrics["test_auc"] - base["baseline_best_auc"])
        row["beats_auc"] = int(row["delta_auc"] >= 0.0)
    if metrics.get("test_ap") is not None and base.get("baseline_best_ap") is not None:
        row["delta_ap"] = float(metrics["test_ap"] - base["baseline_best_ap"])
        row["beats_ap"] = int(row["delta_ap"] >= 0.0)
    beats = int(row["beats_mrr"]) + int(row["beats_auc"]) + int(row["beats_ap"])
    row["beats_majority"] = int(beats >= 2)
    row["meets_strong_gate"] = int(bool(row["beats_mrr"]) and beats >= 2)
    return row


def run_jobs(
    *,
    project_root: Path,
    python_bin: str,
    out_root: Path,
    log_root: Path,
    jobs: List[Dict],
    gpu_ids: List[int],
    max_parallel: int,
    epochs: int,
    batch_size: int,
    lr: float,
    early_stop_patience: int,
    save_memory_artifacts: bool,
    skip_existing: bool,
    poll_sec: int,
) -> List[Dict]:
    queue = jobs[:]
    running: List[Dict] = []
    finished: List[Dict] = []
    launch_count = 0
    print(f"[head2head] jobs={len(queue)} max_parallel={max_parallel}")

    while queue or running:
        while queue and len(running) < max_parallel:
            job = queue.pop(0)
            out_dir = out_root / job["run_name"]
            metrics_path = out_dir / "metrics.json"
            if skip_existing and metrics_path.exists():
                row = {
                    **job,
                    "return_code": 0,
                    "elapsed_sec": 0.0,
                    "out_dir": str(out_dir),
                    "log_path": str(log_root / f"{job['run_name']}.log"),
                }
                row.update(load_metrics(metrics_path))
                finished.append(row)
                print(f"[head2head] skip existing {job['run_name']}")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_root / f"{job['run_name']}.log"
            log_f = log_path.open("w", encoding="utf-8")

            gpu = gpu_ids[launch_count % len(gpu_ids)] if gpu_ids else None
            launch_count += 1

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{project_root / 'src'}:{env.get('PYTHONPATH', '')}"
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""

            cmd = [
                python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(job["data"]),
                "--out-dir",
                str(out_dir),
                "--seed",
                str(job["seed"]),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
                "--early-stop-patience",
                str(early_stop_patience),
            ] + list(job["extra"])
            if gpu is not None:
                cmd += ["--gpu-id", "0"]
            if save_memory_artifacts:
                cmd += ["--save-memory-artifacts"]

            print(f"[head2head] launch {job['run_name']} gpu={gpu}: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append(
                {
                    "job": job,
                    "gpu": gpu,
                    "proc": proc,
                    "log_f": log_f,
                    "log_path": log_path,
                    "out_dir": out_dir,
                    "start": time.time(),
                }
            )

        time.sleep(max(1, poll_sec))
        still_running: List[Dict] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - item["start"]
            metrics = load_metrics(item["out_dir"] / "metrics.json")
            row = {
                **item["job"],
                "gpu": item["gpu"],
                "return_code": int(rc),
                "elapsed_sec": float(elapsed),
                "out_dir": str(item["out_dir"]),
                "log_path": str(item["log_path"]),
            }
            row.update(metrics)
            finished.append(row)
            print(f"[head2head] done {item['job']['run_name']} rc={rc} elapsed={elapsed:.1f}s")
        running = still_running

        if running:
            print(f"[head2head] active={len(running)} queued={len(queue)}")
            for item in running:
                tail = tail_last_line(item["log_path"])
                if tail:
                    print(f"  - {item['job']['run_name']}: {tail}")

    return finished


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_best_by_task(rows: List[Dict], baselines: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
    best: Dict[str, Dict] = {}
    for row in rows:
        if int(row.get("return_code", 1)) != 0:
            continue
        comp = compare_to_baseline(str(row["task"]), row, baselines)
        score = (
            int(comp["meets_strong_gate"]),
            int(comp["beats_majority"]),
            int(comp["beats_mrr"]) + int(comp["beats_auc"]) + int(comp["beats_ap"]),
            float(comp["delta_mrr"] if comp["delta_mrr"] is not None else -1e9),
            float((comp["delta_auc"] or 0.0) + (comp["delta_ap"] or 0.0)),
        )
        cur = best.get(str(row["task"]))
        if cur is None or score > cur["_score"]:
            merged = dict(row)
            merged.update(compare_to_baseline(str(row["task"]), row, baselines))
            merged["_score"] = score
            best[str(row["task"])] = merged
    return best


def main() -> None:
    ap = argparse.ArgumentParser("Run current raw_cn DHC head-to-head vs baselines, then tune and variance-check.")
    ap.add_argument("--project-root", type=str, default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--python-bin", type=str, default="python")
    ap.add_argument("--out-root", type=str, default="")
    ap.add_argument("--baseline-metrics-dir", type=str, default="/home/xuxj/2026/dyn_hetero_community/results-auto/baseline_formal_all_gpu/metrics")
    ap.add_argument("--gpu-ids", type=str, default="3")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variance-seeds", type=str, default="42,43,44")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--save-memory-artifacts", action="store_true", default=True)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root.strip() else project_root / "results-auto" / "head2head_rawcn_current"
    log_root = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    baseline_dir = Path(args.baseline_metrics_dir).resolve()
    baselines = load_best_baselines(baseline_dir)
    tasks = [t for t in build_tasks(project_root) if t["data"].exists()]
    if not tasks:
        raise SystemExit("no raw_cn current tasks found")

    phase1_jobs: List[Dict] = []
    for task in tasks:
        for cfg in main_configs():
            phase1_jobs.append(
                {
                    "phase": "main",
                    "task": task["task"],
                    "task_type": task["task_type"],
                    "config": cfg["name"],
                    "seed": int(args.seed),
                    "data": str(task["data"]),
                    "extra": list(cfg["extra"]),
                    "run_name": f"{task['task']}__{cfg['name']}__s{args.seed}",
                }
            )

    phase1_rows = run_jobs(
        project_root=project_root,
        python_bin=args.python_bin,
        out_root=out_root,
        log_root=log_root,
        jobs=phase1_jobs,
        gpu_ids=gpu_ids,
        max_parallel=max(1, int(args.max_parallel)),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        early_stop_patience=int(args.early_stop_patience),
        save_memory_artifacts=bool(args.save_memory_artifacts),
        skip_existing=bool(args.skip_existing),
        poll_sec=int(args.poll_sec),
    )

    need_tune = []
    for row in phase1_rows:
        if int(row.get("return_code", 1)) != 0:
            need_tune.append(str(row["task"]))
            continue
        comp = compare_to_baseline(str(row["task"]), row, baselines)
        if not comp["meets_strong_gate"]:
            need_tune.append(str(row["task"]))
    need_tune = sorted(set(need_tune))

    phase2_rows: List[Dict] = []
    if need_tune:
        phase2_jobs: List[Dict] = []
        for task in tasks:
            if task["task"] not in need_tune:
                continue
            for cfg in tune_configs():
                phase2_jobs.append(
                    {
                        "phase": "tune",
                        "task": task["task"],
                        "task_type": task["task_type"],
                        "config": cfg["name"],
                        "seed": int(args.seed),
                        "data": str(task["data"]),
                        "extra": list(cfg["extra"]),
                        "run_name": f"{task['task']}__{cfg['name']}__s{args.seed}",
                    }
                )
        phase2_rows = run_jobs(
            project_root=project_root,
            python_bin=args.python_bin,
            out_root=out_root,
            log_root=log_root,
            jobs=phase2_jobs,
            gpu_ids=gpu_ids,
            max_parallel=max(1, int(args.max_parallel)),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            early_stop_patience=int(args.early_stop_patience),
            save_memory_artifacts=bool(args.save_memory_artifacts),
            skip_existing=bool(args.skip_existing),
            poll_sec=int(args.poll_sec),
        )

    all_rows = phase1_rows + phase2_rows
    best = select_best_by_task(all_rows, baselines)

    variance_rows: List[Dict] = []
    variance_seeds = [int(x.strip()) for x in args.variance_seeds.split(",") if x.strip()]
    if best and all(int(v.get("meets_strong_gate", 0)) == 1 for v in best.values()):
        var_jobs: List[Dict] = []
        for task_name, chosen in best.items():
            for seed in variance_seeds:
                if seed == int(chosen["seed"]):
                    continue
                data = next(t["data"] for t in tasks if t["task"] == task_name)
                extra = []
                if chosen["config"] in {cfg["name"] for cfg in main_configs()}:
                    for cfg in main_configs():
                        if cfg["name"] == chosen["config"]:
                            extra = list(cfg["extra"])
                            break
                else:
                    for cfg in tune_configs():
                        if cfg["name"] == chosen["config"]:
                            extra = list(cfg["extra"])
                            break
                var_jobs.append(
                    {
                        "phase": "variance",
                        "task": task_name,
                        "task_type": chosen["task_type"],
                        "config": chosen["config"],
                        "seed": int(seed),
                        "data": str(data),
                        "extra": extra,
                        "run_name": f"{task_name}__{chosen['config']}__s{seed}",
                    }
                )
        if var_jobs:
            variance_rows = run_jobs(
                project_root=project_root,
                python_bin=args.python_bin,
                out_root=out_root,
                log_root=log_root,
                jobs=var_jobs,
                gpu_ids=gpu_ids,
                max_parallel=max(1, int(args.max_parallel)),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                early_stop_patience=int(args.early_stop_patience),
                save_memory_artifacts=bool(args.save_memory_artifacts),
                skip_existing=bool(args.skip_existing),
                poll_sec=int(args.poll_sec),
            )
            all_rows.extend(variance_rows)

    summary_rows: List[Dict] = []
    for row in all_rows:
        comp = compare_to_baseline(str(row["task"]), row, baselines)
        merged = dict(row)
        merged.update(baselines.get(str(row["task"]), {}))
        merged.update(comp)
        summary_rows.append(merged)

    summary_csv = out_root / "head2head_runs.csv"
    write_csv(
        summary_csv,
        summary_rows,
        [
            "phase",
            "task",
            "task_type",
            "config",
            "seed",
            "data",
            "run_name",
            "gpu",
            "return_code",
            "elapsed_sec",
            "out_dir",
            "log_path",
            "test_mrr",
            "test_auc",
            "test_ap",
            "best_epoch",
            "stopped_epoch",
            "avg_epoch_time_sec",
            "baseline_best_mrr",
            "baseline_best_auc",
            "baseline_best_ap",
            "delta_mrr",
            "delta_auc",
            "delta_ap",
            "beats_mrr",
            "beats_auc",
            "beats_ap",
            "beats_majority",
            "meets_strong_gate",
        ],
    )

    best_rows = []
    for task_name in sorted(best):
        row = dict(best[task_name])
        row.update(baselines.get(task_name, {}))
        best_rows.append(row)
    best_csv = out_root / "best_per_task.csv"
    write_csv(
        best_csv,
        best_rows,
        [
            "task",
            "task_type",
            "config",
            "seed",
            "test_mrr",
            "test_auc",
            "test_ap",
            "baseline_best_mrr",
            "baseline_best_auc",
            "baseline_best_ap",
            "delta_mrr",
            "delta_auc",
            "delta_ap",
            "beats_mrr",
            "beats_auc",
            "beats_ap",
            "beats_majority",
            "meets_strong_gate",
            "out_dir",
            "log_path",
            "avg_epoch_time_sec",
        ],
    )

    status = {
        "baseline_dir": str(baseline_dir),
        "tasks": [t["task"] for t in tasks],
        "need_tune_tasks": need_tune,
        "all_tasks_meet_strong_gate": bool(best) and all(int(v.get("meets_strong_gate", 0)) == 1 for v in best.values()),
        "variance_started": bool(variance_rows),
        "summary_csv": str(summary_csv),
        "best_csv": str(best_csv),
    }
    (out_root / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    tracking_script = project_root / "scripts" / "experiments" / "update_tracking_docs.py"
    if tracking_script.exists():
        subprocess.run(
            [
                args.python_bin,
                str(tracking_script),
                "--project-root",
                str(project_root),
            ],
            check=False,
        )

    print(summary_csv)
    print(best_csv)
    print(out_root / "status.json")


if __name__ == "__main__":
    main()
