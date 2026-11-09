#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd


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


def parse_split_csv(path: Path, family: str, task: Dict):
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": task["dataset"],
                "family": family,
                "model": r.get("model", family),
                "task_name": task["task_name"],
                "task_type": task["task_type"],
                "relation_target": task["relation_target"],
                "split": str(r.get("split", "")),
                "status": r.get("status", "ok"),
                "mrr": r.get("mrr"),
                "auc": r.get("auc"),
                "ap": r.get("ap"),
                "node_ml_micro_f1": r.get("node_ml_micro_f1"),
                "node_ml_macro_f1": r.get("node_ml_macro_f1"),
                "node_ml_precision_at_1": r.get("node_ml_precision_at_1"),
                "node_ml_precision_at_3": r.get("node_ml_precision_at_3"),
                "node_ml_precision_at_5": r.get("node_ml_precision_at_5"),
                "node_ml_recall_at_1": r.get("node_ml_recall_at_1"),
                "node_ml_recall_at_3": r.get("node_ml_recall_at_3"),
                "node_ml_recall_at_5": r.get("node_ml_recall_at_5"),
                "node_ml_ndcg_at_1": r.get("node_ml_ndcg_at_1"),
                "node_ml_ndcg_at_3": r.get("node_ml_ndcg_at_3"),
                "node_ml_ndcg_at_5": r.get("node_ml_ndcg_at_5"),
                "avg_epoch_sec": r.get("avg_epoch_sec"),
                "memory_mode": r.get("memory_mode"),
                "source_path": str(path),
            }
        )
    return rows


def parse_model_zoo_csv(path: Path, task: Dict):
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": task["dataset"],
                "family": "dyglib_model_zoo",
                "model": r.get("model", "unknown"),
                "task_name": task["task_name"],
                "task_type": task["task_type"],
                "relation_target": task["relation_target"],
                "split": "test",
                "status": r.get("status", "unknown"),
                "mrr": r.get("test_mrr"),
                "auc": r.get("test_auc"),
                "ap": r.get("test_ap"),
                "node_ml_micro_f1": r.get("node_ml_micro_f1"),
                "node_ml_macro_f1": r.get("node_ml_macro_f1"),
                "node_ml_precision_at_1": r.get("node_ml_precision_at_1"),
                "node_ml_precision_at_3": r.get("node_ml_precision_at_3"),
                "node_ml_precision_at_5": r.get("node_ml_precision_at_5"),
                "node_ml_recall_at_1": r.get("node_ml_recall_at_1"),
                "node_ml_recall_at_3": r.get("node_ml_recall_at_3"),
                "node_ml_recall_at_5": r.get("node_ml_recall_at_5"),
                "node_ml_ndcg_at_1": r.get("node_ml_ndcg_at_1"),
                "node_ml_ndcg_at_3": r.get("node_ml_ndcg_at_3"),
                "node_ml_ndcg_at_5": r.get("node_ml_ndcg_at_5"),
                "avg_epoch_sec": r.get("avg_epoch_sec"),
                "memory_mode": None,
                "source_path": str(path),
            }
        )
    return rows


def output_has_success(family: str, out_path: Path) -> bool:
    if not out_path.exists():
        return False
    try:
        df = pd.read_csv(out_path)
    except Exception:
        return False
    if df.empty:
        return False
    if family == "dyglib_model_zoo":
        return "status" in df.columns and any(str(x).strip().lower() == "ok" for x in df["status"].tolist())
    return "split" in df.columns and "test" in {str(x).strip().lower() for x in df["split"].tolist()}


def build_tasks(project_root: Path) -> Dict[str, List[Dict]]:
    task_root = project_root / "data" / "processed" / "tasks"
    edge_tasks = []
    node_tasks = []
    for ds in ["patent_us", "patent_cn", "patent_in"]:
        edge_tasks.extend(
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
            ]
        )
        node_tasks.append(
            {
                "dataset": ds,
                "task_name": f"{ds}_node_patent_multilabel",
                "task_type": "node_multilabel",
                "relation_target": "all_relations",
                "data": task_root / f"{ds}_node_patent_multilabel.pt",
            }
        )
    return {
        "edge": [x for x in edge_tasks if x["data"].exists()],
        "node": [x for x in node_tasks if x["data"].exists()],
    }


def main() -> None:
    ap = argparse.ArgumentParser("Run patent baseline suite")
    ap.add_argument("--project-root", default="/home/xuxj/2026/dyn_relation_memory")
    ap.add_argument("--python-bin", default="/home/xuxj/.conda/envs/pytorch/bin/python")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--gpu-ids", default="2")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dyglib-max-edges", type=int, default=50000)
    ap.add_argument("--eval-num-negs", type=int, default=20)
    ap.add_argument("--models", default="TGN,TGAT,CAWN,TCL,GraphMixer,DyGFormer")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--task-types", default="")
    ap.add_argument("--families", default="")
    ap.add_argument("--skip-successful", action="store_true", default=True)
    ap.add_argument("--no-skip-successful", dest="skip_successful", action="store_false")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root.strip() else (project_root / "results" / "patent_baseline_formal")
    log_root = out_root / "logs"
    metric_root = out_root / "metrics"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)

    selected_datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    selected_task_types = {x.strip() for x in args.task_types.split(",") if x.strip()}
    selected_families = {x.strip() for x in args.families.split(",") if x.strip()}
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()] or [2]

    base_env = os.environ.copy()
    base_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["MKL_THREADING_LAYER"] = "GNU"
    base_env["PYTHONUNBUFFERED"] = "1"

    tasks = build_tasks(project_root)
    edge_tasks = [
        t for t in tasks["edge"]
        if (not selected_datasets or t["dataset"] in selected_datasets)
        and (not selected_task_types or t["task_type"] in selected_task_types)
    ]
    node_tasks = [
        t for t in tasks["node"]
        if (not selected_datasets or t["dataset"] in selected_datasets)
        and (not selected_task_types or t["task_type"] in selected_task_types)
    ]

    for task in edge_tasks:
        adapter_dir = project_root / "data" / "interim" / "baseline_adapters" / task["task_name"]
        cmd = [
            args.python_bin,
            str(project_root / "scripts" / "datasets" / "export_baseline_formats.py"),
            "--input",
            str(task["data"]),
            "--name",
            task["task_name"],
            "--out-dir",
            str(adapter_dir),
        ]
        rc = subprocess.run(cmd, cwd=str(project_root), env=base_env).returncode
        if rc != 0:
            raise RuntimeError(f"failed exporting adapter for {task['task_name']}")

    jobs: List[Dict] = []

    def add_job(name: str, family: str, task: Dict, cmd: List[str], out_path: Path) -> None:
        if selected_families and family not in selected_families:
            return
        jobs.append(
            {
                "name": name,
                "family": family,
                "task": task,
                "cmd": cmd,
                "out_path": out_path,
                "log_path": log_root / f"{name}.log",
                "gpu": gpu_ids[len(jobs) % len(gpu_ids)],
            }
        )

    for task in edge_tasks:
        task_name = task["task_name"]
        add_job(
            f"tgn__{task_name}",
            "tgb2_custom",
            task,
            [args.python_bin, str(project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"), "--data", str(task["data"]), "--model", "tgn", "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--gpu-id", "{gpu}", "--eval-num-negs", str(args.eval_num_negs), "--out", str(metric_root / f"tgn__{task_name}.csv")],
            metric_root / f"tgn__{task_name}.csv",
        )
        add_job(
            f"tgn_edges__{task_name}",
            "tgb2_custom",
            task,
            [args.python_bin, str(project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"), "--data", str(task["data"]), "--model", "tgn_edgetype", "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--gpu-id", "{gpu}", "--eval-num-negs", str(args.eval_num_negs), "--out", str(metric_root / f"tgn_edges__{task_name}.csv")],
            metric_root / f"tgn_edges__{task_name}.csv",
        )
        add_job(
            f"sthn__{task_name}",
            "sthn",
            task,
            [args.python_bin, str(project_root / "scripts" / "baselines" / "run_tgb2_official_sthn_smoke.py"), "--data", str(task["data"]), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--gpu-id", "{gpu}", "--dataset-name", "thgl-github", "--out", str(metric_root / f"sthn__{task_name}.csv")],
            metric_root / f"sthn__{task_name}.csv",
        )
        add_job(
            f"dyglib_model_zoo__{task_name}",
            "dyglib_model_zoo",
            task,
            [args.python_bin, str(project_root / "scripts" / "baselines" / "run_dyglib_model_zoo_smoke.py"), "--data", str(task["data"]), "--models", args.models, "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--max-edges", str(args.dyglib_max_edges), "--gpu-id", "{gpu}", "--eval-num-negs", str(args.eval_num_negs), "--out", str(metric_root / f"dyglib_model_zoo__{task_name}.csv")],
            metric_root / f"dyglib_model_zoo__{task_name}.csv",
        )
        add_job(
            f"repeatmixer__{task_name}",
            "repeatmixer",
            task,
            [args.python_bin, str(project_root / "scripts" / "baselines" / "run_repeatmixer_smoke.py"), "--data", str(task["data"]), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--gpu-id", "{gpu}", "--eval-num-negs", str(args.eval_num_negs), "--out", str(metric_root / f"repeatmixer__{task_name}.csv")],
            metric_root / f"repeatmixer__{task_name}.csv",
        )
        add_job(
            f"edgebank__{task_name}",
            "edgebank",
            task,
            [args.python_bin, str(project_root / "scripts" / "baselines" / "run_edgebank_smoke.py"), "--data", str(task["data"]), "--eval-num-negs", str(args.eval_num_negs), "--out", str(metric_root / f"edgebank__{task_name}.csv")],
            metric_root / f"edgebank__{task_name}.csv",
        )

    for task in node_tasks:
        task_name = task["task_name"]
        add_job(
            f"dyglib_model_zoo__{task_name}",
            "dyglib_model_zoo",
            task,
            [args.python_bin, str(project_root / "scripts" / "baselines" / "run_dyglib_model_zoo_smoke.py"), "--data", str(task["data"]), "--models", args.models, "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--max-edges", str(args.dyglib_max_edges), "--gpu-id", "{gpu}", "--eval-num-negs", str(args.eval_num_negs), "--out", str(metric_root / f"dyglib_model_zoo__{task_name}.csv")],
            metric_root / f"dyglib_model_zoo__{task_name}.csv",
        )

    print(f"[patent-baselines] total_jobs={len(jobs)} max_parallel={args.max_parallel}")
    if args.skip_successful:
        jobs = [j for j in jobs if not output_has_success(j["family"], j["out_path"])]
    print(f"[patent-baselines] filtered_jobs={len(jobs)}")

    queue = jobs[:]
    running = []
    finished = []
    while queue or running:
        while queue and len(running) < args.max_parallel:
            job = queue.pop(0)
            cmd = [str(x).replace("{gpu}", str(job["gpu"])) for x in job["cmd"]]
            log_f = job["log_path"].open("w", encoding="utf-8")
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            print(f"[patent-baselines] launch {job['name']} gpu={job['gpu']}")
            proc = subprocess.Popen(cmd, cwd=str(project_root), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append({"job": job, "proc": proc, "log_f": log_f, "start": time.time(), "cmd": cmd})

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
            print(f"[patent-baselines] done {job['name']} rc={rc} elapsed={elapsed:.1f}s")
            finished.append(
                {
                    "dataset": job["task"]["dataset"],
                    "name": job["name"],
                    "family": job["family"],
                    "task_name": job["task"]["task_name"],
                    "task_type": job["task"]["task_type"],
                    "relation_target": job["task"]["relation_target"],
                    "gpu": job["gpu"],
                    "return_code": int(rc),
                    "elapsed_sec": elapsed,
                    "log_path": str(job["log_path"]),
                    "out_path": str(job["out_path"]),
                    "cmd": " ".join(item["cmd"]),
                }
            )
        running = new_running
        if running:
            print(f"[patent-baselines] active={len(running)} queued={len(queue)}")
            for item in running:
                tail = tail_last_line(item["job"]["log_path"])
                if tail:
                    print(f"  - {item['job']['name']}: {tail}")

    manifest_path = out_root / "run_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(finished[0].keys()) if finished else ["dataset"])
        writer.writeheader()
        for row in finished:
            writer.writerow(row)
    print(f"[patent-baselines] manifest={manifest_path}")

    metric_rows = []
    for row in finished:
        task = {
            "dataset": row["dataset"],
            "task_name": row["task_name"],
            "task_type": row["task_type"],
            "relation_target": row["relation_target"],
        }
        out_path = Path(row["out_path"])
        if row["family"] == "dyglib_model_zoo":
            metric_rows.extend(parse_model_zoo_csv(out_path, task))
        else:
            metric_rows.extend(parse_split_csv(out_path, row["family"], task))
    detail_csv = out_root / "metrics_long.csv"
    pd.DataFrame(metric_rows).to_csv(detail_csv, index=False)
    print(f"[patent-baselines] detail={detail_csv}")


if __name__ == "__main__":
    main()
