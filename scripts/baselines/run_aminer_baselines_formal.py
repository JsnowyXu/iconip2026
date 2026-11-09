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


def parse_split_csv(path: Path, family: str, model: str, task_name: str, task_type: str, relation_target: str, elapsed_sec: float):
    rows = []
    if not path.exists():
        return rows
    try:
        df = pd.read_csv(path)
    except Exception:
        return rows
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "aminer_paper",
                "family": family,
                "model": r.get("model", model),
                "task_name": task_name,
                "task_type": task_type,
                "relation_target": relation_target,
                "split": str(r.get("split", "")),
                "status": "ok",
                "val_mrr": r.get("mrr", None) if str(r.get("split", "")) == "val" else None,
                "val_auc": r.get("auc", None) if str(r.get("split", "")) == "val" else None,
                "val_ap": r.get("ap", None) if str(r.get("split", "")) == "val" else None,
                "test_mrr": r.get("mrr", None) if str(r.get("split", "")) == "test" else None,
                "test_auc": r.get("auc", None) if str(r.get("split", "")) == "test" else None,
                "test_ap": r.get("ap", None) if str(r.get("split", "")) == "test" else None,
                "node_ml_micro_f1": r.get("node_ml_micro_f1", None),
                "node_ml_macro_f1": r.get("node_ml_macro_f1", None),
                "avg_epoch_sec": r.get("avg_epoch_sec", None),
                "total_sec": elapsed_sec,
                "total_params": r.get("total_params", None),
                "trainable_params": r.get("trainable_params", None),
                "device": r.get("device", None),
                "source_path": str(path),
            }
        )
    return rows


def parse_model_zoo_csv(path: Path, family: str, task_name: str, task_type: str, relation_target: str):
    rows = []
    if not path.exists():
        return rows
    try:
        df = pd.read_csv(path)
    except Exception:
        return rows
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": "aminer_paper",
                "family": family,
                "model": r.get("model", "unknown"),
                "task_name": task_name,
                "task_type": task_type,
                "relation_target": relation_target,
                "split": "test",
                "status": r.get("status", "unknown"),
                "val_mrr": r.get("val_mrr", None),
                "val_auc": r.get("val_auc", None),
                "val_ap": r.get("val_ap", None),
                "test_mrr": r.get("test_mrr", None),
                "test_auc": r.get("test_auc", None),
                "test_ap": r.get("test_ap", None),
                "node_ml_micro_f1": r.get("node_ml_micro_f1", None),
                "node_ml_macro_f1": r.get("node_ml_macro_f1", None),
                "avg_epoch_sec": r.get("avg_epoch_sec", None),
                "total_sec": r.get("total_sec", None),
                "total_params": r.get("total_params", None),
                "trainable_params": r.get("trainable_params", None),
                "device": r.get("device", None),
                "source_path": str(path),
            }
        )
    return rows


def split_csv_has_success(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or "split" not in df.columns:
        return False
    splits = set(str(x).strip().lower() for x in df["split"].tolist())
    return "test" in splits


def model_zoo_csv_has_success(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or "status" not in df.columns:
        return False
    status = [str(x).strip().lower() for x in df["status"].tolist()]
    return any(x == "ok" for x in status)


def output_has_success(family: str, out_path: Path) -> bool:
    if family in {"tgb2_custom", "dyglib_custom", "sthn", "repeatmixer"}:
        return split_csv_has_success(out_path)
    if family == "dyglib_model_zoo":
        return model_zoo_csv_has_success(out_path)
    return False


def build_tasks(project_root: Path) -> Dict[str, List[Dict]]:
    task_root = project_root / "data" / "processed" / "paper_tasks"
    edge_tasks = [
        {
            "task_name": "aminer_paper_citation_edge_transductive",
            "task_type": "edge_transductive",
            "relation_target": "paper_cite_paper",
            "data": task_root / "aminer_paper_citation_edge_transductive.pt",
        },
        {
            "task_name": "aminer_paper_citation_edge_inductive",
            "task_type": "edge_inductive",
            "relation_target": "paper_cite_paper",
            "data": task_root / "aminer_paper_citation_edge_inductive.pt",
        },
        {
            "task_name": "aminer_paper_coauthor_edge_transductive",
            "task_type": "edge_transductive",
            "relation_target": "author_coauthor_author",
            "data": task_root / "aminer_paper_coauthor_edge_transductive.pt",
        },
        {
            "task_name": "aminer_paper_coauthor_edge_inductive",
            "task_type": "edge_inductive",
            "relation_target": "author_coauthor_author",
            "data": task_root / "aminer_paper_coauthor_edge_inductive.pt",
        },
    ]
    node_tasks = [
        {
            "task_name": "aminer_paper_node_multilabel",
            "task_type": "node_multilabel",
            "relation_target": "all_relations",
            "data": task_root / "aminer_paper_node_multilabel.pt",
        }
    ]
    return {
        "edge": [x for x in edge_tasks if x["data"].exists()],
        "node": [x for x in node_tasks if x["data"].exists()],
    }


def main() -> None:
    ap = argparse.ArgumentParser("Run AMiner baseline suite")
    ap.add_argument("--project-root", type=str, default="/home/xuxj/2026/dyn_relation_memory")
    ap.add_argument("--python-bin", type=str, default="/home/xuxj/.conda/envs/pytorch/bin/python")
    ap.add_argument("--out-root", type=str, default="")
    ap.add_argument("--gpu-ids", type=str, default="1")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dyglib-max-edges", type=int, default=50000)
    ap.add_argument("--eval-num-negs", type=int, default=20)
    ap.add_argument("--models", type=str, default="JODIE,DyRep,TGAT,CAWN,TCL,GraphMixer,DyGFormer")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--families", type=str, default="", help="Optional comma-separated family filter.")
    ap.add_argument("--skip-successful", action="store_true", default=True)
    ap.add_argument("--no-skip-successful", dest="skip_successful", action="store_false")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root.strip() else (project_root / "results" / "baseline_formal_aminer_custom")
    log_root = out_root / "logs"
    metric_root = out_root / "metrics"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["MKL_THREADING_LAYER"] = "GNU"
    base_env["PYTHONUNBUFFERED"] = "1"

    tasks = build_tasks(project_root)
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpu_ids:
        gpu_ids = [1]
    selected_families = {x.strip() for x in args.families.split(",") if x.strip()}

    for task in tasks["edge"]:
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
        gpu = gpu_ids[len(jobs) % len(gpu_ids)]
        jobs.append(
            {
                "name": name,
                "family": family,
                "task_name": task["task_name"],
                "task_type": task["task_type"],
                "relation_target": task["relation_target"],
                "data_path": str(task["data"]),
                "gpu": int(gpu),
                "cmd": cmd,
                "out_path": str(out_path),
                "log_path": str(log_root / f"{name}.log"),
            }
        )

    for task in tasks["edge"]:
        task_name = task["task_name"]
        add_job(
            name=f"tgn__{task_name}",
            family="tgb2_custom",
            task=task,
            out_path=metric_root / f"tgn__{task_name}.csv",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(task["data"]),
                "--model",
                "tgn",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--gpu-id",
                "{gpu}",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(metric_root / f"tgn__{task_name}.csv"),
            ],
        )
        add_job(
            name=f"tgn_edges__{task_name}",
            family="tgb2_custom",
            task=task,
            out_path=metric_root / f"tgn_edges__{task_name}.csv",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(task["data"]),
                "--model",
                "tgn_edgetype",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--gpu-id",
                "{gpu}",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(metric_root / f"tgn_edges__{task_name}.csv"),
            ],
        )
        add_job(
            name=f"sthn__{task_name}",
            family="sthn",
            task=task,
            out_path=metric_root / f"sthn__{task_name}.csv",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "baselines" / "run_tgb2_official_sthn_smoke.py"),
                "--data",
                str(task["data"]),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--gpu-id",
                "{gpu}",
                "--dataset-name",
                "thgl-github",
                "--out",
                str(metric_root / f"sthn__{task_name}.csv"),
            ],
        )
        add_job(
            name=f"dyglib_custom__{task_name}",
            family="dyglib_custom",
            task=task,
            out_path=metric_root / f"dyglib_custom__{task_name}.csv",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "baselines" / "run_dyglib_custom_smoke.py"),
                "--data",
                str(task["data"]),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-edges",
                str(args.dyglib_max_edges),
                "--gpu-id",
                "{gpu}",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(metric_root / f"dyglib_custom__{task_name}.csv"),
            ],
        )
        add_job(
            name=f"dyglib_model_zoo__{task_name}",
            family="dyglib_model_zoo",
            task=task,
            out_path=metric_root / f"dyglib_model_zoo__{task_name}.csv",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "baselines" / "run_dyglib_model_zoo_smoke.py"),
                "--data",
                str(task["data"]),
                "--models",
                args.models,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-edges",
                str(args.dyglib_max_edges),
                "--gpu-id",
                "{gpu}",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(metric_root / f"dyglib_model_zoo__{task_name}.csv"),
            ],
        )
        add_job(
            name=f"repeatmixer__{task_name}",
            family="repeatmixer",
            task=task,
            out_path=metric_root / f"repeatmixer__{task_name}.csv",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "baselines" / "run_repeatmixer_smoke.py"),
                "--data",
                str(task["data"]),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--gpu-id",
                "{gpu}",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(metric_root / f"repeatmixer__{task_name}.csv"),
            ],
        )

    for task in tasks["node"]:
        task_name = task["task_name"]
        add_job(
            name=f"dyglib_model_zoo__{task_name}",
            family="dyglib_model_zoo",
            task=task,
            out_path=metric_root / f"dyglib_model_zoo__{task_name}.csv",
            cmd=[
                args.python_bin,
                str(project_root / "scripts" / "baselines" / "run_dyglib_model_zoo_smoke.py"),
                "--data",
                str(task["data"]),
                "--models",
                args.models,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-edges",
                str(args.dyglib_max_edges),
                "--gpu-id",
                "{gpu}",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(metric_root / f"dyglib_model_zoo__{task_name}.csv"),
            ],
        )

    print(f"[aminer-baselines] total_jobs={len(jobs)} max_parallel={args.max_parallel}")
    if selected_families:
        jobs = [j for j in jobs if j["family"] in selected_families]
    if args.skip_successful:
        jobs = [j for j in jobs if not output_has_success(j["family"], Path(j["out_path"]))]
    print(f"[aminer-baselines] filtered_jobs={len(jobs)}")
    queue = jobs[:]
    running: List[Dict] = []
    finished: List[Dict] = []

    while queue or running:
        while queue and len(running) < args.max_parallel:
            job = queue.pop(0)
            cmd = [str(x).replace("{gpu}", str(job["gpu"])) for x in job["cmd"]]
            log_f = Path(job["log_path"]).open("w", encoding="utf-8")
            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            print(f"[aminer-baselines] launch {job['name']} gpu={job['gpu']}")
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
            print(f"[aminer-baselines] done {job['name']} rc={rc} elapsed={elapsed:.1f}s")
            finished.append(
                {
                    "name": job["name"],
                    "family": job["family"],
                    "task_name": job["task_name"],
                    "task_type": job["task_type"],
                    "relation_target": job["relation_target"],
                    "data_path": job["data_path"],
                    "gpu": job["gpu"],
                    "return_code": int(rc),
                    "elapsed_sec": elapsed,
                    "log_path": job["log_path"],
                    "out_path": job["out_path"],
                    "cmd": " ".join(item["cmd"]),
                }
            )
        running = new_running

        if running:
            print(f"[aminer-baselines] active={len(running)} queued={len(queue)}")
            for item in running:
                tail = tail_last_line(Path(item["job"]["log_path"]))
                if tail:
                    print(f"  - {item['job']['name']}: {tail}")

    manifest_path = out_root / "run_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "family",
                "task_name",
                "task_type",
                "relation_target",
                "data_path",
                "gpu",
                "return_code",
                "elapsed_sec",
                "log_path",
                "out_path",
                "cmd",
            ],
        )
        writer.writeheader()
        for row in finished:
            writer.writerow(row)
    print(f"[aminer-baselines] manifest={manifest_path}")

    metric_rows = []
    for row in finished:
        out_path = Path(row["out_path"])
        if row["family"] in {"tgb2_custom", "dyglib_custom", "sthn", "repeatmixer"}:
            metric_rows.extend(
                parse_split_csv(
                    out_path,
                    family=row["family"],
                    model=row["name"].split("__", 1)[0],
                    task_name=row["task_name"],
                    task_type=row["task_type"],
                    relation_target=row["relation_target"],
                    elapsed_sec=float(row["elapsed_sec"]),
                )
            )
        elif row["family"] == "dyglib_model_zoo":
            metric_rows.extend(
                parse_model_zoo_csv(
                    out_path,
                    family=row["family"],
                    task_name=row["task_name"],
                    task_type=row["task_type"],
                    relation_target=row["relation_target"],
                )
            )

    detail_csv = out_root / "metrics_long.csv"
    pd.DataFrame(metric_rows).to_csv(detail_csv, index=False)
    print(f"[aminer-baselines] detail={detail_csv}")

    if metric_rows:
        df = pd.DataFrame(metric_rows)
        summary_rows = []
        for (family, model, task_name, task_type, relation_target), g in df.groupby(
            ["family", "model", "task_name", "task_type", "relation_target"], dropna=False
        ):
            summary_rows.append(
                {
                    "dataset": "aminer_paper",
                    "family": family,
                    "model": model,
                    "task_name": task_name,
                    "task_type": task_type,
                    "relation_target": relation_target,
                    "best_test_mrr": pd.to_numeric(g.get("test_mrr"), errors="coerce").max(),
                    "best_test_auc": pd.to_numeric(g.get("test_auc"), errors="coerce").max(),
                    "best_test_ap": pd.to_numeric(g.get("test_ap"), errors="coerce").max(),
                    "best_node_ml_micro_f1": pd.to_numeric(g.get("node_ml_micro_f1"), errors="coerce").max(),
                    "best_node_ml_macro_f1": pd.to_numeric(g.get("node_ml_macro_f1"), errors="coerce").max(),
                    "mean_avg_epoch_sec": pd.to_numeric(g.get("avg_epoch_sec"), errors="coerce").mean(),
                    "mean_total_sec": pd.to_numeric(g.get("total_sec"), errors="coerce").mean(),
                    "mean_total_params": pd.to_numeric(g.get("total_params"), errors="coerce").mean(),
                }
            )
        summary_csv = out_root / "summary.csv"
        pd.DataFrame(summary_rows).sort_values(["task_name", "family", "model"]).to_csv(summary_csv, index=False)
        print(f"[aminer-baselines] summary={summary_csv}")


if __name__ == "__main__":
    main()
