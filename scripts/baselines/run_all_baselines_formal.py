#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch


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


def run_sync(cmd: List[str], cwd: Optional[Path], env: Dict[str, str]) -> int:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    return int(p.returncode)


def parse_split_csv(path: Path, family: str, task: str, dataset: str, task_type: str, elapsed_sec: float):
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
                "family": family,
                "dataset": dataset,
                "task": task,
                "task_type": task_type,
                "model": r.get("model", family),
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
                "node_ml_precision_at_1": r.get("node_ml_precision_at_1", None),
                "node_ml_precision_at_3": r.get("node_ml_precision_at_3", None),
                "node_ml_precision_at_5": r.get("node_ml_precision_at_5", None),
                "node_ml_recall_at_1": r.get("node_ml_recall_at_1", None),
                "node_ml_recall_at_3": r.get("node_ml_recall_at_3", None),
                "node_ml_recall_at_5": r.get("node_ml_recall_at_5", None),
                "node_ml_ndcg_at_1": r.get("node_ml_ndcg_at_1", None),
                "node_ml_ndcg_at_3": r.get("node_ml_ndcg_at_3", None),
                "node_ml_ndcg_at_5": r.get("node_ml_ndcg_at_5", None),
                "avg_epoch_sec": r.get("avg_epoch_sec", None),
                "total_sec": elapsed_sec,
                "total_params": r.get("total_params", None),
                "trainable_params": r.get("trainable_params", None),
                "device": r.get("device", None),
                "source_path": str(path),
            }
        )
    return rows


def parse_model_zoo_csv(path: Path, family: str, task: str, dataset: str, task_type: str):
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
                "family": family,
                "dataset": dataset,
                "task": task,
                "task_type": task_type,
                "model": r.get("model", "unknown"),
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
                "node_ml_precision_at_1": r.get("node_ml_precision_at_1", None),
                "node_ml_precision_at_3": r.get("node_ml_precision_at_3", None),
                "node_ml_precision_at_5": r.get("node_ml_precision_at_5", None),
                "node_ml_recall_at_1": r.get("node_ml_recall_at_1", None),
                "node_ml_recall_at_3": r.get("node_ml_recall_at_3", None),
                "node_ml_recall_at_5": r.get("node_ml_recall_at_5", None),
                "node_ml_ndcg_at_1": r.get("node_ml_ndcg_at_1", None),
                "node_ml_ndcg_at_3": r.get("node_ml_ndcg_at_3", None),
                "node_ml_ndcg_at_5": r.get("node_ml_ndcg_at_5", None),
                "avg_epoch_sec": r.get("avg_epoch_sec", None),
                "total_sec": r.get("total_sec", None),
                "total_params": r.get("total_params", None),
                "trainable_params": r.get("trainable_params", None),
                "device": r.get("device", None),
                "source_path": str(path),
            }
        )
    return rows


def parse_tgn_external_log(log_path: Path, family: str, task: str, dataset: str, task_type: str, elapsed_sec: float):
    rows = []
    if not log_path.exists():
        return rows
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    old = re.findall(r"Test statistics: Old nodes -- auc: ([0-9.eE+-]+), ap: ([0-9.eE+-]+)", txt)
    new = re.findall(r"Test statistics: New nodes -- auc: ([0-9.eE+-]+), ap: ([0-9.eE+-]+)", txt)
    if old:
        rows.append(
            {
                "family": family,
                "dataset": dataset,
                "task": task,
                "task_type": task_type,
                "model": "tgn_external",
                "split": "old_nodes",
                "status": "ok",
                "val_auc": None,
                "val_ap": None,
                "test_auc": float(old[-1][0]),
                "test_ap": float(old[-1][1]),
                "node_ml_micro_f1": None,
                "node_ml_macro_f1": None,
                "avg_epoch_sec": None,
                "total_sec": elapsed_sec,
                "total_params": None,
                "trainable_params": None,
                "device": None,
                "source_path": str(log_path),
            }
        )
    if new:
        rows.append(
            {
                "family": family,
                "dataset": dataset,
                "task": task,
                "task_type": task_type,
                "model": "tgn_external",
                "split": "new_nodes",
                "status": "ok",
                "val_auc": None,
                "val_ap": None,
                "test_auc": float(new[-1][0]),
                "test_ap": float(new[-1][1]),
                "node_ml_micro_f1": None,
                "node_ml_macro_f1": None,
                "avg_epoch_sec": None,
                "total_sec": elapsed_sec,
                "total_params": None,
                "trainable_params": None,
                "device": None,
                "source_path": str(log_path),
            }
        )
    return rows


def parse_cdgp_log(log_path: Path, family: str, task: str, dataset: str, task_type: str, elapsed_sec: float):
    rows = []
    if not log_path.exists():
        return rows
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    test_rmsle = re.findall(r"test.*rmsle[:\s]+([0-9.eE+-]+)", txt, flags=re.IGNORECASE)
    test_male = re.findall(r"test.*male[:\s]+([0-9.eE+-]+)", txt, flags=re.IGNORECASE)
    test_mape = re.findall(r"test.*mape[:\s]+([0-9.eE+-]+)", txt, flags=re.IGNORECASE)
    if test_rmsle or test_male or test_mape:
        rows.append(
            {
                "family": family,
                "dataset": dataset,
                "task": task,
                "task_type": task_type,
                "model": "cdgp",
                "split": "test",
                "status": "ok",
                "val_auc": None,
                "val_ap": None,
                "test_auc": None,
                "test_ap": None,
                "node_ml_micro_f1": None,
                "node_ml_macro_f1": None,
                "avg_epoch_sec": None,
                "total_sec": elapsed_sec,
                "total_params": None,
                "trainable_params": None,
                "device": None,
                "cdgp_test_rmsle": float(test_rmsle[-1]) if test_rmsle else None,
                "cdgp_test_male": float(test_male[-1]) if test_male else None,
                "cdgp_test_mape": float(test_mape[-1]) if test_mape else None,
                "source_path": str(log_path),
            }
        )
    return rows


def resolve_processed_root(proj: Path) -> Path:
    primary = proj / "data" / "processed"
    fallback = proj / "data" / "processed-auto"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def official_sthn_dataset_name(dataset: str) -> str:
    dataset = str(dataset).strip().lower()
    if dataset in {"thgl_software", "software"}:
        return "thgl-software"
    return "thgl-github"


def prepare_tasks(proj: Path, py: str, env: Dict[str, str]):
    processed_root = resolve_processed_root(proj)
    paper_dir = processed_root / "paper_tasks"
    paper_dir.mkdir(parents=True, exist_ok=True)
    task_dir = processed_root / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)

    prep_cmds = []

    def add_if_exists(path: Path, cmd: List[str]) -> None:
        if path.exists():
            prep_cmds.append(cmd)

    add_if_exists(
        processed_root / "arxiv_unified.pt",
        [
            py,
            str(proj / "scripts" / "datasets" / "build_paper_task_views.py"),
            "--input",
            str(processed_root / "arxiv_unified.pt"),
            "--out-dir",
            str(paper_dir),
            "--name-prefix",
            "arxiv",
        ],
    )
    add_if_exists(
        processed_root / "dblp_unified.pt",
        [
            py,
            str(proj / "scripts" / "datasets" / "build_paper_task_views.py"),
            "--input",
            str(processed_root / "dblp_unified.pt"),
            "--out-dir",
            str(paper_dir),
            "--name-prefix",
            "dblp",
        ],
    )
    add_if_exists(
        processed_root / "amazon_movies_unified.pt",
        [
            py,
            str(proj / "scripts" / "datasets" / "build_paper_task_views.py"),
            "--input",
            str(processed_root / "amazon_movies_unified.pt"),
            "--out-dir",
            str(paper_dir),
            "--name-prefix",
            "amazon_movies",
            "--target-rel",
            "-1",
        ],
    )
    add_if_exists(
        processed_root / "aminer_paper_unified.pt",
        [
            py,
            str(proj / "scripts" / "datasets" / "build_paper_task_views.py"),
            "--input",
            str(processed_root / "aminer_paper_unified.pt"),
            "--out-dir",
            str(paper_dir),
            "--name-prefix",
            "aminer_paper",
            "--target-rel",
            "3",
        ],
    )
    add_if_exists(
        processed_root / "raw_cn_unified.pt",
        [
            py,
            str(proj / "scripts" / "datasets" / "build_task_views.py"),
            "--input",
            str(processed_root / "raw_cn_unified.pt"),
            "--out-dir",
            str(task_dir),
            "--name-prefix",
            "raw_cn",
        ],
    )
    add_if_exists(
        processed_root / "raw_us_unified.pt",
        [
            py,
            str(proj / "scripts" / "datasets" / "build_task_views.py"),
            "--input",
            str(processed_root / "raw_us_unified.pt"),
            "--out-dir",
            str(task_dir),
            "--name-prefix",
            "raw_us",
        ],
    )
    for c in prep_cmds:
        rc = run_sync(c, cwd=proj, env=env)
        if rc != 0:
            raise RuntimeError(f"failed preparing tasks: {' '.join(c)}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--repo-root", default="/home/xuxj/2026")
    ap.add_argument("--python-bin", default="python")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dyglib-max-edges", type=int, default=50000)
    ap.add_argument("--eval-num-negs", type=int, default=20)
    ap.add_argument("--models", default="JODIE,DyRep,TGN,TGAT,CAWN,TCL,GraphMixer,DyGFormer")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--include-tgn-external", action="store_true")
    ap.add_argument("--include-cdgp", action="store_true")
    ap.add_argument("--include-single-type-datasets", action="store_true", default=False)
    ap.add_argument("--cdgp-epochs", type=int, default=10)
    ap.add_argument("--tgn-epochs", type=int, default=10)
    ap.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    args = ap.parse_args()

    proj = Path(args.project_root).resolve()
    repo = Path(args.repo_root).resolve()

    out_root = Path(args.out_root).resolve() if args.out_root.strip() else (proj / "results" / "baseline_formal_all")
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

    prepare_tasks(proj=proj, py=args.python_bin, env=base_env)
    processed_root = resolve_processed_root(proj)

    edge_tasks = [
        ("raw_cn", "edge_transductive", processed_root / "tasks" / "raw_cn_edge_assignee_ipc_transductive.pt"),
        ("raw_cn", "edge_inductive", processed_root / "tasks" / "raw_cn_edge_assignee_ipc_inductive.pt"),
        ("raw_cn", "edge_patent_ipc", processed_root / "tasks" / "raw_cn_edge_patent_ipc.pt"),
        ("raw_us", "edge_transductive", processed_root / "tasks" / "raw_us_edge_assignee_ipc_transductive.pt"),
        ("raw_us", "edge_inductive", processed_root / "tasks" / "raw_us_edge_assignee_ipc_inductive.pt"),
        ("raw_us", "edge_patent_ipc", processed_root / "tasks" / "raw_us_edge_patent_ipc.pt"),
        ("arxiv", "edge_transductive", processed_root / "paper_tasks" / "arxiv_edge_transductive.pt"),
        ("arxiv", "edge_inductive", processed_root / "paper_tasks" / "arxiv_edge_inductive.pt"),
        ("dblp", "edge_transductive", processed_root / "paper_tasks" / "dblp_edge_transductive.pt"),
        ("dblp", "edge_inductive", processed_root / "paper_tasks" / "dblp_edge_inductive.pt"),
        ("amazon_movies", "edge_transductive", processed_root / "paper_tasks" / "amazon_movies_edge_transductive.pt"),
        ("amazon_movies", "edge_inductive", processed_root / "paper_tasks" / "amazon_movies_edge_inductive.pt"),
        ("aminer_paper", "edge_transductive", processed_root / "paper_tasks" / "aminer_paper_edge_transductive.pt"),
        ("aminer_paper", "edge_inductive", processed_root / "paper_tasks" / "aminer_paper_edge_inductive.pt"),
    ]
    node_tasks = [
        ("raw_cn", "node_multilabel", processed_root / "tasks" / "raw_cn_node_patent_multilabel.pt"),
        ("raw_us", "node_multilabel", processed_root / "tasks" / "raw_us_node_patent_multilabel.pt"),
        ("arxiv", "node_multilabel", processed_root / "paper_tasks" / "arxiv_node_multilabel.pt"),
        ("dblp", "node_multilabel", processed_root / "paper_tasks" / "dblp_node_multilabel.pt"),
        ("amazon_movies", "node_multilabel", processed_root / "paper_tasks" / "amazon_movies_node_multilabel.pt"),
        ("aminer_paper", "node_multilabel", processed_root / "paper_tasks" / "aminer_paper_node_multilabel.pt"),
    ]

    edge_tasks = [x for x in edge_tasks if x[2].exists()]
    node_tasks = [x for x in node_tasks if x[2].exists()]
    if not args.include_single_type_datasets:
        edge_skipped = [x for x in edge_tasks if not task_is_hetero_main_ready(x[2])]
        node_skipped = [x for x in node_tasks if not task_is_hetero_main_ready(x[2])]
        edge_tasks = [x for x in edge_tasks if task_is_hetero_main_ready(x[2])]
        node_tasks = [x for x in node_tasks if task_is_hetero_main_ready(x[2])]
        if edge_skipped or node_skipped:
            skipped = edge_skipped + node_skipped
            print(
                "[scheduler] skip single-type datasets: "
                + ", ".join(sorted({f"{x[0]}::{x[1]}" for x in skipped}))
            )
    selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
    if selected:
        edge_tasks = [x for x in edge_tasks if x[0] in selected]
        node_tasks = [x for x in node_tasks if x[0] in selected]

    gpus = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if not gpus:
        gpus = [0, 1, 2, 3]

    tgn_dir = repo / "baselines" / "src" / "tgn-master" / "tgn-master"
    cdgp_dir = repo / "baselines" / "src" / "cdgp" / "CDGP"

    # Prepare adapters for external baselines.
    for ds, ttype, p in edge_tasks:
        task_name = p.stem
        adapter_dir = proj / "data" / "interim" / "baseline_adapters" / task_name
        cmd = [
            args.python_bin,
            str(proj / "scripts" / "datasets" / "export_baseline_formats.py"),
            "--input",
            str(p),
            "--name",
            task_name,
            "--out-dir",
            str(adapter_dir),
        ]
        rc = run_sync(cmd, cwd=proj, env=base_env)
        if rc != 0:
            raise RuntimeError(f"failed exporting baseline format for {task_name}")

        if args.include_tgn_external and tgn_dir.exists():
            (tgn_dir / "data").mkdir(parents=True, exist_ok=True)
            for fname in [
                f"{task_name}.csv",
                f"ml_{task_name}.csv",
                f"ml_{task_name}.npy",
                f"ml_{task_name}_node.npy",
            ]:
                src = adapter_dir / "tgn" / fname
                if src.exists():
                    shutil.copy2(src, tgn_dir / "data" / fname)

        if args.include_cdgp and cdgp_dir.exists():
            cmd = [
                args.python_bin,
                str(proj / "scripts" / "baselines" / "prepare_cdgp_adapter.py"),
                "--input",
                str(p),
                "--out-dir",
                str(cdgp_dir / "data" / task_name),
                "--name",
                task_name,
            ]
            rc = run_sync(cmd, cwd=proj, env=base_env)
            if rc != 0:
                raise RuntimeError(f"failed preparing cdgp adapter for {task_name}")

    jobs = []

    def add_job(name, family, dataset, task_type, data_path, cmd, cwd=None, out_path=None):
        jid = len(jobs)
        gpu = gpus[jid % len(gpus)]
        jobs.append(
            {
                "name": name,
                "family": family,
                "dataset": dataset,
                "task_type": task_type,
                "data_path": str(data_path),
                "gpu": gpu,
                "cmd": cmd,
                "cwd": str(cwd) if cwd else str(proj),
                "out_path": str(out_path) if out_path else "",
                "log_path": str(log_root / f"{name}.log"),
            }
        )

    for ds, ttype, p in edge_tasks:
        task_name = p.stem
        tgb2_out = metric_root / f"tgb2_custom__{task_name}.csv"
        add_job(
            name=f"tgb2_custom__{task_name}",
            family="tgb2_custom",
            dataset=ds,
            task_type=ttype,
            data_path=p,
            out_path=tgb2_out,
            cmd=[
                args.python_bin,
                str(proj / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(p),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--gpu-id",
                "{gpu}",
                "--eval-num-negs",
                str(args.eval_num_negs),
                "--out",
                str(tgb2_out),
            ],
        )

        tgb2_edgetype_out = metric_root / f"tgb2_custom_tgn_edgetype__{task_name}.csv"
        add_job(
            name=f"tgb2_custom_tgn_edgetype__{task_name}",
            family="tgb2_custom",
            dataset=ds,
            task_type=ttype,
            data_path=p,
            out_path=tgb2_edgetype_out,
            cmd=[
                args.python_bin,
                str(proj / "scripts" / "baselines" / "run_tgb2_custom_smoke.py"),
                "--data",
                str(p),
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
                str(tgb2_edgetype_out),
            ],
        )

        sthn_out = metric_root / f"sthn__{task_name}.csv"
        add_job(
            name=f"sthn__{task_name}",
            family="sthn",
            dataset=ds,
            task_type=ttype,
            data_path=p,
            out_path=sthn_out,
            cmd=[
                args.python_bin,
                str(proj / "scripts" / "baselines" / "run_tgb2_official_sthn_smoke.py"),
                "--data",
                str(p),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--gpu-id",
                "{gpu}",
                "--dataset-name",
                official_sthn_dataset_name(ds),
                "--out",
                str(sthn_out),
            ],
        )

        dyglib_out = metric_root / f"dyglib_custom__{task_name}.csv"
        add_job(
            name=f"dyglib_custom__{task_name}",
            family="dyglib_custom",
            dataset=ds,
            task_type=ttype,
            data_path=p,
            out_path=dyglib_out,
            cmd=[
                args.python_bin,
                str(proj / "scripts" / "baselines" / "run_dyglib_custom_smoke.py"),
                "--data",
                str(p),
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
                str(dyglib_out),
            ],
        )

        zoo_out = metric_root / f"dyglib_model_zoo__{task_name}.csv"
        add_job(
            name=f"dyglib_model_zoo__{task_name}",
            family="dyglib_model_zoo",
            dataset=ds,
            task_type=ttype,
            data_path=p,
            out_path=zoo_out,
            cmd=[
                args.python_bin,
                str(proj / "scripts" / "baselines" / "run_dyglib_model_zoo_smoke.py"),
                "--data",
                str(p),
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
                str(zoo_out),
            ],
        )

        if args.include_tgn_external and tgn_dir.exists():
            add_job(
                name=f"tgn_external__{task_name}",
                family="tgn_external",
                dataset=ds,
                task_type=ttype,
                data_path=p,
                cwd=tgn_dir,
                out_path=log_root / f"tgn_external__{task_name}.log",
                cmd=[
                    args.python_bin,
                    "train_self_supervised.py",
                    "--data",
                    task_name,
                    "--n_epoch",
                    str(args.tgn_epochs),
                    "--bs",
                    str(args.batch_size),
                    "--n_runs",
                    "1",
                    "--n_degree",
                    "10",
                    "--gpu",
                    "0",
                ],
            )

        if args.include_cdgp and cdgp_dir.exists():
            add_job(
                name=f"cdgp__{task_name}",
                family="cdgp",
                dataset=ds,
                task_type=ttype,
                data_path=p,
                cwd=cdgp_dir,
                out_path=log_root / f"cdgp__{task_name}.log",
                cmd=[
                    args.python_bin,
                    "train.py",
                    "--data",
                    task_name,
                    "--n_epoch",
                    str(args.cdgp_epochs),
                    "--n_runs",
                    "1",
                    "--bs",
                    str(args.batch_size),
                    "--patience",
                    "10",
                    "--gpu",
                    "0",
                ],
            )

    for ds, ttype, p in node_tasks:
        task_name = p.stem
        zoo_out = metric_root / f"dyglib_model_zoo__{task_name}.csv"
        add_job(
            name=f"dyglib_model_zoo__{task_name}",
            family="dyglib_model_zoo",
            dataset=ds,
            task_type=ttype,
            data_path=p,
            out_path=zoo_out,
            cmd=[
                args.python_bin,
                str(proj / "scripts" / "baselines" / "run_dyglib_model_zoo_smoke.py"),
                "--data",
                str(p),
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
                str(zoo_out),
            ],
        )

    print(f"[scheduler] total_jobs={len(jobs)} edge_tasks={len(edge_tasks)} node_tasks={len(node_tasks)}")

    queue = list(jobs)
    running = []
    finished = []

    while queue or running:
        while queue and len(running) < args.max_parallel:
            pick = None
            for i, cand in enumerate(queue):
                if cand["family"] == "sthn" and any(it["job"]["family"] == "sthn" for it in running):
                    continue
                pick = i
                break
            if pick is None:
                break
            job = queue.pop(pick)
            gpu = int(job["gpu"])
            cmd = [str(x).replace("{gpu}", str(gpu)) for x in job["cmd"]]
            log_path = Path(job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("w", encoding="utf-8")

            env = base_env.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            print(f"[scheduler] launch {job['name']} gpu={gpu}")
            p = subprocess.Popen(cmd, cwd=job["cwd"], env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append({"job": job, "proc": p, "log_f": log_f, "start": time.time(), "cmd": cmd})

        time.sleep(max(1, args.poll_sec))

        nr = []
        for it in running:
            rc = it["proc"].poll()
            if rc is None:
                nr.append(it)
                continue
            it["log_f"].close()
            elapsed = float(time.time() - it["start"])
            print(f"[scheduler] done {it['job']['name']} rc={rc} elapsed={elapsed:.1f}s")
            finished.append(
                {
                    "name": it["job"]["name"],
                    "family": it["job"]["family"],
                    "dataset": it["job"]["dataset"],
                    "task_type": it["job"]["task_type"],
                    "data_path": it["job"]["data_path"],
                    "gpu": it["job"]["gpu"],
                    "return_code": int(rc),
                    "elapsed_sec": elapsed,
                    "log_path": it["job"]["log_path"],
                    "out_path": it["job"]["out_path"],
                    "cmd": " ".join(it["cmd"]),
                }
            )
        running = nr

        if running:
            print(f"[scheduler] active={len(running)} queued={len(queue)}")
            for it in running:
                tail = tail_last_line(Path(it["job"]["log_path"]))
                if tail:
                    print(f"  - {it['job']['name']}: {tail}")

    manifest = out_root / "run_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "family",
                "dataset",
                "task_type",
                "data_path",
                "gpu",
                "return_code",
                "elapsed_sec",
                "log_path",
                "out_path",
                "cmd",
            ],
        )
        w.writeheader()
        for r in finished:
            w.writerow(r)
    print(f"[scheduler] manifest saved: {manifest}")

    metric_rows = []
    for r in finished:
        fam = r["family"]
        task = r["name"].split("__", 1)[-1]
        dataset = r["dataset"]
        task_type = r["task_type"]
        elapsed = float(r["elapsed_sec"])
        out_path = Path(r["out_path"]) if r["out_path"] else None
        log_path = Path(r["log_path"])

        if fam in {"tgb2_custom", "dyglib_custom", "sthn"} and out_path is not None:
            metric_rows.extend(parse_split_csv(out_path, fam, task, dataset, task_type, elapsed))
        elif fam == "dyglib_model_zoo" and out_path is not None:
            metric_rows.extend(parse_model_zoo_csv(out_path, fam, task, dataset, task_type))
        elif fam == "tgn_external":
            metric_rows.extend(parse_tgn_external_log(log_path, fam, task, dataset, task_type, elapsed))
        elif fam == "cdgp":
            metric_rows.extend(parse_cdgp_log(log_path, fam, task, dataset, task_type, elapsed))

    detail_csv = out_root / "metrics_long.csv"
    pd.DataFrame(metric_rows).to_csv(detail_csv, index=False)
    print(f"[scheduler] metrics saved: {detail_csv}")

    # A compact summary for quick paper updates.
    if metric_rows:
        df = pd.DataFrame(metric_rows)
        sum_rows = []
        for (fam, ds, ttype, model), g in df.groupby(["family", "dataset", "task_type", "model"], dropna=False):
            row = {
                "family": fam,
                "dataset": ds,
                "task_type": ttype,
                "model": model,
                "best_test_mrr": pd.to_numeric(g.get("test_mrr"), errors="coerce").max(),
                "best_test_auc": pd.to_numeric(g.get("test_auc"), errors="coerce").max(),
                "best_test_ap": pd.to_numeric(g.get("test_ap"), errors="coerce").max(),
                "best_node_ml_micro_f1": pd.to_numeric(g.get("node_ml_micro_f1"), errors="coerce").max(),
                "best_node_ml_macro_f1": pd.to_numeric(g.get("node_ml_macro_f1"), errors="coerce").max(),
                "best_node_ml_precision_at_1": pd.to_numeric(g.get("node_ml_precision_at_1"), errors="coerce").max(),
                "best_node_ml_precision_at_3": pd.to_numeric(g.get("node_ml_precision_at_3"), errors="coerce").max(),
                "best_node_ml_precision_at_5": pd.to_numeric(g.get("node_ml_precision_at_5"), errors="coerce").max(),
                "best_node_ml_recall_at_1": pd.to_numeric(g.get("node_ml_recall_at_1"), errors="coerce").max(),
                "best_node_ml_recall_at_3": pd.to_numeric(g.get("node_ml_recall_at_3"), errors="coerce").max(),
                "best_node_ml_recall_at_5": pd.to_numeric(g.get("node_ml_recall_at_5"), errors="coerce").max(),
                "best_node_ml_ndcg_at_1": pd.to_numeric(g.get("node_ml_ndcg_at_1"), errors="coerce").max(),
                "best_node_ml_ndcg_at_3": pd.to_numeric(g.get("node_ml_ndcg_at_3"), errors="coerce").max(),
                "best_node_ml_ndcg_at_5": pd.to_numeric(g.get("node_ml_ndcg_at_5"), errors="coerce").max(),
                "mean_avg_epoch_sec": pd.to_numeric(g.get("avg_epoch_sec"), errors="coerce").mean(),
                "mean_total_params": pd.to_numeric(g.get("total_params"), errors="coerce").mean(),
            }
            sum_rows.append(row)
        summary_csv = out_root / "summary.csv"
        pd.DataFrame(sum_rows).sort_values(["dataset", "task_type", "family", "model"]).to_csv(summary_csv, index=False)
        print(f"[scheduler] summary saved: {summary_csv}")


if __name__ == "__main__":
    main()
