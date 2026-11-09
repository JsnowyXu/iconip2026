#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd


def dataset_map(project_root: Path) -> Dict[str, Path]:
    return {
        "raw_cn": project_root / "data" / "processed" / "tasks" / "raw_cn_node_patent_multilabel.pt",
        "raw_us": project_root / "data" / "processed" / "tasks" / "raw_us_node_patent_multilabel.pt",
        "arxiv": project_root / "data" / "processed" / "paper_tasks" / "arxiv_node_multilabel.pt",
        "dblp": project_root / "data" / "processed" / "paper_tasks" / "dblp_node_multilabel.pt",
        "amazon_movies": project_root / "data" / "processed" / "paper_tasks" / "amazon_movies_node_multilabel.pt",
        "aminer_paper": project_root / "data" / "processed" / "paper_tasks" / "aminer_paper_node_multilabel.pt",
    }


def config_grid() -> List[Dict]:
    return [
        {"name": "base", "extra": []},
        {"name": "thr", "extra": ["--ml-threshold-search"]},
        {"name": "thr_pos", "extra": ["--ml-threshold-search", "--ml-pos-weight"]},
        {
            "name": "thr_pos_l03",
            "extra": ["--ml-threshold-search", "--ml-pos-weight", "--lambda-ml", "0.3"],
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
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--python-bin", type=str, default="python")
    ap.add_argument("--datasets", type=str, default="raw_cn,raw_us,arxiv,dblp,amazon_movies")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--poll-sec", type=int, default=15)
    ap.add_argument("--threshold-candidates", type=int, default=21)
    ap.add_argument("--ml-calib-max-nodes", type=int, default=0)
    ap.add_argument("--node-eval-max-nodes", type=int, default=0)
    ap.add_argument("--auc-drop-tol", type=float, default=0.0)
    ap.add_argument("--ap-drop-tol", type=float, default=0.0)
    ap.add_argument("--out-root", type=str, default="")
    ap.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU ids. Empty keeps the original CPU-only behavior.")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root.strip()
        else project_root / "results" / "grid_search_all_node_multilabel"
    )
    log_root = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    dmap = dataset_map(project_root)
    selected = [x.strip() for x in args.datasets.split(",") if x.strip()]
    selected = [x for x in selected if x in dmap]
    if not selected:
        raise SystemExit("no valid datasets selected")

    jobs = []
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    for ds in selected:
        for cfg in config_grid():
            jobs.append(
                {
                    "dataset": ds,
                    "config": cfg["name"],
                    "data": dmap[ds],
                    "extra": cfg["extra"],
                    "run_name": f"{ds}__{cfg['name']}",
                    "gpu": gpu_ids[len(jobs) % len(gpu_ids)] if gpu_ids else None,
                }
            )

    running = []
    finished = []
    queue = jobs[:]
    print(f"[grid] total_jobs={len(queue)} max_parallel={args.max_parallel}")

    while queue or running:
        while queue and len(running) < args.max_parallel:
            job = queue.pop(0)
            if not job["data"].exists():
                finished.append(
                    {
                        "run_name": job["run_name"],
                        "dataset": job["dataset"],
                        "config": job["config"],
                        "data": str(job["data"]),
                        "return_code": 99,
                        "elapsed_sec": 0.0,
                        "out_dir": "",
                        "log_path": "",
                        "extra": " ".join(job["extra"]),
                    }
                )
                print(f"[grid] skip {job['run_name']}: missing data")
                continue

            out_dir = out_root / job["run_name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_root / f"{job['run_name']}.log"
            log_f = log_path.open("w", encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{project_root / 'src'}:{env.get('PYTHONPATH', '')}"
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            if job["gpu"] is None:
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(int(job["gpu"]))

            cmd = [
                args.python_bin,
                "-m",
                "dhc.train.run_train",
                "--data",
                str(job["data"]),
                "--out-dir",
                str(out_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--ml-threshold-num-candidates",
                str(args.threshold_candidates),
                "--ml-calib-max-nodes",
                str(args.ml_calib_max_nodes),
                "--node-eval-max-nodes",
                str(args.node_eval_max_nodes),
            ] + job["extra"]
            if job["gpu"] is not None:
                cmd += ["--gpu-id", "0"]

            print(f"[grid] launch {job['run_name']}: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            running.append(
                {
                    "job": job,
                    "proc": proc,
                    "log_f": log_f,
                    "log_path": log_path,
                    "out_dir": out_dir,
                    "start": time.time(),
                }
            )

        time.sleep(max(1, args.poll_sec))
        still_running = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - item["start"]
            job = item["job"]
            print(f"[grid] done {job['run_name']} rc={rc} elapsed={elapsed:.1f}s")
            finished.append(
                {
                    "run_name": job["run_name"],
                    "dataset": job["dataset"],
                    "config": job["config"],
                    "data": str(job["data"]),
                    "return_code": int(rc),
                    "elapsed_sec": float(elapsed),
                    "out_dir": str(item["out_dir"]),
                    "log_path": str(item["log_path"]),
                    "extra": " ".join(job["extra"]),
                }
            )
        running = still_running

        if running:
            print(f"[grid] active={len(running)} queued={len(queue)}")
            for item in running:
                tail = tail_last_line(item["log_path"])
                if tail:
                    print(f"  - {item['job']['run_name']}: {tail}")

    manifest_path = out_root / "run_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "dataset",
                "config",
                "data",
                "return_code",
                "elapsed_sec",
                "out_dir",
                "log_path",
                "extra",
            ],
        )
        writer.writeheader()
        for row in finished:
            writer.writerow(row)
    print(f"[grid] manifest saved: {manifest_path}")

    rows = []
    for row in finished:
        out_dir = Path(row["out_dir"]) if row["out_dir"] else None
        metrics_path = out_dir / "metrics.json" if out_dir is not None else None
        metrics = load_metrics(metrics_path) if metrics_path is not None else {}
        node_eval = metrics.get("node_eval", {}) if isinstance(metrics, dict) else {}
        calib = metrics.get("multilabel_threshold_calibration", {}) if isinstance(metrics, dict) else {}

        rows.append(
            {
                **row,
                "status": "ok" if row["return_code"] == 0 and metrics else "failed",
                "best_epoch": metrics.get("best_epoch") if metrics else None,
                "stopped_epoch": metrics.get("stopped_epoch") if metrics else None,
                "test_mrr": metrics.get("test", {}).get("mrr") if metrics else None,
                "test_auc": metrics.get("test", {}).get("auc") if metrics else None,
                "test_ap": metrics.get("test", {}).get("ap") if metrics else None,
                "node_ml_micro_f1": node_eval.get("node_ml_micro_f1"),
                "node_ml_map": node_eval.get("node_ml_map"),
                "node_ml_threshold": node_eval.get("node_ml_threshold"),
                "calib_micro_f1": calib.get("calib_micro_f1"),
                "calib_nodes": calib.get("calib_nodes"),
                "num_threshold_candidates": calib.get("num_threshold_candidates"),
                "avg_epoch_time_sec": metrics.get("avg_epoch_time_sec") if metrics else None,
            }
        )

    df = pd.DataFrame(rows)
    summary_csv = out_root / "summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"[grid] summary saved: {summary_csv}")

    ok_df = df[df["status"] == "ok"].copy()
    if ok_df.empty:
        print("[grid] no successful runs, skip recommendation")
        return

    rec_rows = []
    for ds, g in ok_df.groupby("dataset"):
        base = g[g["config"] == "base"].copy()
        if base.empty:
            base_auc = float(g["test_auc"].max())
            base_ap = float(g["test_ap"].max())
        else:
            base_auc = float(base.iloc[0]["test_auc"])
            base_ap = float(base.iloc[0]["test_ap"])

        cand = g[
            (g["test_auc"] >= base_auc - args.auc_drop_tol)
            & (g["test_ap"] >= base_ap - args.ap_drop_tol)
        ].copy()
        if cand.empty:
            cand = g.copy()

        cand = cand.sort_values(
            ["node_ml_micro_f1", "test_ap", "test_auc", "test_mrr"],
            ascending=[False, False, False, False],
        )
        best = cand.iloc[0]
        rec_rows.append(
            {
                "dataset": ds,
                "recommended_run": best["run_name"],
                "recommended_config": best["config"],
                "base_auc": base_auc,
                "base_ap": base_ap,
                "best_auc": float(best["test_auc"]),
                "best_ap": float(best["test_ap"]),
                "best_node_ml_micro_f1": float(best["node_ml_micro_f1"]),
                "best_node_ml_threshold": float(best["node_ml_threshold"]),
            }
        )

    rec_df = pd.DataFrame(rec_rows).sort_values("dataset")
    rec_csv = out_root / "recommended_configs.csv"
    rec_df.to_csv(rec_csv, index=False)
    print(f"[grid] recommendation saved: {rec_csv}")

    md_lines = [
        "# Multi-dataset Grid Search Summary",
        "",
        f"- jobs: {len(df)}",
        f"- successful: {int((df['status'] == 'ok').sum())}",
        f"- datasets: {', '.join(selected)}",
        "",
        "## Recommended Configs",
        "",
        rec_df.to_markdown(index=False),
        "",
        "## Top Runs",
        "",
    ]
    top_df = ok_df.sort_values(
        ["dataset", "node_ml_micro_f1", "test_ap", "test_auc"],
        ascending=[True, False, False, False],
    ).groupby("dataset").head(3)
    md_lines.append(top_df.to_markdown(index=False))
    summary_md = out_root / "summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[grid] markdown saved: {summary_md}")


if __name__ == "__main__":
    main()
