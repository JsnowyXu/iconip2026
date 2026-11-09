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


def _default_boost_configs() -> List[Dict]:
    return [
        {"name": "boost_base", "extra": []},
        {
            "name": "boost_relw_light",
            "extra": ["--relation-loss-weights", "patent_ipc=2.0,assignee_ipc=0.7,patent_assignee=0.5"],
        },
        {
            "name": "boost_relw_strong",
            "extra": ["--relation-loss-weights", "patent_ipc=3.0,assignee_ipc=0.4,patent_assignee=0.2"],
        },
        {
            "name": "boost_relw_sparse",
            "extra": [
                "--relation-loss-weights",
                "patent_ipc=2.5,assignee_ipc=0.5,patent_assignee=0.3",
                "--community-assign-temperature",
                "0.7",
                "--community-assign-topk",
                "8",
                "--lambda-community-entropy",
                "0.01",
            ],
        },
        {
            "name": "boost_relw_sparse_strong",
            "extra": [
                "--relation-loss-weights",
                "patent_ipc=3.0,assignee_ipc=0.35,patent_assignee=0.15",
                "--community-assign-temperature",
                "0.6",
                "--community-assign-topk",
                "8",
                "--lambda-community-entropy",
                "0.02",
            ],
        },
    ]


def _load_metrics(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    test = obj.get("test", {}) if isinstance(obj, dict) else {}
    node = obj.get("node_eval", {}) if isinstance(obj, dict) else {}
    return {
        "test_mrr": test.get("mrr"),
        "test_auc": test.get("auc"),
        "test_ap": test.get("ap"),
        "node_ml_micro_f1": node.get("node_ml_micro_f1"),
        "best_epoch": obj.get("best_epoch"),
        "stopped_epoch": obj.get("stopped_epoch"),
        "avg_epoch_time_sec": obj.get("avg_epoch_time_sec"),
    }


def _tail(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return ""
            n = min(size, 4096)
            f.seek(-n, os.SEEK_END)
            txt = f.read().decode("utf-8", errors="ignore")
        lines = [x.strip() for x in txt.splitlines() if x.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser("Cross-relation grid search for patent->IPC")
    ap.add_argument("--project-root", type=str, default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--python-bin", type=str, default="python")
    ap.add_argument("--datasets", type=str, default="raw_cn,raw_us")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--gpu-id", type=int, default=-1)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--poll-sec", type=int, default=10)
    ap.add_argument("--out-root", type=str, default="")
    args = ap.parse_args()

    proj = Path(args.project_root).resolve()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root.strip()
        else proj / "results" / "iter_opt" / "cross_relation_grid_search"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{proj / 'src'}:{env.get('PYTHONPATH', '')}"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MKL_THREADING_LAYER"] = "GNU"
    if int(args.gpu_id) < 0:
        env["CUDA_VISIBLE_DEVICES"] = ""

    queue = []
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        base_data = proj / "data" / "processed" / "tasks" / f"{ds}_edge_patent_ipc.pt"
        boost_data = proj / "data" / "processed" / "tasks" / f"{ds}_edge_patent_ipc_plus_assignee_ipc.pt"
        if not base_data.exists() or not boost_data.exists():
            continue
        queue.append(
            {
                "dataset": ds,
                "variant": "base",
                "data": base_data,
                "extra": [],
                "run_name": f"{ds}__base",
            }
        )
        for cfg in _default_boost_configs():
            queue.append(
                {
                    "dataset": ds,
                    "variant": cfg["name"],
                    "data": boost_data,
                    "extra": cfg["extra"],
                    "run_name": f"{ds}__{cfg['name']}",
                }
            )

    running = []
    finished = []

    while queue or running:
        while queue and len(running) < max(1, int(args.max_parallel)):
            job = queue.pop(0)
            out_dir = out_root / job["run_name"]
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_root / "logs" / f"{job['run_name']}.log"
            log_f = log_path.open("w", encoding="utf-8")
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
                "--ml-threshold-search",
            ] + job["extra"]
            if int(args.gpu_id) >= 0:
                cmd += ["--gpu-id", str(int(args.gpu_id))]
            print("[run]", " ".join(cmd))
            p = subprocess.Popen(cmd, cwd=str(proj), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append(
                {
                    "proc": p,
                    "job": job,
                    "out_dir": out_dir,
                    "log_path": log_path,
                    "log_f": log_f,
                    "start": time.time(),
                }
            )

        time.sleep(max(1, int(args.poll_sec)))
        keep = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                keep.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - item["start"]
            met = _load_metrics(item["out_dir"] / "metrics.json")
            row = {
                "dataset": item["job"]["dataset"],
                "variant": item["job"]["variant"],
                "data": str(item["job"]["data"]),
                "out_dir": str(item["out_dir"]),
                "log_path": str(item["log_path"]),
                "return_code": int(rc),
                "elapsed_sec": float(elapsed),
                "extra_args": " ".join(item["job"]["extra"]),
            }
            row.update(met)
            finished.append(row)
            print(f"[done] {item['job']['run_name']} rc={rc} elapsed={elapsed:.1f}s")
        running = keep

        if running:
            print(f"[poll] active={len(running)} queued={len(queue)}")
            for item in running:
                tail = _tail(item["log_path"])
                if tail:
                    print(f"  - {item['job']['run_name']}: {tail}")

    manifest_csv = out_root / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "variant",
                "data",
                "out_dir",
                "log_path",
                "return_code",
                "elapsed_sec",
                "extra_args",
                "test_mrr",
                "test_auc",
                "test_ap",
                "node_ml_micro_f1",
                "best_epoch",
                "stopped_epoch",
                "avg_epoch_time_sec",
            ],
        )
        writer.writeheader()
        for row in finished:
            writer.writerow(row)

    df = pd.DataFrame(finished)
    summary_csv = out_root / "summary.csv"
    df.to_csv(summary_csv, index=False)

    rec_rows = []
    delta_rows = []
    if not df.empty:
        ok = df[df["return_code"] == 0].copy()
        for ds, g in ok.groupby("dataset"):
            base = g[g["variant"] == "base"]
            if base.empty:
                continue
            b = base.iloc[0]
            cand = g[g["variant"] != "base"].copy()
            if cand.empty:
                continue
            cand = cand.sort_values(["test_mrr", "test_ap", "test_auc"], ascending=[False, False, False])
            best = cand.iloc[0]
            rec_rows.append(
                {
                    "dataset": ds,
                    "base_variant": "base",
                    "best_boost_variant": best["variant"],
                    "base_mrr": float(b["test_mrr"]),
                    "best_boost_mrr": float(best["test_mrr"]),
                    "delta_mrr": float(best["test_mrr"]) - float(b["test_mrr"]),
                    "base_auc": float(b["test_auc"]),
                    "best_boost_auc": float(best["test_auc"]),
                    "delta_auc": float(best["test_auc"]) - float(b["test_auc"]),
                    "base_ap": float(b["test_ap"]),
                    "best_boost_ap": float(best["test_ap"]),
                    "delta_ap": float(best["test_ap"]) - float(b["test_ap"]),
                    "boost_dominates_base": bool(
                        float(best["test_mrr"]) >= float(b["test_mrr"])
                        and float(best["test_auc"]) >= float(b["test_auc"])
                        and float(best["test_ap"]) >= float(b["test_ap"])
                    ),
                }
            )
            for _, r in cand.iterrows():
                delta_rows.append(
                    {
                        "dataset": ds,
                        "variant": r["variant"],
                        "delta_mrr": float(r["test_mrr"]) - float(b["test_mrr"]),
                        "delta_auc": float(r["test_auc"]) - float(b["test_auc"]),
                        "delta_ap": float(r["test_ap"]) - float(b["test_ap"]),
                    }
                )

    rec_df = pd.DataFrame(rec_rows)
    delta_df = pd.DataFrame(delta_rows)
    rec_csv = out_root / "recommended.csv"
    delta_csv = out_root / "delta_vs_base.csv"
    rec_df.to_csv(rec_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)

    md_lines = ["# Cross-Relation Grid Search", ""]
    md_lines.append(f"- total_runs: {len(df)}")
    md_lines.append(f"- success_runs: {int((df['return_code'] == 0).sum()) if not df.empty else 0}")
    md_lines.append("")
    if not df.empty:
        md_lines.append("## Summary")
        md_lines.append("")
        md_lines.append(df.to_markdown(index=False))
        md_lines.append("")
    if not rec_df.empty:
        md_lines.append("## Recommended Boost Configs")
        md_lines.append("")
        md_lines.append(rec_df.to_markdown(index=False))
        md_lines.append("")
    if not delta_df.empty:
        md_lines.append("## Delta vs Base")
        md_lines.append("")
        md_lines.append(delta_df.to_markdown(index=False))
        md_lines.append("")
    (out_root / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(manifest_csv)
    print(summary_csv)
    print(rec_csv)
    print(delta_csv)


if __name__ == "__main__":
    main()
