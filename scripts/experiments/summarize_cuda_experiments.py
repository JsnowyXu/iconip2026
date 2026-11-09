#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=str)
    ap.add_argument("--out-csv", required=True, type=str)
    ap.add_argument("--out-md", required=True, type=str)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    rows = []
    for _, r in manifest.iterrows():
        out_dir = Path(r["out_dir"])
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            rows.append(
                {
                    "name": r["name"],
                    "return_code": int(r["return_code"]),
                    "status": "missing_metrics",
                }
            )
            continue

        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        data_path = Path(r["data"])
        train_edges = None
        if data_path.exists():
            try:
                obj = torch.load(data_path, map_location="cpu")
                train_edges = int(obj["split"]["train"].numel())
            except Exception:
                train_edges = None

        avg_epoch = safe_get(m, ["avg_epoch_time_sec"], None)
        throughput = None
        if train_edges is not None and avg_epoch is not None and avg_epoch > 0:
            throughput = float(train_edges / avg_epoch)

        rows.append(
            {
                "name": r["name"],
                "gpu": r["gpu"],
                "data": r["data"],
                "extra": r["extra"],
                "return_code": int(r["return_code"]),
                "elapsed_sec": float(r["elapsed_sec"]),
                "best_epoch": safe_get(m, ["best_epoch"], None),
                "stopped_epoch": safe_get(m, ["stopped_epoch"], None),
                "test_mrr": safe_get(m, ["test", "mrr"], None),
                "test_auc": safe_get(m, ["test", "auc"], None),
                "test_ap": safe_get(m, ["test", "ap"], None),
                "node_ml_micro_f1": safe_get(m, ["node_eval", "node_ml_micro_f1"], None),
                "node_ml_map": safe_get(m, ["node_eval", "node_ml_map"], None),
                "node_mc_acc": safe_get(m, ["node_eval", "node_mc_acc"], None),
                "avg_epoch_time_sec": avg_epoch,
                "train_edges": train_edges,
                "train_edges_per_sec": throughput,
                "total_params": safe_get(m, ["model_stats", "total_params"], None),
                "trainable_params": safe_get(m, ["model_stats", "trainable_params"], None),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    lines = ["# CUDA Experiment Summary", "", f"Source manifest: `{args.manifest}`", ""]
    ok = out_df[out_df["return_code"] == 0].copy()
    if len(ok) > 0:
        lines.extend(
            [
                "| name | gpu | best_epoch | stopped_epoch | test_mrr | test_auc | test_ap | node_ml_micro_f1 | avg_epoch_time_sec | train_edges_per_sec | total_params |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, rr in ok.iterrows():
            lines.append(
                f"| {rr['name']} | {rr['gpu']} | {rr['best_epoch']} | {rr['stopped_epoch']} | "
                f"{rr['test_mrr']:.4f} | {rr['test_auc']:.4f} | {rr['test_ap']:.4f} | "
                f"{(rr['node_ml_micro_f1'] if pd.notna(rr['node_ml_micro_f1']) else 0.0):.4f} | "
                f"{rr['avg_epoch_time_sec']:.2f} | "
                f"{(rr['train_edges_per_sec'] if pd.notna(rr['train_edges_per_sec']) else 0.0):.1f} | "
                f"{int(rr['total_params']) if pd.notna(rr['total_params']) else 0} |"
            )
    fail = out_df[out_df["return_code"] != 0]
    if len(fail) > 0:
        lines.extend(["", "## Failed Runs", ""])
        for _, rr in fail.iterrows():
            lines.append(f"- `{rr['name']}` rc={int(rr['return_code'])}")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_csv)
    print(out_md)
    print(out_df)


if __name__ == "__main__":
    main()
