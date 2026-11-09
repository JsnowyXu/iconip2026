#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


EDGE_METRICS = ["test_mrr", "test_auc", "test_ap"]
NODE_METRICS = [
    "node_ml_auc",
    "node_ml_micro_f1",
    "node_ml_macro_f1",
    "node_ml_map",
    "node_ml_precision_at_1",
    "node_ml_precision_at_3",
    "node_ml_precision_at_5",
    "node_ml_recall_at_1",
    "node_ml_recall_at_3",
    "node_ml_recall_at_5",
    "node_ml_ndcg_at_1",
    "node_ml_ndcg_at_3",
    "node_ml_ndcg_at_5",
]


def safe_get(obj: Dict, keys: List[str], default=None):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_metrics(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--baseline-csv", type=str, default="/home/xuxj/2026/dyn_hetero_community/results/baseline_formal_all_gpu/metrics_long.csv")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    manifest = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir.strip() else manifest.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    run_df = pd.read_csv(manifest)
    rows = []
    for _, r in run_df.iterrows():
        metrics = load_metrics(Path(r["out_dir"]) / "metrics.json")
        node_eval = metrics.get("node_eval", {}) if isinstance(metrics, dict) else {}
        run_name = r.get("run_name", r.get("name", ""))
        row = {
            "run_name": run_name,
            "dataset": r["dataset"],
            "task": r["task"],
            "task_type": r["task_type"],
            "variant": r.get("variant", "full"),
            "seed": int(r.get("seed", -1)),
            "return_code": int(r["return_code"]),
            "elapsed_sec": float(r["elapsed_sec"]),
            "status": "ok" if int(r["return_code"]) == 0 and metrics else "failed",
        }
        row.update({m: safe_get(metrics, ["test", m.replace("test_", "")], np.nan) for m in EDGE_METRICS})
        row.update({m: safe_get(node_eval, [m], np.nan) for m in NODE_METRICS})
        row["best_epoch"] = safe_get(metrics, ["best_epoch"], np.nan)
        row["avg_epoch_time_sec"] = safe_get(metrics, ["avg_epoch_time_sec"], np.nan)
        rows.append(row)

    full_df = pd.DataFrame(rows)
    full_csv = out_dir / "all_runs_metrics.csv"
    full_df.to_csv(full_csv, index=False)

    ok_df = full_df[full_df["status"] == "ok"].copy()
    agg_rows = []
    metric_cols = [c for c in EDGE_METRICS + NODE_METRICS if c in ok_df.columns]
    for keys, g in ok_df.groupby(["dataset", "task", "task_type", "variant"]):
        row = {
            "dataset": keys[0],
            "task": keys[1],
            "task_type": keys[2],
            "variant": keys[3],
            "runs": int(len(g)),
        }
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 0 else np.nan
        row["avg_epoch_time_sec_mean"] = float(pd.to_numeric(g["avg_epoch_time_sec"], errors="coerce").mean())
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows).sort_values(["dataset", "task_type", "task", "variant"])
    agg_csv = out_dir / "aggregate_mean_std.csv"
    agg_df.to_csv(agg_csv, index=False)

    baseline_path = Path(args.baseline_csv)
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        baseline_best = (
            baseline_df.groupby(["dataset", "task_type"])
            .agg(
                baseline_best_auc=("test_auc", "max"),
                baseline_best_ap=("test_ap", "max"),
                baseline_best_node_ml_micro_f1=("node_ml_micro_f1", "max"),
                baseline_best_node_ml_macro_f1=("node_ml_macro_f1", "max"),
            )
            .reset_index()
        )
    else:
        baseline_best = pd.DataFrame(
            columns=[
                "dataset",
                "task_type",
                "baseline_best_auc",
                "baseline_best_ap",
                "baseline_best_node_ml_micro_f1",
                "baseline_best_node_ml_macro_f1",
            ]
        )

    full_only = agg_df[agg_df["variant"] == "full"].copy()
    compare = full_only.merge(baseline_best, on=["dataset", "task_type"], how="left")
    compare["delta_auc_vs_best_baseline"] = compare["test_auc_mean"] - compare["baseline_best_auc"]
    compare["delta_ap_vs_best_baseline"] = compare["test_ap_mean"] - compare["baseline_best_ap"]
    compare["delta_node_ml_micro_f1_vs_best_baseline"] = (
        compare["node_ml_micro_f1_mean"] - compare["baseline_best_node_ml_micro_f1"]
    )
    compare["delta_node_ml_macro_f1_vs_best_baseline"] = (
        compare["node_ml_macro_f1_mean"] - compare["baseline_best_node_ml_macro_f1"]
    )
    compare_csv = out_dir / "full_vs_baselines.csv"
    compare.to_csv(compare_csv, index=False)

    ablation = agg_df[agg_df["variant"] != "full"].copy()
    ablation_base = full_only[
        ["dataset", "task", "task_type", "test_mrr_mean", "test_auc_mean", "test_ap_mean", "node_ml_micro_f1_mean", "node_ml_map_mean"]
    ].rename(
        columns={
            "test_mrr_mean": "base_test_mrr_mean",
            "test_auc_mean": "base_test_auc_mean",
            "test_ap_mean": "base_test_ap_mean",
            "node_ml_micro_f1_mean": "base_node_ml_micro_f1_mean",
            "node_ml_map_mean": "base_node_ml_map_mean",
        }
    )
    ablation = ablation.merge(ablation_base, on=["dataset", "task", "task_type"], how="left")
    ablation["delta_test_ap"] = ablation["test_ap_mean"] - ablation["base_test_ap_mean"]
    ablation["delta_test_auc"] = ablation["test_auc_mean"] - ablation["base_test_auc_mean"]
    ablation["delta_node_ml_micro_f1"] = ablation["node_ml_micro_f1_mean"] - ablation["base_node_ml_micro_f1_mean"]
    ablation_csv = out_dir / "ablation_deltas.csv"
    ablation.to_csv(ablation_csv, index=False)

    md_lines = [
        "# Full Benchmark Summary",
        "",
        f"- manifest: `{manifest}`",
        f"- successful runs: {int((full_df['status'] == 'ok').sum())}/{len(full_df)}",
        f"- failed runs: {int((full_df['status'] != 'ok').sum())}",
        "",
        "## Full Model vs. Best Baselines",
        "",
        table_text(compare[
            [
                "dataset",
                "task",
                "task_type",
                "test_auc_mean",
                "test_auc_std",
                "baseline_best_auc",
                "delta_auc_vs_best_baseline",
                "test_ap_mean",
                "test_ap_std",
                "baseline_best_ap",
                "delta_ap_vs_best_baseline",
            ]
        ]),
        "",
        "## Node Multi-label Metrics",
        "",
        table_text(full_only[
            [
                "dataset",
                "task",
                "node_ml_micro_f1_mean",
                "node_ml_map_mean",
                "node_ml_precision_at_1_mean",
                "node_ml_precision_at_3_mean",
                "node_ml_precision_at_5_mean",
                "node_ml_recall_at_1_mean",
                "node_ml_recall_at_3_mean",
                "node_ml_recall_at_5_mean",
                "node_ml_ndcg_at_1_mean",
                "node_ml_ndcg_at_3_mean",
                "node_ml_ndcg_at_5_mean",
            ]
        ]),
        "",
        "## Ablation Deltas",
        "",
        table_text(ablation[
            ["dataset", "task", "variant", "test_ap_mean", "delta_test_ap", "test_auc_mean", "delta_test_auc", "node_ml_micro_f1_mean", "delta_node_ml_micro_f1"]
        ]),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(full_csv)
    print(agg_csv)
    print(compare_csv)
    print(ablation_csv)
    print(out_dir / "summary.md")


if __name__ == "__main__":
    main()
