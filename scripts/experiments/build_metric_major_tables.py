#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EDGE_DATASETS_DEFAULT = ["raw_cn", "raw_us", "dblp", "amazon_movies", "aminer_paper", "thgl_github", "thgl_software"]
NODE_DATASETS_DEFAULT = ["raw_cn", "raw_us", "dblp", "amazon_movies", "aminer_paper", "thgl_github", "thgl_software"]

EDGE_TABLE_SPECS = [
    ("edge_transductive", "test_mrr", "transductive_mrr"),
    ("edge_transductive", "test_ap", "transductive_ap"),
    ("edge_inductive", "test_mrr", "inductive_mrr"),
    ("edge_inductive", "test_ap", "inductive_ap"),
]

NODE_TABLE_SPECS = [
    ("node_ml_precision_at_1", "node_precision_at_1"),
    ("node_ml_precision_at_3", "node_precision_at_3"),
    ("node_ml_precision_at_5", "node_precision_at_5"),
    ("node_ml_recall_at_1", "node_recall_at_1"),
    ("node_ml_recall_at_3", "node_recall_at_3"),
    ("node_ml_recall_at_5", "node_recall_at_5"),
    ("node_ml_ndcg_at_1", "node_ndcg_at_1"),
    ("node_ml_ndcg_at_3", "node_ndcg_at_3"),
    ("node_ml_ndcg_at_5", "node_ndcg_at_5"),
]


def fmt_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or pd.isna(mean):
        return "-"
    if std is None or pd.isna(std):
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} ± {float(std):.4f}"


def table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def normalize_method_name(family: str, model: Optional[str]) -> str:
    family = str(family or "").strip()
    model = str(model or "").strip()
    if family == "dhc":
        return "DHC"
    if family == "tgb2_custom":
        if model == "tgn_edgetype":
            return "TGB2-TGN_edgetype"
        return "TGB2-TGN"
    if family == "dyglib_custom":
        return "DyGLib-custom"
    if family == "dyglib_model_zoo":
        return model or "DyGLib"
    if family == "tgn_external":
        return "tgn-master"
    if family == "cdgp":
        return "CDGP"
    if family == "sthn":
        return "STHN"
    return model or family or "Unknown"


def load_dhc(dhc_csv: Path) -> pd.DataFrame:
    if not dhc_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(dhc_csv)
    df = df[df["variant"] == "full"].copy()
    if df.empty:
        return df
    df["family"] = "dhc"
    df["model"] = "DHC"
    out_rows = []
    for _, row in df.iterrows():
        item = {
            "family": "dhc",
            "model": "DHC",
            "method": "DHC",
            "dataset": row["dataset"],
            "task_type": row["task_type"],
        }
        for col in df.columns:
            if col.endswith("_mean") or col.endswith("_std"):
                item[col] = row[col]
        out_rows.append(item)
    return pd.DataFrame(out_rows)


def load_baseline_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    rows = []
    metric_cols = [
        "test_mrr",
        "test_auc",
        "test_ap",
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
    for keys, g in df.groupby(["family", "dataset", "task_type", "model"], dropna=False):
        row = {
            "family": keys[0],
            "dataset": keys[1],
            "task_type": keys[2],
            "model": keys[3],
            "method": normalize_method_name(keys[0], keys[3]),
        }
        for col in metric_cols:
            vals = pd.to_numeric(g.get(col), errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_baseline_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    rename_map = {
        "best_test_mrr": "test_mrr_mean",
        "best_test_auc": "test_auc_mean",
        "best_test_ap": "test_ap_mean",
        "best_node_ml_micro_f1": "node_ml_micro_f1_mean",
        "best_node_ml_macro_f1": "node_ml_macro_f1_mean",
    }
    df = df.rename(columns=rename_map)
    keep_rows = []
    for _, row in df.iterrows():
        item = {
            "family": row.get("family", ""),
            "dataset": row.get("dataset", ""),
            "task_type": row.get("task_type", ""),
            "model": row.get("model", ""),
            "method": normalize_method_name(row.get("family", ""), row.get("model", "")),
        }
        for col in rename_map.values():
            item[col] = row.get(col, np.nan)
            item[col.replace("_mean", "_std")] = np.nan
        keep_rows.append(item)
    return pd.DataFrame(keep_rows)


def choose_baseline_df(long_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    return long_df if not long_df.empty else summary_df


def build_table(df: pd.DataFrame, task_type: str, metric: str, datasets: List[str]) -> pd.DataFrame:
    sub = df[df["task_type"] == task_type].copy()
    methods = sorted(sub["method"].dropna().unique().tolist())
    rows = []
    for method in methods:
        row = {"method": method}
        part = sub[sub["method"] == method]
        for dataset in datasets:
            hit = part[part["dataset"] == dataset]
            if hit.empty:
                row[dataset] = "-"
                continue
            best = hit.sort_values(by=f"{metric}_mean", ascending=False, na_position="last").iloc[0]
            row[dataset] = fmt_mean_std(best.get(f"{metric}_mean"), best.get(f"{metric}_std"))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dhc-aggregate-csv", type=str, required=True)
    ap.add_argument("--baseline-long-csv", type=str, default="")
    ap.add_argument("--baseline-summary-csv", type=str, default="")
    ap.add_argument("--edge-datasets", type=str, default=",".join(EDGE_DATASETS_DEFAULT))
    ap.add_argument("--node-datasets", type=str, default=",".join(NODE_DATASETS_DEFAULT))
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_datasets = [x.strip() for x in args.edge_datasets.split(",") if x.strip()]
    node_datasets = [x.strip() for x in args.node_datasets.split(",") if x.strip()]

    dhc_df = load_dhc(Path(args.dhc_aggregate_csv).resolve())
    base_df = choose_baseline_df(
        load_baseline_long(Path(args.baseline_long_csv).resolve()) if args.baseline_long_csv.strip() else pd.DataFrame(),
        load_baseline_summary(Path(args.baseline_summary_csv).resolve()) if args.baseline_summary_csv.strip() else pd.DataFrame(),
    )
    full_df = pd.concat([df for df in [dhc_df, base_df] if not df.empty], ignore_index=True)
    if full_df.empty:
        raise RuntimeError("No available rows from DHC aggregate or baseline results.")

    summary_lines = ["# Metric-major Benchmark Tables", ""]

    for task_type, metric, stem in EDGE_TABLE_SPECS:
        table = build_table(full_df, task_type=task_type, metric=metric, datasets=edge_datasets)
        csv_path = out_dir / f"{stem}.csv"
        md_path = out_dir / f"{stem}.md"
        table.to_csv(csv_path, index=False)
        md_path.write_text(table_text(table), encoding="utf-8")
        summary_lines += [f"## {stem}", "", table_text(table), ""]

    for metric, stem in NODE_TABLE_SPECS:
        table = build_table(full_df, task_type="node_multilabel", metric=metric, datasets=node_datasets)
        csv_path = out_dir / f"{stem}.csv"
        md_path = out_dir / f"{stem}.md"
        table.to_csv(csv_path, index=False)
        md_path.write_text(table_text(table), encoding="utf-8")
        summary_lines += [f"## {stem}", "", table_text(table), ""]

    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
