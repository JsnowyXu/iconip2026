#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DATASETS = ["raw_cn", "raw_us", "arxiv", "dblp", "amazon_movies"]
CURRENT_NAME = "Current(dyn_hetero_community)"

EDGE_METHODS = [
    ("tgn_external", None, "tgn-master"),
    ("tgb2_custom", None, "TGB2-custom(TGN-like)"),
    ("tgb2_custom", "tgn_edgetype", "TGB2-custom(TGN_edgetype)"),
    ("sthn", "sthn", "STHN"),
    ("dyglib_custom", None, "DyGLib-custom(TGN-like)"),
    ("dyglib_model_zoo", "TGN", "DyGLib-TGN"),
    ("dyglib_model_zoo", "TGAT", "DyGLib-TGAT"),
    ("dyglib_model_zoo", "CAWN", "DyGLib-CAWN"),
    ("dyglib_model_zoo", "TCL", "DyGLib-TCL"),
    ("dyglib_model_zoo", "GraphMixer", "DyGLib-GraphMixer"),
    ("dyglib_model_zoo", "DyGFormer", "DyGLib-DyGFormer"),
    ("cdgp", "cdgp", "CDGP"),
]

NODE_METHODS = [
    ("dyglib_model_zoo", "TGN", "DyGLib-TGN"),
    ("dyglib_model_zoo", "TGAT", "DyGLib-TGAT"),
    ("dyglib_model_zoo", "CAWN", "DyGLib-CAWN"),
    ("dyglib_model_zoo", "TCL", "DyGLib-TCL"),
    ("dyglib_model_zoo", "GraphMixer", "DyGLib-GraphMixer"),
    ("dyglib_model_zoo", "DyGFormer", "DyGLib-DyGFormer"),
]


def fmt_float(val: Optional[float]) -> str:
    if val is None or pd.isna(val):
        return ""
    return f"{float(val):.4f}"


def format_edge_cell(row: Optional[pd.Series]) -> str:
    if row is None:
        return "-"
    mrr = fmt_float(row.get("test_mrr_mean"))
    auc = fmt_float(row.get("test_auc_mean"))
    ap = fmt_float(row.get("test_ap_mean"))
    if mrr and auc and ap:
        return f"MRR/AUC/AP={mrr}/{auc}/{ap}"
    if auc and ap:
        return f"AUC/AP={auc}/{ap}"
    return "-"


def format_node_cell(row: Optional[pd.Series]) -> str:
    if row is None:
        return "-"
    parts: List[str] = []
    for prefix, col in [
        ("P@1", "node_ml_precision_at_1_mean"),
        ("P@3", "node_ml_precision_at_3_mean"),
        ("P@5", "node_ml_precision_at_5_mean"),
        ("R@1", "node_ml_recall_at_1_mean"),
        ("R@3", "node_ml_recall_at_3_mean"),
        ("R@5", "node_ml_recall_at_5_mean"),
        ("nDCG@1", "node_ml_ndcg_at_1_mean"),
        ("nDCG@3", "node_ml_ndcg_at_3_mean"),
        ("nDCG@5", "node_ml_ndcg_at_5_mean"),
    ]:
        val = fmt_float(row.get(col))
        if val:
            parts.append(f"{prefix}={val}")
    return "; ".join(parts) if parts else "-"


def reduce_best(df: pd.DataFrame, value_cols: List[str], group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + value_cols)
    out_rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {k: v for k, v in zip(group_cols, keys)}
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[col] = vals.max() if vals.notna().any() else pd.NA
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def get_row(df: pd.DataFrame, filters: Dict[str, object]) -> Optional[pd.Series]:
    if df.empty:
        return None
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        mask &= df[key] == value
    sub = df[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


def build_current_rows(agg_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    full = agg_df[agg_df["variant"] == "full"].copy()
    rows = {
        "edge_transductive": {"method": CURRENT_NAME},
        "edge_inductive": {"method": CURRENT_NAME},
        "node_multilabel": {"method": CURRENT_NAME},
    }
    for dataset in DATASETS:
        rows["edge_transductive"][dataset] = format_edge_cell(
            get_row(full, {"dataset": dataset, "task_type": "edge_transductive"})
        )
        rows["edge_inductive"][dataset] = format_edge_cell(
            get_row(full, {"dataset": dataset, "task_type": "edge_inductive"})
        )
        rows["node_multilabel"][dataset] = format_node_cell(
            get_row(full, {"dataset": dataset, "task_type": "node_multilabel"})
        )
    return rows


def load_baseline_edge(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(summary_csv)
    keep = ["family", "dataset", "task_type", "model", "best_test_auc", "best_test_ap"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for col in ["best_test_auc", "best_test_ap"]:
        if col not in df.columns:
            df[col] = pd.NA
    return reduce_best(df, ["best_test_auc", "best_test_ap"], ["family", "dataset", "task_type", "model"])


def load_baseline_node(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(summary_csv)
    keep = [
        "family",
        "dataset",
        "task_type",
        "model",
        "best_test_auc",
        "best_test_ap",
        "best_node_ml_micro_f1",
        "best_node_ml_macro_f1",
        "best_node_ml_precision_at_1",
        "best_node_ml_precision_at_3",
        "best_node_ml_precision_at_5",
        "best_node_ml_recall_at_1",
        "best_node_ml_recall_at_3",
        "best_node_ml_recall_at_5",
        "best_node_ml_ndcg_at_1",
        "best_node_ml_ndcg_at_3",
        "best_node_ml_ndcg_at_5",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    for col in [
        "best_test_auc",
        "best_test_ap",
        "best_node_ml_micro_f1",
        "best_node_ml_macro_f1",
        "best_node_ml_precision_at_1",
        "best_node_ml_precision_at_3",
        "best_node_ml_precision_at_5",
        "best_node_ml_recall_at_1",
        "best_node_ml_recall_at_3",
        "best_node_ml_recall_at_5",
        "best_node_ml_ndcg_at_1",
        "best_node_ml_ndcg_at_3",
        "best_node_ml_ndcg_at_5",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.rename(
        columns={
            "best_test_auc": "test_auc_mean",
            "best_test_ap": "test_ap_mean",
            "best_node_ml_micro_f1": "node_ml_micro_f1_mean",
            "best_node_ml_macro_f1": "node_ml_macro_f1_mean",
            "best_node_ml_precision_at_1": "node_ml_precision_at_1_mean",
            "best_node_ml_precision_at_3": "node_ml_precision_at_3_mean",
            "best_node_ml_precision_at_5": "node_ml_precision_at_5_mean",
            "best_node_ml_recall_at_1": "node_ml_recall_at_1_mean",
            "best_node_ml_recall_at_3": "node_ml_recall_at_3_mean",
            "best_node_ml_recall_at_5": "node_ml_recall_at_5_mean",
            "best_node_ml_ndcg_at_1": "node_ml_ndcg_at_1_mean",
            "best_node_ml_ndcg_at_3": "node_ml_ndcg_at_3_mean",
            "best_node_ml_ndcg_at_5": "node_ml_ndcg_at_5_mean",
        }
    )
    return reduce_best(
        df,
        [
            "test_auc_mean",
            "test_ap_mean",
            "node_ml_micro_f1_mean",
            "node_ml_macro_f1_mean",
            "node_ml_precision_at_1_mean",
            "node_ml_precision_at_3_mean",
            "node_ml_precision_at_5_mean",
            "node_ml_recall_at_1_mean",
            "node_ml_recall_at_3_mean",
            "node_ml_recall_at_5_mean",
            "node_ml_ndcg_at_1_mean",
            "node_ml_ndcg_at_3_mean",
            "node_ml_ndcg_at_5_mean",
        ],
        ["family", "dataset", "task_type", "model"],
    )


def load_cdgp(metrics_csv: Path) -> pd.DataFrame:
    if not metrics_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_csv)
    if "family" not in df.columns:
        return pd.DataFrame()
    df = df[df["family"] == "cdgp"].copy()
    needed = [
        "family",
        "dataset",
        "task_type",
        "model",
        "cdgp_test_rmsle",
        "cdgp_test_male",
        "cdgp_test_mape",
    ]
    df = df[[c for c in needed if c in df.columns]].copy()
    for col in ["cdgp_test_rmsle", "cdgp_test_male", "cdgp_test_mape"]:
        if col not in df.columns:
            df[col] = pd.NA
    return reduce_best(
        df,
        ["cdgp_test_rmsle", "cdgp_test_male", "cdgp_test_mape"],
        ["family", "dataset", "task_type", "model"],
    )


def build_edge_table(current_row: Dict[str, str], edge_df: pd.DataFrame, cdgp_df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    rows = [current_row]
    for family, model, label in EDGE_METHODS:
        row = {"method": label}
        for dataset in DATASETS:
            if family == "cdgp":
                cdgp_row = get_row(cdgp_df, {"family": family, "dataset": dataset, "task_type": task_type, "model": model})
                if cdgp_row is None:
                    row[dataset] = "-"
                else:
                    rmsle = fmt_float(cdgp_row.get("cdgp_test_rmsle"))
                    male = fmt_float(cdgp_row.get("cdgp_test_male"))
                    mape = fmt_float(cdgp_row.get("cdgp_test_mape"))
                    row[dataset] = f"R/M/M={rmsle}/{male}/{mape}" if rmsle and male and mape else "-"
                continue
            filters = {
                "family": family,
                "dataset": dataset,
                "task_type": task_type,
            }
            if model is not None:
                filters["model"] = model
            base_row = get_row(edge_df, filters)
            row[dataset] = format_node_cell(base_row) if task_type == "node_multilabel" else format_edge_cell(base_row)
        rows.append(row)
    return pd.DataFrame(rows)


def build_node_table(current_row: Dict[str, str], node_df: pd.DataFrame) -> pd.DataFrame:
    rows = [current_row]
    for family, model, label in NODE_METHODS:
        row = {"method": label}
        for dataset in DATASETS:
            base_row = get_row(
                node_df,
                {
                    "family": family,
                    "dataset": dataset,
                    "task_type": "node_multilabel",
                    "model": model,
                },
            )
            row[dataset] = format_node_cell(base_row)
        rows.append(row)
    return pd.DataFrame(rows)


def prune_empty_method_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = []
    for idx, row in df.iterrows():
        if idx == 0:
            keep.append(True)
            continue
        keep.append(any(str(row.get(dataset, "-")).strip() != "-" for dataset in DATASETS))
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def write_markdown(df: pd.DataFrame, path: Path, title: str) -> None:
    lines = [f"# {title}", ""]
    try:
        lines.append(df.to_markdown(index=False))
    except Exception:
        lines.append(df.to_string(index=False))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-aggregate-csv", default="/home/xuxj/2026/dyn_hetero_community/results/full_benchmark_repeats/aggregate_mean_std.csv")
    ap.add_argument("--baseline-summary-csv", default="/home/xuxj/2026/dyn_hetero_community/results/baseline_formal_all/summary.csv")
    ap.add_argument("--baseline-metrics-csv", default="/home/xuxj/2026/dyn_hetero_community/results/baseline_formal_all/metrics_long.csv")
    ap.add_argument("--out-dir", default="/home/xuxj/2026/dyn_hetero_community/results/dataset_statistics")
    args = ap.parse_args()

    agg_df = pd.read_csv(args.full_aggregate_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_rows = build_current_rows(agg_df)
    edge_df = load_baseline_edge(Path(args.baseline_summary_csv))
    node_df = load_baseline_node(Path(args.baseline_summary_csv))
    cdgp_df = load_cdgp(Path(args.baseline_metrics_csv))

    trans_df = prune_empty_method_rows(build_edge_table(current_rows["edge_transductive"], edge_df, cdgp_df, "edge_transductive"))
    ind_df = prune_empty_method_rows(build_edge_table(current_rows["edge_inductive"], edge_df, cdgp_df, "edge_inductive"))
    node_table = prune_empty_method_rows(build_node_table(current_rows["node_multilabel"], node_df))

    specs = [
        (trans_df, out_dir / "table_transductive.csv", out_dir / "table_transductive.md", "Transductive Link Prediction"),
        (ind_df, out_dir / "table_inductive.csv", out_dir / "table_inductive.md", "Inductive Link Prediction"),
        (node_table, out_dir / "table_multilabel.csv", out_dir / "table_multilabel.md", "Node Multi-label Classification"),
    ]
    for df, csv_path, md_path, title in specs:
        df.to_csv(csv_path, index=False)
        write_markdown(df, md_path, title)
        print(csv_path)
        print(md_path)


if __name__ == "__main__":
    main()
