#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from edge_benchmark_common import build_edge_task_specs


DATASETS = ["patent_us", "patent_in", "patent_cn", "thgl_software", "aminer_paper", "amazon_movies", "dblp"]
METHODS = ["XGBoost", "RandomForest", "LogisticRegression", "TemporalCount"]
METRICS = ["mrr", "auc", "ap"]
TASK_TYPES = ["edge_inductive", "edge_transductive"]


def fmt_metric(value) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.4f}"
    except Exception:
        return "-"


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No completed tabular ML rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def collect_rows(project_root: Path, metrics_root: Path) -> pd.DataFrame:
    specs = [s for s in build_edge_task_specs(project_root) if s.dataset in DATASETS]
    rows: List[Dict] = []
    for spec in specs:
        path = metrics_root / f"tabular_ml__{spec.task_name}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for _, r in df.iterrows():
            if str(r.get("split", "")) != "test":
                continue
            rows.append(
                {
                    "dataset": spec.dataset,
                    "task_name": spec.task_name,
                    "task_type": spec.task_type,
                    "relation_target": spec.relation_target,
                    "method": str(r.get("model", "")),
                    "mrr": r.get("mrr"),
                    "auc": r.get("auc"),
                    "ap": r.get("ap"),
                    "device": r.get("device", ""),
                    "eval_num_negs": r.get("eval_num_negs", ""),
                    "max_train_pos": r.get("max_train_pos", ""),
                    "max_eval_pos": r.get("max_eval_pos", ""),
                    "use_common_neighbors": r.get("use_common_neighbors", ""),
                    "source_path": str(path),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    method_order = {m: i for i, m in enumerate(METHODS)}
    dataset_order = {d: i for i, d in enumerate(DATASETS)}
    df["_method_order"] = df["method"].map(method_order).fillna(99)
    df["_dataset_order"] = df["dataset"].map(dataset_order).fillna(99)
    return df.sort_values(["task_type", "_method_order", "_dataset_order"]).drop(columns=["_method_order", "_dataset_order"])


def build_metric_table(df: pd.DataFrame, task_type: str, metric: str) -> pd.DataFrame:
    sub = df[df["task_type"] == task_type]
    rows = []
    for method in METHODS:
        row = {"method": method}
        part = sub[sub["method"] == method]
        for dataset in DATASETS:
            hit = part[part["dataset"] == dataset]
            row[dataset] = fmt_metric(hit.iloc[0][metric]) if not hit.empty else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser("Build tabular-only ML reports for edge tasks")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument("--campaign-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal"))
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    campaign_root = args.campaign_root.resolve()
    metrics_root = campaign_root / "baselines" / "metrics"
    reports_root = campaign_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    df = collect_rows(project_root, metrics_root)
    long_csv = reports_root / "tabular_ml_test_metrics_long.csv"
    df.to_csv(long_csv, index=False)

    summary = ["# Tabular-only ML edge task reports", ""]
    summary.append("These tables include only XGBoost, RandomForest, LogisticRegression, and TemporalCount.")
    summary.append("They are not mixed with neural or historical graph baselines.")
    summary.append("")
    for task_type in TASK_TYPES:
        for metric in METRICS:
            table = build_metric_table(df, task_type=task_type, metric=metric)
            stem = f"tabular_ml_{task_type}_{metric}"
            table.to_csv(reports_root / f"{stem}.csv", index=False)
            (reports_root / f"{stem}.md").write_text(to_markdown(table), encoding="utf-8")
            summary.extend([f"## {stem}", "", to_markdown(table), ""])
    (reports_root / "tabular_ml_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(long_csv)


if __name__ == "__main__":
    main()
