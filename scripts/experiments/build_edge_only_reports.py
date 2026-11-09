#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from edge_benchmark_common import DATASET_ORDER, build_edge_task_specs


METHOD_ORDER = [
    "DHC",
    "TGNedge-type",
    "EdgeBank",
    "XGBoost",
    "RandomForest",
    "LogisticRegression",
    "TemporalCount",
    "STHN",
    "RepeatMixer",
    "MemoryModel(TGN)",
    "TGAT",
    "CAWN",
    "DyGFormer",
]


def fmt_metric(value) -> str:
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "-"


def normalize_method(family: str, model: str) -> str:
    if family == "dhc":
        return "DHC"
    if family == "tgn_edgetype":
        return "TGNedge-type"
    if family == "edgebank":
        return "EdgeBank"
    if family == "tabular_ml":
        return model or "TabularML"
    if family == "sthn":
        return "STHN"
    if family == "repeatmixer":
        return "RepeatMixer"
    if family == "repeatmixer_model_zoo":
        if model == "TGN":
            return "MemoryModel(TGN)"
        return model or "RepeatMixer-zoo"
    return model or family


def collect_main_rows(project_root: Path, main_root: Path) -> List[Dict]:
    manifest_path = main_root / "run_manifest.csv"
    manifest_lookup = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manifest_lookup[row.get("run_name", "")] = row

    rows: List[Dict] = []
    task_lookup = {spec.task_name: spec for spec in build_edge_task_specs(project_root)}
    for metrics_path in sorted(main_root.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        meta = manifest_lookup.get(run_name, {})
        task_name = meta.get("task_name", run_name.rsplit("__seed", 1)[0])
        task = task_lookup.get(task_name)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = payload.get("test", {})
        best_val = payload.get("best_val", {})
        rows.append(
            {
                "experiment_group": "main",
                "dataset": meta.get("dataset", task.dataset if task is not None else run_name.split("__")[0]),
                "task": task_name,
                "task_type": meta.get("task_type", task.task_type if task is not None else ""),
                "relation_target": meta.get("relation_target", task.relation_target if task is not None else ""),
                "method_family": "dhc",
                "method": "DHC",
                "status": "ok" if int(meta.get("return_code", 0)) == 0 else "failed",
                "val_mrr": best_val.get("val_mrr"),
                "val_auc": best_val.get("val_auc"),
                "val_ap": best_val.get("val_ap"),
                "mrr": test.get("mrr"),
                "auc": test.get("auc"),
                "ap": test.get("ap"),
                "source_path": str(metrics_path),
            }
        )
    return rows


def collect_baseline_rows(baseline_root: Path) -> List[Dict]:
    path = baseline_root / "metrics_long.csv"
    rows = []
    if path.exists():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "experiment_group": "baseline",
                    "dataset": r.get("dataset", ""),
                    "task": r.get("task_name", ""),
                    "task_type": r.get("task_type", ""),
                    "relation_target": r.get("relation_target", ""),
                    "method_family": r.get("family", ""),
                    "method": normalize_method(str(r.get("family", "")), str(r.get("model", ""))),
                    "status": r.get("status", "ok"),
                    "val_mrr": r.get("val_mrr"),
                    "val_auc": r.get("val_auc"),
                    "val_ap": r.get("val_ap"),
                    "mrr": r.get("test_mrr", r.get("mrr")),
                    "auc": r.get("test_auc", r.get("auc")),
                    "ap": r.get("test_ap", r.get("ap")),
                    "source_path": r.get("source_path", str(path)),
                }
            )
    return rows


def collect_baseline_metric_file_rows(project_root: Path, baseline_root: Path) -> List[Dict]:
    task_lookup = {spec.task_name: spec for spec in build_edge_task_specs(project_root)}
    rows: List[Dict] = []
    metric_root = baseline_root / "metrics"
    for csv_path in sorted(metric_root.glob("*.csv")):
        stem = csv_path.stem
        if "__" not in stem:
            continue
        family, task_name = stem.split("__", 1)
        task = task_lookup.get(task_name)
        if task is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "split" in df.columns:
            for _, r in df.iterrows():
                split_name = str(r.get("split", ""))
                if split_name not in {"val", "test"}:
                    continue
                rows.append(
                    {
                        "experiment_group": "baseline",
                        "dataset": task.dataset,
                        "task": task.task_name,
                        "task_type": task.task_type,
                        "relation_target": task.relation_target,
                        "method_family": family,
                        "method": normalize_method(family, str(r.get("model", family))),
                        "status": "ok",
                        "val_mrr": r.get("mrr") if split_name == "val" else None,
                        "val_auc": r.get("auc") if split_name == "val" else None,
                        "val_ap": r.get("ap") if split_name == "val" else None,
                        "mrr": r.get("mrr") if split_name == "test" else None,
                        "auc": r.get("auc") if split_name == "test" else None,
                        "ap": r.get("ap") if split_name == "test" else None,
                        "source_path": str(csv_path),
                    }
                )
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "experiment_group": "baseline",
                    "dataset": task.dataset,
                    "task": task.task_name,
                    "task_type": task.task_type,
                    "relation_target": task.relation_target,
                    "method_family": family,
                    "method": normalize_method(family, str(r.get("model", ""))),
                    "status": r.get("status", "unknown"),
                    "val_mrr": r.get("val_mrr"),
                    "val_auc": r.get("val_auc"),
                    "val_ap": r.get("val_ap"),
                    "mrr": r.get("test_mrr"),
                    "auc": r.get("test_auc"),
                    "ap": r.get("test_ap"),
                    "source_path": str(csv_path),
                }
            )
    return rows


def build_table(df: pd.DataFrame, task_type: str, metric: str, datasets: List[str]) -> pd.DataFrame:
    sub = df[df["task_type"] == task_type].copy()
    rows = []
    for method in METHOD_ORDER:
        part = sub[sub["method"] == method]
        if part.empty:
            continue
        row = {"method": method}
        for dataset in datasets:
            hit = part[part["dataset"] == dataset]
            if hit.empty:
                row[dataset] = "-"
                continue
            best = hit.sort_values(metric, ascending=False, na_position="last").iloc[0]
            row[dataset] = fmt_metric(best.get(metric))
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_暂无数据_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser("Build edge-only benchmark summary tables")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument("--campaign-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory/results/edge_only_formal"))
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    campaign_root = args.campaign_root.resolve()
    reports_root = campaign_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    rows = (
        collect_main_rows(project_root, campaign_root / "main")
        + collect_baseline_rows(campaign_root / "baselines")
        + collect_baseline_metric_file_rows(project_root, campaign_root / "baselines")
    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows available for report generation.")
    df = df.drop_duplicates(subset=["dataset", "task_type", "task", "method", "source_path"], keep="last")

    big_csv = reports_root / "edge_only_big_table.csv"
    df.sort_values(["dataset", "task_type", "task", "experiment_group", "method"]).to_csv(big_csv, index=False)

    summary_lines = ["# Edge-only benchmark tables", ""]
    all_edge_datasets = DATASET_ORDER[:]
    patent_datasets = ["patent_cn", "patent_in", "patent_us"]
    for task_type in ["edge_transductive", "edge_inductive"]:
        for metric in ["mrr", "ap", "auc"]:
            table = build_table(df, task_type=task_type, metric=metric, datasets=all_edge_datasets)
            stem = f"{task_type}_{metric}"
            table.to_csv(reports_root / f"{stem}.csv", index=False)
            (reports_root / f"{stem}.md").write_text(to_markdown(table), encoding="utf-8")
            summary_lines.extend([f"## {stem}", "", to_markdown(table), ""])

            patent_table = build_table(df, task_type=task_type, metric=metric, datasets=patent_datasets)
            patent_stem = f"patent_{task_type}_{metric}"
            patent_table.to_csv(reports_root / f"{patent_stem}.csv", index=False)
            (reports_root / f"{patent_stem}.md").write_text(to_markdown(patent_table), encoding="utf-8")
            summary_lines.extend([f"## {patent_stem}", "", to_markdown(patent_table), ""])

    (reports_root / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(big_csv)


if __name__ == "__main__":
    main()
