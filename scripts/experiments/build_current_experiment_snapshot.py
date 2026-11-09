#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DATASETS = [
    "aminer_paper",
    "amazon_movies",
    "arxiv",
    "dblp",
    "raw_cn",
    "raw_us",
    "thgl_github",
    "thgl_software",
]


def infer_dataset(task: str) -> str:
    for ds in sorted(DATASETS, key=len, reverse=True):
        if task.startswith(ds):
            return ds
    return task.split("_edge_")[0].split("_node_")[0]


def infer_task_type(task: str) -> str:
    if "edge_inductive" in task:
        return "edge_inductive"
    if "edge_transductive" in task:
        return "edge_transductive"
    if "edge_patent_ipc" in task:
        return "edge_patent_ipc"
    if "node_multilabel" in task:
        return "node_multilabel"
    if "node_multiclass" in task:
        return "node_multiclass"
    if task.endswith("_ind"):
        return "edge_inductive"
    if task.endswith("_trans"):
        return "edge_transductive"
    return "unknown"


def clean_float(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def parse_baseline_metrics_dir(metrics_dir: Path, source_group: str) -> List[Dict]:
    rows: List[Dict] = []
    if not metrics_dir.exists():
        return rows

    for csv_path in sorted(metrics_dir.glob("*.csv")):
        stem = csv_path.stem
        if "__" not in stem:
            continue
        prefix, task = stem.split("__", 1)
        dataset = infer_dataset(task)
        task_type = infer_task_type(task)
        df = pd.read_csv(csv_path)

        if {"split", "mrr", "auc", "ap"}.issubset(df.columns):
            test_df = df[df["split"] == "test"].copy()
            if test_df.empty:
                continue
            for _, r in test_df.iterrows():
                method = prefix
                if "model" in test_df.columns and pd.notna(r.get("model")):
                    model_name = str(r["model"]).strip()
                    if prefix == "tgb2_custom":
                        method = f"tgb2_custom_{model_name}"
                    elif prefix == "sthn":
                        method = "sthn"
                    else:
                        method = f"{prefix}_{model_name}" if model_name else prefix
                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "task_type": task_type,
                        "method_family": "baseline",
                        "method": method,
                        "variant": "default",
                        "source_group": source_group,
                        "source_path": str(csv_path),
                        "runs": 1,
                        "status": "ok",
                        "mrr": clean_float(r.get("mrr")),
                        "mrr_std": None,
                        "auc": clean_float(r.get("auc")),
                        "auc_std": None,
                        "ap": clean_float(r.get("ap")),
                        "ap_std": None,
                        "node_ml_micro_f1": None,
                        "node_ml_micro_f1_std": None,
                        "node_ml_macro_f1": None,
                        "node_ml_macro_f1_std": None,
                        "node_ml_map": None,
                        "node_ml_map_std": None,
                        "avg_epoch_sec": clean_float(r.get("avg_epoch_sec")),
                        "elapsed_sec": None,
                        "notes": "",
                    }
                )
            continue

        if {"model", "status", "test_auc", "test_ap"}.issubset(df.columns):
            for _, r in df.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "task_type": task_type,
                        "method_family": "baseline",
                        "method": str(r.get("model", prefix)).strip(),
                        "variant": "default",
                        "source_group": source_group,
                        "source_path": str(csv_path),
                        "runs": 1,
                        "status": str(r.get("status", "ok")),
                        "mrr": clean_float(r.get("test_mrr")),
                        "mrr_std": None,
                        "auc": clean_float(r.get("test_auc")),
                        "auc_std": None,
                        "ap": clean_float(r.get("test_ap")),
                        "ap_std": None,
                        "node_ml_micro_f1": clean_float(r.get("node_ml_micro_f1")),
                        "node_ml_micro_f1_std": None,
                        "node_ml_macro_f1": clean_float(r.get("node_ml_macro_f1")),
                        "node_ml_macro_f1_std": None,
                        "node_ml_map": None,
                        "node_ml_map_std": None,
                        "avg_epoch_sec": clean_float(r.get("avg_epoch_sec")),
                        "elapsed_sec": clean_float(r.get("total_sec")),
                        "notes": "",
                    }
                )
            continue

    return rows


def parse_dhc_aggregate(csv_path: Path, source_group: str) -> List[Dict]:
    rows: List[Dict] = []
    if not csv_path.exists():
        return rows
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        rows.append(
            {
                "dataset": str(r["dataset"]),
                "task": str(r["task"]),
                "task_type": str(r["task_type"]),
                "method_family": "dhc",
                "method": "DHC",
                "variant": str(r.get("variant", "full")),
                "source_group": source_group,
                "source_path": str(csv_path),
                "runs": int(r.get("runs", 0) or 0),
                "status": "ok",
                "mrr": clean_float(r.get("test_mrr_mean")),
                "mrr_std": clean_float(r.get("test_mrr_std")),
                "auc": clean_float(r.get("test_auc_mean")),
                "auc_std": clean_float(r.get("test_auc_std")),
                "ap": clean_float(r.get("test_ap_mean")),
                "ap_std": clean_float(r.get("test_ap_std")),
                "node_ml_micro_f1": clean_float(r.get("node_ml_micro_f1_mean")),
                "node_ml_micro_f1_std": clean_float(r.get("node_ml_micro_f1_std")),
                "node_ml_macro_f1": clean_float(r.get("node_ml_macro_f1_mean")),
                "node_ml_macro_f1_std": clean_float(r.get("node_ml_macro_f1_std")),
                "node_ml_map": clean_float(r.get("node_ml_map_mean")),
                "node_ml_map_std": clean_float(r.get("node_ml_map_std")),
                "avg_epoch_sec": clean_float(r.get("avg_epoch_time_sec_mean")),
                "elapsed_sec": None,
                "notes": "",
            }
        )
    return rows


def parse_simple_summary(csv_path: Path, dataset: str, source_group: str) -> List[Dict]:
    rows: List[Dict] = []
    if not csv_path.exists():
        return rows
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        setting = str(r.get("setting", "unknown"))
        task_type = "edge_transductive" if "trans" in setting else "edge_inductive"
        task = f"{dataset}_{setting}"
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "task_type": task_type,
                "method_family": "dhc" if str(r.get("family")) == "dhc" else "baseline",
                "method": str(r.get("model")),
                "variant": setting,
                "source_group": source_group,
                "source_path": str(csv_path),
                "runs": 1,
                "status": "ok",
                "mrr": clean_float(r.get("mrr")),
                "mrr_std": None,
                "auc": clean_float(r.get("auc")),
                "auc_std": None,
                "ap": clean_float(r.get("ap")),
                "ap_std": None,
                "node_ml_micro_f1": None,
                "node_ml_micro_f1_std": None,
                "node_ml_macro_f1": None,
                "node_ml_macro_f1_std": None,
                "node_ml_map": None,
                "node_ml_map_std": None,
                "avg_epoch_sec": clean_float(r.get("avg_epoch_sec")),
                "elapsed_sec": None,
                "notes": "",
            }
        )
    return rows


def parse_head2head_metrics(root: Path, source_group: str) -> List[Dict]:
    rows: List[Dict] = []
    if not root.exists():
        return rows
    for metrics_path in sorted(root.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        if run_name.startswith("_probe_"):
            continue
        try:
            obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pieces = run_name.split("__")
        if len(pieces) < 3:
            continue
        task = "__".join(pieces[:-2]) if pieces[-1].startswith("s") else "__".join(pieces[:-1])
        task = pieces[0]
        config = pieces[1] if len(pieces) > 1 else "default"
        seed = pieces[2] if len(pieces) > 2 else ""
        dataset = infer_dataset(task)
        task_type = infer_task_type(task)
        test = obj.get("test", {})
        node_eval = obj.get("node_eval", {})
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "task_type": task_type,
                "method_family": "dhc",
                "method": "DHC",
                "variant": config,
                "source_group": source_group,
                "source_path": str(metrics_path),
                "runs": 1,
                "status": "ok",
                "mrr": clean_float(test.get("mrr")),
                "mrr_std": None,
                "auc": clean_float(test.get("auc")),
                "auc_std": None,
                "ap": clean_float(test.get("ap")),
                "ap_std": None,
                "node_ml_micro_f1": clean_float(node_eval.get("node_ml_micro_f1")),
                "node_ml_micro_f1_std": None,
                "node_ml_macro_f1": clean_float(node_eval.get("node_ml_macro_f1")),
                "node_ml_macro_f1_std": None,
                "node_ml_map": clean_float(node_eval.get("node_ml_map")),
                "node_ml_map_std": None,
                "avg_epoch_sec": clean_float(obj.get("avg_epoch_time_sec")),
                "elapsed_sec": None,
                "notes": seed,
            }
        )
    return rows


def build_best_tables(df: pd.DataFrame) -> pd.DataFrame:
    metric_specs = [
        ("mrr", "higher"),
        ("auc", "higher"),
        ("ap", "higher"),
        ("node_ml_micro_f1", "higher"),
        ("node_ml_macro_f1", "higher"),
        ("node_ml_map", "higher"),
    ]
    rows = []
    valid = df.copy()
    for (dataset, task, task_type), g in valid.groupby(["dataset", "task", "task_type"]):
        for metric, _ in metric_specs:
            subset = g[pd.to_numeric(g[metric], errors="coerce").notna()].copy()
            if subset.empty:
                continue
            best_idx = subset[metric].astype(float).idxmax()
            best = subset.loc[best_idx]
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "task_type": task_type,
                    "metric": metric,
                    "best_value": clean_float(best[metric]),
                    "best_method": best["method"],
                    "best_variant": best["variant"],
                    "source_group": best["source_group"],
                    "source_path": best["source_path"],
                }
            )
    return pd.DataFrame(rows)


def first_existing_path(project_root: Path, rel_paths: List[str]) -> Optional[Path]:
    for rel in rel_paths:
        candidate = project_root / rel
        if candidate.exists():
            return candidate
    return None


def to_markdown_safe(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main() -> None:
    ap = argparse.ArgumentParser("Build a current experiment snapshot across available result roots.")
    ap.add_argument("--project-root", type=str, default="/home/xuxj/2026/dyn_hetero_community")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir.strip() else project_root / "results" / "experiment_snapshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    baseline_all_gpu = first_existing_path(project_root, ["results/baseline_formal_all_gpu/metrics", "results-auto/baseline_formal_all_gpu/metrics"])
    baseline_aminer = first_existing_path(project_root, ["results/baseline_formal_aminer/metrics", "results-auto/baseline_formal_aminer/metrics"])
    full_benchmark = first_existing_path(project_root, ["results/full_benchmark_repeats/aggregate_mean_std.csv", "results-auto/full_benchmark_repeats/aggregate_mean_std.csv"])
    full_benchmark_aminer = first_existing_path(project_root, ["results/full_benchmark_aminer_formal/aggregate_mean_std.csv", "results-auto/full_benchmark_aminer_formal/aggregate_mean_std.csv"])
    tgb2_github = first_existing_path(project_root, ["results/tgb2_thgl_github_full_rel/summary.csv", "results-auto/tgb2_thgl_github_full_rel/summary.csv"])
    tgb2_software = first_existing_path(project_root, ["results/tgb2_thgl_software_full_rel/summary.csv", "results-auto/tgb2_thgl_software_full_rel/summary.csv"])
    head2head_current = first_existing_path(project_root, ["results/head2head_rawcn_current", "results-auto/head2head_rawcn_current"])
    head2head_manual = first_existing_path(project_root, ["results/head2head_rawcn_current_manual", "results-auto/head2head_rawcn_current_manual"])

    rows += parse_baseline_metrics_dir(baseline_all_gpu or Path("/nonexistent"), "baseline_formal_all_gpu")
    rows += parse_baseline_metrics_dir(baseline_aminer or Path("/nonexistent"), "baseline_formal_aminer")

    rows += parse_dhc_aggregate(full_benchmark or Path("/nonexistent"), "full_benchmark_repeats")
    rows += parse_dhc_aggregate(full_benchmark_aminer or Path("/nonexistent"), "full_benchmark_aminer_formal")

    rows += parse_simple_summary(tgb2_github or Path("/nonexistent"), "thgl_github", "tgb2_thgl_github_full_rel")
    rows += parse_simple_summary(tgb2_software or Path("/nonexistent"), "thgl_software", "tgb2_thgl_software_full_rel")

    rows += parse_head2head_metrics(head2head_current or Path("/nonexistent"), "head2head_rawcn_current")
    rows += parse_head2head_metrics(head2head_manual or Path("/nonexistent"), "head2head_rawcn_current_manual")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No results found to aggregate.")

    df = df.sort_values(["dataset", "task_type", "task", "method_family", "method", "variant", "source_group"]).reset_index(drop=True)
    long_csv = out_dir / "all_results_long.csv"
    df.to_csv(long_csv, index=False)

    # Headline snapshot: keep full DHC/mainline baseline plus current raw_cn tuning.
    headline = df[
        (
            ((df["method_family"] == "baseline") & df["variant"].eq("default"))
            | ((df["method_family"] == "dhc") & df["variant"].eq("full"))
            | (df["source_group"].isin(["head2head_rawcn_current", "head2head_rawcn_current_manual"]))
            | (df["dataset"].isin(["thgl_github", "thgl_software"]))
        )
    ].copy()
    headline_csv = out_dir / "headline_results.csv"
    headline.to_csv(headline_csv, index=False)

    best_df = build_best_tables(df)
    best_csv = out_dir / "best_by_task_metric.csv"
    best_df.to_csv(best_csv, index=False)

    coverage = (
        df.groupby(["source_group", "dataset", "method_family"])["task"]
        .nunique()
        .reset_index(name="num_tasks")
        .sort_values(["source_group", "dataset", "method_family"])
    )
    coverage_csv = out_dir / "coverage_by_source.csv"
    coverage.to_csv(coverage_csv, index=False)

    # Compact markdown summary.
    md_lines = [
        "# Current Experiment Snapshot",
        "",
        f"- rows: {len(df)}",
        f"- output dir: `{out_dir}`",
        "",
        "## Sources",
        "",
        to_markdown_safe(coverage),
        "",
        "## Best Available Per Task/Metric",
        "",
        to_markdown_safe(best_df.head(60)),
        "",
        "## Raw CN Current-Tuning Rows",
        "",
        to_markdown_safe(
            df[
                (df["dataset"] == "raw_cn")
                & (df["source_group"].isin(["head2head_rawcn_current", "head2head_rawcn_current_manual"]))
            ][
                ["task", "method", "variant", "mrr", "auc", "ap", "avg_epoch_sec", "source_group", "source_path"]
            ].sort_values(["task", "mrr", "auc"], ascending=[True, False, False])
        ),
        "",
        "## Headline Rows",
        "",
        to_markdown_safe(
            headline[
                [
                    "dataset",
                    "task",
                    "task_type",
                    "method_family",
                    "method",
                    "variant",
                    "mrr",
                    "auc",
                    "ap",
                    "node_ml_micro_f1",
                    "node_ml_map",
                    "avg_epoch_sec",
                    "runs",
                    "source_group",
                ]
            ].sort_values(["dataset", "task", "method_family", "method", "variant"])
        ),
        "",
    ]
    summary_md = out_dir / "summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    tracking_script = project_root / "scripts" / "experiments" / "update_tracking_docs.py"
    if tracking_script.exists():
        subprocess.run(
            [
                "python",
                str(tracking_script),
                "--project-root",
                str(project_root),
            ],
            check=False,
        )

    print(long_csv)
    print(headline_csv)
    print(best_csv)
    print(coverage_csv)
    print(summary_md)


if __name__ == "__main__":
    main()
