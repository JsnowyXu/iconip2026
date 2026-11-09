#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_ORDER = [
    ("patent_cn", "edge_inductive"),
    ("patent_cn", "edge_transductive"),
    ("thgl_software", "edge_inductive"),
    ("thgl_software", "edge_transductive"),
]
METHOD_ORDER = ["DHC", "TAMI"]
METHOD_COLORS = {
    "DHC": "#7FB3D5",
    "TAMI": "#F6C177",
}
METHOD_EDGES = {
    "DHC": "#2F6F9F",
    "TAMI": "#B46D1F",
}
METHOD_HATCHES = {
    "DHC": "///",
    "TAMI": "\\\\",
}


def task_label(dataset: str, task_type: str) -> str:
    dataset_name = {
        "patent_cn": "Patent-CN",
        "thgl_software": "THGL-Software",
    }.get(dataset, dataset.replace("_", "-"))
    split_name = {
        "edge_inductive": "Ind.",
        "edge_transductive": "Trans.",
    }.get(task_type, task_type)
    return f"{dataset_name}\n{split_name}"


def read_dhc_metric_row(row: pd.Series) -> dict:
    path = Path(str(row["metrics_path"]))
    obj = json.loads(path.read_text(encoding="utf-8"))
    test = obj.get("test", {})
    epoch_times = [float(x) for x in obj.get("epoch_time_sec", [])]
    best_epoch = int(obj.get("best_epoch") or 0)
    stopped_epoch = int(obj.get("stopped_epoch") or len(epoch_times))
    elapsed_sec = float(sum(epoch_times)) if epoch_times else float("nan")
    time_to_best_sec = float(sum(epoch_times[:best_epoch])) if epoch_times and best_epoch > 0 else elapsed_sec
    return {
        "dataset": row["dataset"],
        "task_name": row["task_name"],
        "task_type": row["task_type"],
        "method": "DHC",
        "source_group": row.get("source_group", ""),
        "run_name": row.get("run_name", ""),
        "mrr": float(test.get("mrr", row.get("best_test_mrr", np.nan))),
        "auc": float(test.get("auc", np.nan)),
        "ap": float(test.get("ap", np.nan)),
        "avg_epoch_sec": float(obj.get("avg_epoch_time_sec", np.nan)),
        "best_epoch": best_epoch,
        "epochs_run": stopped_epoch,
        "time_to_best_sec": time_to_best_sec,
        "elapsed_sec": elapsed_sec,
        "time_source": "sum(epoch_time_sec)",
        "metrics_path": str(path),
    }


def build_comparison_csv(project_root: Path, out_csv: Path) -> pd.DataFrame:
    rows = []
    best_catalog = project_root / "results" / "best_config_sensitivity" / "best_run_catalog.csv"
    dhc = pd.read_csv(best_catalog)
    dhc = dhc[dhc["dataset"].isin(["patent_cn", "thgl_software"])].copy()
    for _, row in dhc.iterrows():
        if (row["dataset"], row["task_type"]) in TASK_ORDER:
            rows.append(read_dhc_metric_row(row))

    tami_summary = project_root / "results" / "tami_baseline_formal_20260508" / "summary.csv"
    tami = pd.read_csv(tami_summary)
    tami = tami[
        (tami["dataset"].isin(["patent_cn", "thgl_software"]))
        & (tami["split"] == "test")
        & (tami["task_type"].isin(["edge_inductive", "edge_transductive"]))
    ].copy()
    for _, row in tami.iterrows():
        avg_epoch = float(row["avg_epoch_sec"])
        best_epoch = int(row["best_epoch"])
        epochs_run = int(row["epochs_run"])
        rows.append(
            {
                "dataset": row["dataset"],
                "task_name": row["task"],
                "task_type": row["task_type"],
                "method": row["baseline"],
                "source_group": "tami_baseline_formal_20260508",
                "run_name": Path(str(row["source_path"])).stem,
                "mrr": float(row["mrr"]),
                "auc": float(row["auc"]),
                "ap": float(row["ap"]),
                "avg_epoch_sec": avg_epoch,
                "best_epoch": best_epoch,
                "epochs_run": epochs_run,
                "time_to_best_sec": avg_epoch * best_epoch,
                "elapsed_sec": avg_epoch * epochs_run,
                "time_source": "avg_epoch_sec * epoch_count",
                "metrics_path": row["source_path"],
            }
        )

    df = pd.DataFrame(rows)
    order = {(dataset, task_type, method): i for i, (dataset, task_type) in enumerate(TASK_ORDER) for method in METHOD_ORDER}
    df["_order"] = df.apply(lambda r: order.get((r["dataset"], r["task_type"], r["method"]), 999), axis=1)
    df = df.sort_values(["_order", "method"]).drop(columns=["_order"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def plot_comparison(df: pd.DataFrame, out_png: Path) -> None:
    df = df.copy()
    df["time_to_best_min"] = pd.to_numeric(df["time_to_best_sec"], errors="coerce") / 60.0
    df["mrr"] = pd.to_numeric(df["mrr"], errors="coerce")
    labels = [task_label(dataset, task_type) for dataset, task_type in TASK_ORDER]
    x = np.arange(len(TASK_ORDER))
    width = 0.34

    fig, (ax_mrr, ax_time) = plt.subplots(2, 1, figsize=(9.2, 6.7), sharex=True)
    for idx, method in enumerate(METHOD_ORDER):
        vals_mrr = []
        vals_time = []
        for dataset, task_type in TASK_ORDER:
            sub = df[(df["dataset"] == dataset) & (df["task_type"] == task_type) & (df["method"] == method)]
            vals_mrr.append(float(sub["mrr"].iloc[0]) if not sub.empty else np.nan)
            vals_time.append(float(sub["time_to_best_min"].iloc[0]) if not sub.empty else np.nan)
        offset = (idx - 0.5) * width
        ax_mrr.bar(
            x + offset,
            vals_mrr,
            width,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor=METHOD_EDGES[method],
            hatch=METHOD_HATCHES[method],
            linewidth=0.9,
        )
        ax_time.bar(
            x + offset,
            vals_time,
            width,
            label=method,
            color=METHOD_COLORS[method],
            edgecolor=METHOD_EDGES[method],
            hatch=METHOD_HATCHES[method],
            linewidth=0.9,
        )

    ax_mrr.set_ylabel("Test MRR")
    ax_mrr.set_ylim(0.55, 1.02)
    ax_mrr.grid(axis="y", alpha=0.24, linewidth=0.7)
    ax_mrr.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax_mrr.set_title("Performance and time-to-best comparison on Patent-CN and THGL-Software", fontsize=11, pad=22)

    ax_time.set_ylabel("Time to best epoch (min)")
    ax_time.set_yscale("log")
    ax_time.grid(axis="y", alpha=0.24, linewidth=0.7, which="both")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels)

    for ax in [ax_mrr, ax_time]:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser("Build and plot runtime-performance comparison for DHC vs baseline")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("/home/xuxj/2026/dyn_relation_memory/ICONIP/data/runtime_performance_patent_cn_thgl.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/home/xuxj/2026/dyn_relation_memory/ICONIP/figures/runtime_performance_patent_cn_thgl.png"),
    )
    ap.add_argument("--from-csv", action="store_true", help="Plot an existing CSV instead of rebuilding it.")
    args = ap.parse_args()

    if args.from_csv:
        df = pd.read_csv(args.csv)
    else:
        df = build_comparison_csv(args.project_root.resolve(), args.csv)
    plot_comparison(df, args.out)
    print(args.csv)
    print(args.out)


if __name__ == "__main__":
    main()
