#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def config_meta(config_name: str) -> Dict:
    table = {
        "base": {"lambda_ml": 1.0, "threshold_search": 0, "pos_weight": 0},
        "thr": {"lambda_ml": 1.0, "threshold_search": 1, "pos_weight": 0},
        "thr_pos": {"lambda_ml": 1.0, "threshold_search": 1, "pos_weight": 1},
        "thr_pos_l03": {"lambda_ml": 0.3, "threshold_search": 1, "pos_weight": 1},
    }
    return table.get(config_name, {"lambda_ml": np.nan, "threshold_search": np.nan, "pos_weight": np.nan})


def grouped_bar_f1(df: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(df["dataset"].unique().tolist())
    configs = sorted(df["config"].unique().tolist())
    x = np.arange(len(datasets))
    width = min(0.18, 0.8 / max(1, len(configs)))

    plt.figure(figsize=(10, 4.6))
    for i, cfg in enumerate(configs):
        vals = []
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["config"] == cfg)]
            vals.append(float(sub["node_ml_micro_f1"].iloc[0]) if len(sub) > 0 else 0.0)
        plt.bar(x + (i - (len(configs) - 1) / 2) * width, vals, width=width, label=cfg)
    plt.xticks(x, datasets)
    plt.ylabel("Node micro-F1")
    plt.title("Grid Search Results: Node micro-F1 by Dataset/Config")
    plt.grid(axis="y", alpha=0.2)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def base_vs_recommended(df: pd.DataFrame, rec_df: pd.DataFrame, out_path: Path) -> None:
    datasets = rec_df["dataset"].tolist()
    base_auc = rec_df["base_auc"].astype(float).values
    base_ap = rec_df["base_ap"].astype(float).values
    best_auc = rec_df["best_auc"].astype(float).values
    best_ap = rec_df["best_ap"].astype(float).values

    x = np.arange(len(datasets))
    w = 0.18

    plt.figure(figsize=(10, 4.8))
    plt.bar(x - 1.5 * w, base_auc, width=w, label="base AUC", color="#4e79a7")
    plt.bar(x - 0.5 * w, best_auc, width=w, label="recommended AUC", color="#2a9d8f")
    plt.bar(x + 0.5 * w, base_ap, width=w, label="base AP", color="#f4a261")
    plt.bar(x + 1.5 * w, best_ap, width=w, label="recommended AP", color="#e76f51")
    plt.xticks(x, datasets)
    plt.ylabel("Score")
    plt.title("Base vs Recommended (AUC/AP)")
    plt.grid(axis="y", alpha=0.2)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def pareto_plot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.2, 5.0))
    datasets = sorted(df["dataset"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        plt.scatter(
            sub["test_ap"].astype(float).values,
            sub["node_ml_micro_f1"].astype(float).values,
            label=ds,
            s=42,
            alpha=0.9,
            color=cmap(i % 10),
        )
        for _, r in sub.iterrows():
            plt.annotate(str(r["config"]), (float(r["test_ap"]), float(r["node_ml_micro_f1"])), fontsize=7, alpha=0.7)
    plt.xlabel("Test AP")
    plt.ylabel("Node micro-F1")
    plt.title("Pareto View: AP vs micro-F1")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def sensitivity_plot(df: pd.DataFrame, out_path: Path) -> None:
    ext = df.copy()
    meta = ext["config"].apply(config_meta).apply(pd.Series)
    ext = pd.concat([ext, meta], axis=1)

    grp = (
        ext.groupby(["lambda_ml", "threshold_search", "pos_weight"], dropna=False)[
            ["node_ml_micro_f1", "test_ap", "test_auc"]
        ]
        .mean()
        .reset_index()
    )
    grp = grp.sort_values(["threshold_search", "pos_weight", "lambda_ml"])

    x = np.arange(len(grp))
    labels = [
        f"lm={r['lambda_ml']},thr={int(r['threshold_search'])},pw={int(r['pos_weight'])}"
        for _, r in grp.iterrows()
    ]

    plt.figure(figsize=(11, 4.8))
    plt.plot(x, grp["node_ml_micro_f1"].values, marker="o", label="micro-F1", color="#e76f51")
    plt.plot(x, grp["test_ap"].values, marker="s", label="AP", color="#2a9d8f")
    plt.plot(x, grp["test_auc"].values, marker="^", label="AUC", color="#4e79a7")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Parameter Sensitivity (Averaged Across Datasets)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def training_curves(rec_df: pd.DataFrame, out_root: Path, out_path: Path) -> None:
    plt.figure(figsize=(10, 5.2))
    plotted = 0
    for _, r in rec_df.iterrows():
        run_name = str(r["recommended_run"])
        hist_path = out_root / run_name / "train_history.csv"
        if not hist_path.exists():
            continue
        try:
            h = pd.read_csv(hist_path)
        except Exception:
            continue
        if "epoch" not in h.columns:
            continue
        y_col = "val_auc" if "val_auc" in h.columns else ("loss" if "loss" in h.columns else None)
        if y_col is None:
            continue
        plt.plot(h["epoch"], h[y_col], marker="o", label=f"{r['dataset']} ({run_name})")
        plotted += 1

    if plotted == 0:
        return
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC / Loss")
    plt.title("Training Curves of Recommended Runs")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True, type=str)
    ap.add_argument("--recommended-csv", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--runs-root", required=True, type=str)
    args = ap.parse_args()

    summary = pd.read_csv(args.summary_csv)
    summary = summary[summary["status"] == "ok"].copy()
    rec_df = pd.read_csv(args.recommended_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = Path(args.runs_root)

    grouped_bar_f1(summary, out_dir / "bar_f1_by_dataset.png")
    base_vs_recommended(summary, rec_df, out_dir / "bar_base_vs_recommended_auc_ap.png")
    pareto_plot(summary, out_dir / "scatter_pareto_ap_vs_f1.png")
    sensitivity_plot(summary, out_dir / "line_parameter_sensitivity.png")
    training_curves(rec_df, runs_root, out_dir / "training_curves_recommended.png")

    report_lines = [
        "# Grid Search Figures",
        "",
        "- bar_f1_by_dataset.png",
        "- bar_base_vs_recommended_auc_ap.png",
        "- scatter_pareto_ap_vs_f1.png",
        "- line_parameter_sensitivity.png",
        "- training_curves_recommended.png (if history exists)",
    ]
    (out_dir / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
