#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["test_mrr", "test_auc", "test_ap"]


def load_summary(paths):
    frames = []
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        frames.append(df)
    if not frames:
        raise RuntimeError("no valid summary csv")
    return pd.concat(frames, axis=0, ignore_index=True)


def pick_base_and_best_boost(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        base = g[g["variant"] == "base"]
        boost = g[g["variant"] != "base"]
        if base.empty or boost.empty:
            continue
        b = base.iloc[0]
        x = boost.sort_values(["test_mrr", "test_ap", "test_auc"], ascending=False).iloc[0]
        rows.append(
            {
                "dataset": ds,
                "base_variant": "base",
                "boost_variant": x["variant"],
                "base_mrr": float(b["test_mrr"]),
                "boost_mrr": float(x["test_mrr"]),
                "base_auc": float(b["test_auc"]),
                "boost_auc": float(x["test_auc"]),
                "base_ap": float(b["test_ap"]),
                "boost_ap": float(x["test_ap"]),
            }
        )
    return pd.DataFrame(rows)


def plot_base_vs_best(best_df: pd.DataFrame, out_path: Path) -> None:
    if best_df.empty:
        return
    datasets = best_df["dataset"].tolist()
    x = np.arange(len(datasets))
    w = 0.12

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 2.5 * w, best_df["base_mrr"], width=w, label="base MRR", color="#4e79a7")
    ax.bar(x - 1.5 * w, best_df["boost_mrr"], width=w, label="boost MRR", color="#a0cbe8")
    ax.bar(x - 0.5 * w, best_df["base_auc"], width=w, label="base AUC", color="#f28e2b")
    ax.bar(x + 0.5 * w, best_df["boost_auc"], width=w, label="boost AUC", color="#ffbe7d")
    ax.bar(x + 1.5 * w, best_df["base_ap"], width=w, label="base AP", color="#59a14f")
    ax.bar(x + 2.5 * w, best_df["boost_ap"], width=w, label="boost AP", color="#8cd17d")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.93, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Base vs Best Boost (MRR/AUC/AP)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_delta(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for ds, g in df.groupby("dataset"):
        base = g[g["variant"] == "base"]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in g[g["variant"] != "base"].iterrows():
            rows.append(
                {
                    "dataset": ds,
                    "variant": r["variant"],
                    "delta_mrr": float(r["test_mrr"]) - float(b["test_mrr"]),
                    "delta_auc": float(r["test_auc"]) - float(b["test_auc"]),
                    "delta_ap": float(r["test_ap"]) - float(b["test_ap"]),
                }
            )
    ddf = pd.DataFrame(rows)
    if ddf.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m, c in zip(axes, ["delta_mrr", "delta_auc", "delta_ap"], ["#4e79a7", "#f28e2b", "#59a14f"]):
        for i, (ds, g) in enumerate(ddf.groupby("dataset")):
            gg = g.sort_values("variant")
            x = np.arange(len(gg)) + i * (len(gg) + 1)
            ax.bar(x, gg[m].values, color=c, alpha=0.8, label=ds if i == 0 else None)
            for xx, yy, vv in zip(x, gg[m].values, gg["variant"].values):
                ax.text(xx, yy, vv.replace("boost_", ""), rotation=90, va="bottom", ha="center", fontsize=6)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(m)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Boost Delta vs Base")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown(df: pd.DataFrame, best_df: pd.DataFrame, out_md: Path) -> None:
    lines = [
        "# Cross-Relation Boost Figures",
        "",
        "## Summary",
        "",
        df.to_markdown(index=False),
        "",
    ]
    if not best_df.empty:
        lines.extend(["## Base vs Best Boost", "", best_df.to_markdown(index=False), ""])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser("Plot cross-relation boost result figures")
    ap.add_argument("--inputs", nargs="+", required=True, help="summary.csv paths")
    ap.add_argument("--out-dir", required=True, type=str)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.inputs)
    best_df = pick_base_and_best_boost(df)
    plot_base_vs_best(best_df, out_dir / "bar_base_vs_best_boost_metrics.png")
    plot_delta(df, out_dir / "bar_delta_vs_base_by_variant.png")
    write_markdown(df, best_df, out_dir / "cross_relation_figures_summary.md")
    best_df.to_csv(out_dir / "best_boost_vs_base.csv", index=False)
    print(out_dir)


if __name__ == "__main__":
    main()
