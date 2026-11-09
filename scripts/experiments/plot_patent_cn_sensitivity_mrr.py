#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PARAMS = [
    "typed_negative_ratio",
    "popular_negative_ratio",
    "recent_neighbor_size",
    "neighbor_aggregation",
    "community_assign_topk",
    "num_communities",
]
TASK_LABELS = {
    "edge_transductive": "Transductive",
    "edge_inductive": "Inductive",
}
COLORS = {
    "edge_transductive": "#4C78A8",
    "edge_inductive": "#59A14F",
}
MARKERS = {
    "edge_transductive": "o",
    "edge_inductive": "s",
}


def sort_values_for_plot(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    return pd.Series(range(len(series)), index=series.index)


def parameter_title(parameter: str) -> str:
    return parameter.replace("_", " ")


def draw_plot(df: pd.DataFrame, out_path: Path) -> None:
    draw_plot_with_params(df, out_path, PARAMS, "Patent-CN parameter sensitivity on MRR")


def draw_boxplot_with_params(df: pd.DataFrame, out_path: Path, params: list[str], title: str) -> None:
    df = df.copy()
    df["test_mrr"] = pd.to_numeric(df["test_mrr"], errors="coerce")
    df = df.dropna(subset=["test_mrr"])
    params = [p for p in params if p in set(df["parameter"].astype(str))]
    if not params:
        raise SystemExit("None of the requested parameters are present in the input CSV.")

    data = []
    labels = []
    for parameter in params:
        values = df[df["parameter"] == parameter]["test_mrr"].tolist()
        if values:
            data.append(values)
            labels.append(parameter.replace("_", "\n"))
    if not data:
        raise SystemExit("No data available for the requested parameters.")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        boxprops=dict(facecolor="#CFE4F4", color="#4C78A8", linewidth=1.0, hatch="///"),
        medianprops=dict(color="#B23A48", linewidth=1.4),
        meanprops=dict(marker="D", markerfacecolor="#2F6F9F", markeredgecolor="white", markersize=5),
        whiskerprops=dict(color="#4C78A8", linewidth=0.9),
        capprops=dict(color="#4C78A8", linewidth=0.9),
        flierprops=dict(marker="o", markerfacecolor="#F6C177", markeredgecolor="#B46D1F", markersize=4, alpha=0.75),
    )

    for patch, color, edge, hatch in zip(
        bp["boxes"],
        ["#CFE4F4", "#F9D6A6", "#CDEBCB", "#E3D1EC", "#D7E8BA", "#F4C7C3"],
        ["#4C78A8", "#C27628", "#4C9A5D", "#8E6CA6", "#6B8E23", "#B23A48"],
        ["///", "\\\\", "xx", "..", "--", "++"],
    ):
        patch.set_facecolor(color)
        patch.set_edgecolor(edge)
        patch.set_hatch(hatch)

    ax.set_title(title, fontsize=11, pad=14)
    ax.set_ylabel("Test MRR")
    ax.grid(axis="y", alpha=0.24, linewidth=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _add_plot_metric(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    if metric == "test_mrr":
        df["_plot_metric"] = pd.to_numeric(df["test_mrr"], errors="coerce")
        return df, "Test MRR"
    if metric == "delta_mrr":
        if "delta_mrr_vs_anchor" in df.columns:
            raw_delta = pd.to_numeric(df["delta_mrr_vs_anchor"], errors="coerce")
            test = pd.to_numeric(df.get("test_mrr"), errors="coerce")
            anchor = pd.to_numeric(df.get("anchor_test_mrr"), errors="coerce")
            computed = test - anchor
            # Some hand-collected CSVs use anchor-test despite the column name.
            if raw_delta.notna().any() and computed.notna().any():
                same = (raw_delta - computed).abs().mean()
                reversed_same = (raw_delta + computed).abs().mean()
                df["_plot_metric"] = -raw_delta if reversed_same < same else raw_delta
            else:
                df["_plot_metric"] = raw_delta
        else:
            df["_plot_metric"] = pd.to_numeric(df["test_mrr"], errors="coerce") - pd.to_numeric(
                df["anchor_test_mrr"], errors="coerce"
            )
        return df, "MRR gain over anchor"
    raise ValueError(f"unsupported metric: {metric}")


def draw_value_boxplot_panels(df: pd.DataFrame, out_path: Path, params: list[str], title: str, metric: str) -> None:
    df = df.copy()
    df, ylabel = _add_plot_metric(df, metric)
    df = df.dropna(subset=["_plot_metric"])
    params = [p for p in params if p in set(df["parameter"].astype(str))]
    if not params:
        raise SystemExit("None of the requested parameters are present in the input CSV.")

    ncols = 2 if len(params) <= 4 else 3
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.6 * nrows), sharey=False)
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = [axes]

    fill_colors = ["#CFE4F4", "#F9D6A6", "#CDEBCB", "#E3D1EC", "#D7E8BA", "#F4C7C3"]
    edge_colors = ["#4C78A8", "#C27628", "#4C9A5D", "#8E6CA6", "#6B8E23", "#B23A48"]
    hatches = ["///", "\\\\", "xx", "..", "--", "++"]

    for ax, parameter in zip(axes, params):
        sub = df[df["parameter"] == parameter].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub["_value_sort"] = sort_values_for_plot(sub["value"])
        value_rows = sub[["value", "_value_sort"]].drop_duplicates().sort_values("_value_sort")
        values = value_rows["value"].astype(str).tolist()
        data = [sub[sub["value"].astype(str) == value]["_plot_metric"].tolist() for value in values]

        bp = ax.boxplot(
            data,
            labels=values,
            patch_artist=True,
            widths=0.56,
            showmeans=True,
            boxprops=dict(facecolor="#CFE4F4", color="#4C78A8", linewidth=1.0),
            medianprops=dict(color="#B23A48", linewidth=1.35),
            meanprops=dict(marker="D", markerfacecolor="#2F6F9F", markeredgecolor="white", markersize=4.5),
            whiskerprops=dict(color="#4C78A8", linewidth=0.9),
            capprops=dict(color="#4C78A8", linewidth=0.9),
            flierprops=dict(marker="o", markerfacecolor="#F6C177", markeredgecolor="#B46D1F", markersize=3.8, alpha=0.75),
        )
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(fill_colors[idx % len(fill_colors)])
            patch.set_edgecolor(edge_colors[idx % len(edge_colors)])
            patch.set_hatch(hatches[idx % len(hatches)])

        ax.set_xlabel(parameter)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        if metric == "delta_mrr":
            ax.axhline(0.0, color="#555555", linestyle=":", linewidth=0.9, alpha=0.8)
        ax.grid(axis="y", alpha=0.24, linewidth=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes[len(params) :]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def draw_plot_with_params(df: pd.DataFrame, out_path: Path, params: list[str], title: str) -> None:
    df = df.copy()
    df["test_mrr"] = pd.to_numeric(df["test_mrr"], errors="coerce")
    df["anchor_test_mrr"] = pd.to_numeric(df.get("anchor_test_mrr"), errors="coerce")
    df = df.dropna(subset=["test_mrr"])
    if df.empty:
        raise SystemExit("No valid test_mrr values found in the input CSV.")

    params = [p for p in params if p in set(df["parameter"].astype(str))]
    if not params:
        raise SystemExit("None of the requested parameters are present in the input CSV.")
    ncols = 2 if len(params) <= 4 else 3
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.3 * nrows), sharey=False)
    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.ravel()
        except AttributeError:
            axes = [axes]
    else:
        axes = list(axes)
    for ax, parameter in zip(axes, params):
        sub_param = df[df["parameter"] == parameter].copy()
        if sub_param.empty:
            ax.axis("off")
            continue

        for task_type, sub in sub_param.groupby("task_type", sort=False):
            sub = sub.copy()
            sub["plot_x"] = sort_values_for_plot(sub["value"])
            sub = sub.sort_values("plot_x")
            label = TASK_LABELS.get(str(task_type), str(task_type))
            ax.plot(
                sub["plot_x"],
                sub["test_mrr"],
                marker=MARKERS.get(str(task_type), "o"),
                linewidth=1.9,
                markersize=5.2,
                color=COLORS.get(str(task_type)),
                label=label,
            )

            anchor_mrr = sub["anchor_test_mrr"].dropna()
            if not anchor_mrr.empty:
                ax.axhline(
                    float(anchor_mrr.iloc[0]),
                    color=COLORS.get(str(task_type), "#777777"),
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.55,
                )

        values = sub_param.drop_duplicates("value").copy()
        values["plot_x"] = sort_values_for_plot(values["value"])
        values = values.sort_values("plot_x")
        ax.set_xticks(values["plot_x"].tolist())
        ax.set_xticklabels(values["value"].astype(str).tolist(), rotation=30, ha="right")
        ax.set_title(parameter_title(parameter), fontsize=10)
        ax.set_ylabel("Test MRR")
        ax.grid(axis="y", alpha=0.24, linewidth=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes[len(params) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, fontsize=12, y=1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_best_settings(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    clean = df.copy()
    clean["test_mrr"] = pd.to_numeric(clean["test_mrr"], errors="coerce")
    clean = clean.dropna(subset=["test_mrr"])
    for (task_type, parameter), sub in clean.groupby(["task_type", "parameter"]):
        best = sub.sort_values("test_mrr", ascending=False).iloc[0]
        rows.append(
            {
                "task_type": task_type,
                "parameter": parameter,
                "value": best["value"],
                "test_mrr": best["test_mrr"],
                "run_name": best.get("run_name", ""),
            }
        )
    best_df = pd.DataFrame(rows).sort_values(["task_type", "parameter"])
    best_df.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser("Plot Patent-CN MRR parameter sensitivity from CSV")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("/home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_cn_sensitivity_mrr.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/home/xuxj/2026/dyn_relation_memory/ICONIP/figures/patent_cn_sensitivity_mrr.png"),
    )
    ap.add_argument(
        "--best-out",
        type=Path,
        default=Path("/home/xuxj/2026/dyn_relation_memory/ICONIP/data/patent_cn_sensitivity_best_settings.csv"),
    )
    ap.add_argument(
        "--params",
        type=str,
        default="",
        help="Comma-separated parameters to plot. Defaults to all parameters present in the input CSV.",
    )
    ap.add_argument(
        "--exclude-community",
        action="store_true",
        help="Drop parameters whose names contain 'community'.",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="",
        help="Figure title. Defaults to '<dataset> parameter sensitivity on MRR'.",
    )
    ap.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw the top-level figure title.",
    )
    ap.add_argument(
        "--plot-kind",
        choices=["line", "box", "value-box"],
        default="line",
        help="Use line panels, parameter-level boxplots, or value-level boxplots inside parameter panels.",
    )
    ap.add_argument(
        "--box-metric",
        choices=["test_mrr", "delta_mrr"],
        default="test_mrr",
        help="Metric for value-box plots. delta_mrr uses test MRR minus anchor MRR.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.exclude_community:
        df = df[~df["parameter"].astype(str).str.contains("community", case=False, na=False)].copy()
    if args.params.strip():
        params = [x.strip() for x in args.params.split(",") if x.strip()]
    else:
        params = [p for p in PARAMS if p in set(df["parameter"].astype(str))]
        params += [p for p in df["parameter"].astype(str).drop_duplicates().tolist() if p not in params]
    dataset = str(df["dataset"].dropna().iloc[0]) if "dataset" in df.columns and not df.empty else "Dataset"
    title = "" if args.no_title else (args.title.strip() or f"{dataset.replace('_', '-').title()} parameter sensitivity on MRR")
    if args.plot_kind == "box":
        draw_boxplot_with_params(df, args.out, params, title)
    elif args.plot_kind == "value-box":
        draw_value_boxplot_panels(df, args.out, params, title, args.box_metric)
    else:
        draw_plot_with_params(df, args.out, params, title)
    write_best_settings(df, args.best_out)
    print(args.out)
    print(args.best_out)


if __name__ == "__main__":
    main()
