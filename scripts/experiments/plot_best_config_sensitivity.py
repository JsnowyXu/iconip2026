#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _try_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series)
    except Exception:
        return pd.Series(range(len(series)), index=series.index)


def build_report_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task_name, parameter), sub in df.groupby(["task_name", "parameter"]):
        sub = sub.copy()
        sub["test_mrr"] = pd.to_numeric(sub["test_mrr"], errors="coerce")
        sub = sub.dropna(subset=["test_mrr"])
        if sub.empty:
            continue
        sub = sub.sort_values("test_mrr", ascending=False)
        best = sub.iloc[0]
        worst = sub.iloc[-1]
        anchor = pd.to_numeric(sub["source_test_mrr"], errors="coerce").dropna()
        anchor_mrr = float(anchor.iloc[0]) if not anchor.empty else None
        rows.append(
            {
                "task_name": task_name,
                "dataset": str(best.get("dataset", "")),
                "task_type": str(best.get("task_type", "")),
                "parameter": parameter,
                "anchor_value": best.get("anchor_value", ""),
                "anchor_test_mrr": anchor_mrr,
                "best_value": best.get("value", ""),
                "best_test_mrr": float(best["test_mrr"]),
                "worst_value": worst.get("value", ""),
                "worst_test_mrr": float(worst["test_mrr"]),
                "swing_mrr": float(best["test_mrr"] - worst["test_mrr"]),
                "best_delta_vs_anchor": (float(best["test_mrr"]) - anchor_mrr) if anchor_mrr is not None else "",
            }
        )
    return pd.DataFrame(rows)


def plot_task_panels(df: pd.DataFrame, out_dir: Path) -> None:
    for task_name, g in df.groupby("task_name"):
        params = list(dict.fromkeys(g["parameter"].tolist()))
        ncols = 2
        nrows = (len(params) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.8 * nrows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for idx, parameter in enumerate(params):
            ax = axes[idx]
            sub = g[g["parameter"] == parameter].copy()
            anchor_mrr = pd.to_numeric(sub["source_test_mrr"], errors="coerce").dropna()
            anchor_mrr = float(anchor_mrr.iloc[0]) if not anchor_mrr.empty else None
            anchor_value = sub["anchor_value"].iloc[0] if "anchor_value" in sub.columns and not sub.empty else ""
            if parameter == "neighbor_aggregation":
                sub["plot_x"] = range(len(sub))
                ax.plot(list(sub["plot_x"]), sub["test_mrr"], marker="o", linewidth=2)
                ax.set_xticks(list(sub["plot_x"]))
                ax.set_xticklabels(sub["value"].astype(str).tolist())
                values = sub["value"].astype(str).tolist()
                if str(anchor_value) in values:
                    ax.axvline(values.index(str(anchor_value)), color="#999999", linestyle="--", linewidth=1.2)
            else:
                sub["plot_x"] = _try_numeric(sub["value"])
                sub = sub.sort_values("plot_x")
                ax.plot(sub["plot_x"], sub["test_mrr"], marker="o", linewidth=2)
                try:
                    ax.axvline(float(anchor_value), color="#999999", linestyle="--", linewidth=1.2)
                except Exception:
                    pass
            if anchor_mrr is not None:
                ax.axhline(anchor_mrr, color="#d62728", linestyle=":", linewidth=1.2)
            ax.set_title(parameter)
            ax.set_xlabel(parameter)
            ax.set_ylabel("test MRR")
            ax.grid(alpha=0.25)
        for idx in range(len(params), len(axes)):
            axes[idx].axis("off")
        fig.suptitle(task_name)
        fig.tight_layout()
        fig.savefig(out_dir / f"{task_name}_sensitivity.png", dpi=180)
        plt.close(fig)


def plot_parameter_panels(df: pd.DataFrame, out_dir: Path) -> None:
    for parameter, g in df.groupby("parameter"):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        anchor_drawn = False
        for task_name, sub in g.groupby("task_name"):
            sub = sub.copy()
            anchor_mrr = pd.to_numeric(sub["source_test_mrr"], errors="coerce").dropna()
            anchor_mrr = float(anchor_mrr.iloc[0]) if not anchor_mrr.empty else None
            if parameter == "neighbor_aggregation":
                sub["plot_x"] = range(len(sub))
                ax.plot(list(sub["plot_x"]), sub["test_mrr"], marker="o", linewidth=2, label=task_name)
                ax.set_xticks(list(sub["plot_x"]))
                ax.set_xticklabels(sub["value"].astype(str).tolist())
            else:
                sub["plot_x"] = _try_numeric(sub["value"])
                sub = sub.sort_values("plot_x")
                ax.plot(sub["plot_x"], sub["test_mrr"], marker="o", linewidth=2, label=task_name)
            if anchor_mrr is not None and not anchor_drawn:
                ax.axhline(anchor_mrr, color="#d62728", linestyle=":", linewidth=1.0, label="anchor MRR")
                anchor_drawn = True
        ax.set_title(parameter)
        ax.set_xlabel(parameter)
        ax.set_ylabel("test MRR")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"parameter_{parameter}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser("Plot one-factor sensitivity sweeps from summary.csv")
    ap.add_argument("--summary-csv", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory/results/best_config_sensitivity/summary.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory/results/best_config_sensitivity/plots"))
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise SystemExit(f"empty summary: {args.summary_csv}")
    df = df[df["return_code"] == 0].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_task_panels(df, args.out_dir)
    plot_parameter_panels(df, args.out_dir)
    report_df = build_report_rows(df)
    if not report_df.empty:
        report_df.to_csv(args.out_dir / "sensitivity_report.csv", index=False)
        lines = [
            "# Sensitivity Plot Summary",
            "",
            "Dashed vertical lines mark the anchor parameter values.",
            "Dotted horizontal lines mark the anchor test MRR.",
            "",
        ]
        for _, row in report_df.sort_values(["dataset", "task_type", "task_name", "parameter"]).iterrows():
            lines.append(
                f"- `{row['task_name']}` / `{row['parameter']}`: best `{row['best_value']}` -> `{row['best_test_mrr']:.6f}`, "
                f"worst `{row['worst_value']}` -> `{row['worst_test_mrr']:.6f}`, swing `{row['swing_mrr']:.6f}`"
            )
        (args.out_dir / "README.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()
