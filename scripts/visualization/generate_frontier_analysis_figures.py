#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "frontier_visual_analysis"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"


COLORS = {
    "ink": "#15202B",
    "muted": "#667085",
    "grid": "#E7EAEE",
    "blue": "#2563EB",
    "cyan": "#06B6D4",
    "green": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "purple": "#7C3AED",
}


def hide_spines(ax, names=("top", "right")) -> None:
    for name in names:
        ax.spines[name].set_visible(False)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D5DD",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def read_metric(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    test = obj.get("test", {})
    return {
        "test_mrr": float(test["mrr"]) if "mrr" in test else np.nan,
        "test_auc": float(test["auc"]) if "auc" in test else np.nan,
        "test_ap": float(test["ap"]) if "ap" in test else np.nan,
    }


def compact_task_name(name: str) -> str:
    name = name.replace("_edge_org_codeIII", "")
    name = name.replace("_edge", "")
    name = name.replace("patent_", "patent-")
    name = name.replace("thgl_software", "software")
    name = name.replace("aminer_paper_citation", "aminer-cite")
    name = name.replace("_", " ")
    return name


def numeric_value(v: object) -> float | None:
    try:
        return float(str(v).replace("p", "."))
    except Exception:
        return None


def display_path(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def build_sensitivity_table() -> pd.DataFrame:
    plan_path = ROOT / "results" / "best_config_sensitivity" / "job_plan.csv"
    if not plan_path.exists():
        return pd.DataFrame()
    plan = pd.read_csv(plan_path)
    rows = []
    for _, row in plan.iterrows():
        metrics = read_metric(row["metrics_path"])
        if not metrics:
            continue
        rows.append({**row.to_dict(), **metrics})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["source_test_mrr"] = pd.to_numeric(df["source_test_mrr"], errors="coerce")
    df["test_mrr"] = pd.to_numeric(df["test_mrr"], errors="coerce")
    df["test_auc"] = pd.to_numeric(df["test_auc"], errors="coerce")
    df["test_ap"] = pd.to_numeric(df["test_ap"], errors="coerce")
    df["delta_mrr"] = df["test_mrr"] - df["source_test_mrr"]
    df["delta_mrr_bp"] = df["delta_mrr"] * 10000.0
    df["value_numeric"] = df["value"].map(numeric_value)
    df.to_csv(DATA_DIR / "sensitivity_metrics_from_job_plan.csv", index=False)
    return df


def plot_sensitivity_gain(df: pd.DataFrame) -> Path | None:
    if df.empty:
        return None
    params = [p for p in df["parameter"].dropna().unique().tolist()]
    ncols = 3
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, max(6.0, 3.6 * nrows)))
    axes = np.ravel(axes)
    palette = [COLORS["blue"], COLORS["cyan"], COLORS["green"], COLORS["amber"], COLORS["purple"], COLORS["red"]]
    for ax, parameter in zip(axes, params):
        sub = df[df["parameter"] == parameter].copy()
        for idx, (task, g) in enumerate(sub.groupby("task_name")):
            g = g.copy()
            color = palette[idx % len(palette)]
            if g["value_numeric"].notna().all():
                g = g.sort_values("value_numeric")
                x = g["value_numeric"]
                ax.plot(x, g["delta_mrr_bp"], marker="o", linewidth=1.8, markersize=4.5, color=color, alpha=0.9, label=compact_task_name(task))
            else:
                g = g.sort_values("value")
                x = np.arange(len(g))
                ax.plot(x, g["delta_mrr_bp"], marker="o", linewidth=1.8, markersize=4.5, color=color, alpha=0.9, label=compact_task_name(task))
                ax.set_xticks(x)
                ax.set_xticklabels(g["value"].astype(str), rotation=0)
        ax.axhline(0, color=COLORS["ink"], linewidth=1.0)
        ax.set_title(parameter)
        ax.set_ylabel("MRR gain vs anchor (bp)")
        ax.set_xlabel("value")
        hide_spines(ax)
    for ax in axes[len(params) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:8], labels[:8], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Sensitivity Gain Landscape", y=1.055, fontsize=15, fontweight="bold")
    out = FIG_DIR / "sensitivity_gain_landscape.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_sensitivity_heatmap(df: pd.DataFrame) -> Path | None:
    if df.empty:
        return None
    best = df.groupby(["task_name", "parameter"], as_index=False)["delta_mrr_bp"].agg(["max", "min"]).reset_index()
    best = best.rename(columns={"max": "best_gain_bp", "min": "worst_gain_bp"})
    best["swing_bp"] = best["best_gain_bp"] - best["worst_gain_bp"]
    pivot = best.pivot(index="parameter", columns="task_name", values="swing_bp")
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    arr = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
    im = ax.imshow(arr, cmap="magma", vmin=0, vmax=max(vmax, 1e-9), aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([compact_task_name(c) for c in pivot.columns], rotation=28, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Parameter Sensitivity Swing")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white" if val > vmax * 0.45 else COLORS["ink"], fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("best-worst MRR swing (bp)")
    out = FIG_DIR / "sensitivity_swing_heatmap.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    best.to_csv(DATA_DIR / "sensitivity_swing_by_task_parameter.csv", index=False)
    return out


def plot_training_curves() -> list[Path]:
    candidates = []
    roots = [
        ROOT / "results" / "edge_only_formal" / "main",
        ROOT / "results" / "patent_dhc_formal",
        ROOT / "results" / "aminer_dhc_formal",
        ROOT / "results" / "paper_dhc_formal",
    ]
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.glob("*/train_history.csv")):
            task_name = p.parent.name
            if "_edge" not in task_name:
                continue
            candidates.append({"task_name": task_name, "history_path": p, "source_group": root.name})

    catalog_path = ROOT / "results" / "best_config_sensitivity" / "best_run_catalog.csv"
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        for _, r in catalog.iterrows():
            p = Path(str(r["run_dir"])) / "train_history.csv"
            candidates.append({"task_name": str(r["task_name"]), "history_path": p, "source_group": "best_config_sensitivity"})

    if not candidates:
        return []

    rows = []
    seen = set()
    for r in candidates:
        hist_path = Path(r["history_path"])
        if not hist_path.exists():
            continue
        key = (r["task_name"], str(hist_path))
        if key in seen:
            continue
        seen.add(key)
        hist = pd.read_csv(hist_path)
        if "epoch" not in hist.columns or "loss" not in hist.columns:
            continue
        for _, h in hist.iterrows():
            rows.append(
                {
                    "task_name": r["task_name"],
                    "source_group": r["source_group"],
                    "history_path": display_path(hist_path),
                    "epoch": h.get("epoch"),
                    "loss": h.get("loss"),
                    "val_mrr": h.get("val_mrr"),
                    "val_ap": h.get("val_ap"),
                    "epoch_sec": h.get("epoch_sec"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["val_mrr"] = pd.to_numeric(df["val_mrr"], errors="coerce")
    df = df.dropna(subset=["epoch", "loss"])
    df.to_csv(DATA_DIR / "dhc_training_history_long.csv", index=False)
    tasks = df["task_name"].drop_duplicates().tolist()
    ncols = 3
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, max(6.5, 2.85 * nrows)))
    axes = np.ravel(axes)
    for ax, task in zip(axes, tasks):
        sub = df[df["task_name"] == task].sort_values("epoch")
        ax2 = ax.twinx()
        ax.plot(sub["epoch"], sub["loss"], color=COLORS["red"], linewidth=1.8, label="loss")
        if sub["val_mrr"].notna().any():
            ax2.plot(sub["epoch"], sub["val_mrr"], color=COLORS["blue"], linewidth=2.0, marker="o", markersize=3.0, label="val MRR")
        ax.set_title(compact_task_name(task))
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss", color=COLORS["red"])
        ax2.set_ylabel("val MRR", color=COLORS["blue"])
        hide_spines(ax, ("top",))
        hide_spines(ax2, ("top",))
    for ax in axes[len(tasks) :]:
        ax.axis("off")
    fig.suptitle("DHC Epoch Dynamics Across Datasets / Tasks", fontsize=15, fontweight="bold", y=1.01)
    out = FIG_DIR / "dhc_epoch_loss_mrr_multi_task.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return [out]


def plot_memory_evolution() -> list[Path]:
    base = ROOT / "results" / "community_memory_case_study"
    paths = sorted(base.glob("*/memory_evolution/memory_evolution.csv"))
    if not paths:
        return []
    rows = []
    for p in paths:
        task = p.parts[-3]
        df = pd.read_csv(p)
        df["task_name"] = task
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(DATA_DIR / "memory_evolution_long.csv", index=False)

    tasks = all_df["task_name"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 8.2), sharex=False)
    for task, g in all_df.groupby("task_name"):
        label = compact_task_name(task)
        if "rel_mem_norm_mean" in g:
            axes[0].plot(g["epoch"], g["rel_mem_norm_mean"], linewidth=1.8, alpha=0.92, label=label)
        if "community_mem_norm_mean" in g:
            axes[1].plot(g["epoch"], g["community_mem_norm_mean"], linewidth=1.8, alpha=0.92, label=label)
    axes[0].set_title("Relation Memory Norm Evolution")
    axes[1].set_title("Community Memory Norm Evolution")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean L2 norm")
        hide_spines(ax)
    axes[0].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    out1 = FIG_DIR / "relation_memory_evolution_multi_task.png"
    fig.savefig(out1, dpi=220)
    plt.close(fig)

    churn_rows = []
    for p in sorted(base.glob("*/monitor_assignment_churn.csv")):
        df = pd.read_csv(p)
        df["task_name"] = p.parent.name
        churn_rows.append(df)
    outs = [out1]
    if churn_rows:
        churn = pd.concat(churn_rows, ignore_index=True)
        churn.to_csv(DATA_DIR / "monitor_assignment_churn_long.csv", index=False)
        fig, ax = plt.subplots(figsize=(12.8, 5.8))
        for task, g in churn.groupby("task_name"):
            ax.plot(g["epoch"], g["membership_churn"], linewidth=1.8, alpha=0.9, label=compact_task_name(task))
        ax.set_title("Monitor-Node Community Churn")
        ax.set_xlabel("epoch")
        ax.set_ylabel("membership churn")
        hide_spines(ax)
        ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.2))
        out2 = FIG_DIR / "community_churn_multi_task.png"
        fig.savefig(out2, dpi=220)
        plt.close(fig)
        outs.append(out2)
    return outs


def plot_raw_relation_memory_fingerprints() -> list[Path]:
    try:
        import torch
    except Exception:
        return []
    evidence_path = ROOT / "results" / "relation_temporal_context_case_study" / "aggregate_case_study_evidence.csv"
    if evidence_path.exists():
        evidence = pd.read_csv(evidence_path)
        trace_paths = evidence["memory_trace"].dropna().drop_duplicates().tolist()
    else:
        trace_paths = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "results").glob("**/memory_trace.pt"))[:8]]
    rows = []
    for rel_path in trace_paths:
        p = ROOT / rel_path
        if not p.exists():
            p = Path(rel_path)
        if not p.exists():
            continue
        try:
            obj = torch.load(p, map_location="cpu")
        except Exception:
            continue
        for snap in obj.get("snapshots", []):
            rel_mem = snap.get("rel_memory")
            if rel_mem is None:
                continue
            arr = np.asarray(rel_mem, dtype=np.float32)
            if arr.ndim != 2 or arr.size == 0:
                continue
            norms = np.linalg.norm(arr, axis=1)
            for slot, norm in enumerate(norms):
                rows.append(
                    {
                        "trace": display_path(p),
                        "run": p.parent.name,
                        "epoch": int(snap.get("epoch", 0)),
                        "relation_slot": int(slot),
                        "relation_memory_norm": float(norm),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    df.to_csv(DATA_DIR / "raw_relation_memory_fingerprints.csv", index=False)
    runs = df["run"].drop_duplicates().tolist()[:8]
    ncols = 2
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, max(4.8, 3.1 * nrows)))
    axes = np.ravel(axes)
    for ax, run in zip(axes, runs):
        sub = df[df["run"] == run]
        pivot = sub.pivot_table(index="relation_slot", columns="epoch", values="relation_memory_norm", aggfunc="mean")
        arr = pivot.to_numpy(dtype=float)
        im = ax.imshow(arr, cmap="viridis", aspect="auto")
        ax.set_title(compact_task_name(run.replace("__", " ")), fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("relation slot")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.astype(int), fontsize=7)
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.astype(int), fontsize=8)
    for ax in axes[len(runs) :]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes[: len(runs)].tolist(), fraction=0.024, pad=0.015)
    cbar.set_label("relation-memory L2 norm")
    fig.suptitle("Raw Relation Memory Fingerprints", fontsize=15, fontweight="bold", y=1.01)
    out = FIG_DIR / "raw_relation_memory_fingerprints.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return [out]


def plot_case_studies() -> list[Path]:
    outs: list[Path] = []
    rel_path = ROOT / "results" / "relation_temporal_context_case_study" / "relation_memory_ablation_case.csv"
    if rel_path.exists():
        df = pd.read_csv(rel_path)
        df.to_csv(DATA_DIR / "relation_memory_ablation_case.csv", index=False)
        plot = df.sort_values("delta_ap_vs_full")
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        y = np.arange(len(plot))
        colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in plot["delta_ap_vs_full"]]
        ax.barh(y, plot["delta_ap_vs_full"], color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(plot["variant"])
        ax.axvline(0, color=COLORS["ink"], linewidth=1.0)
        ax.set_xlabel("AP delta vs full")
        ax.set_title("Relation Memory Ablation Case")
        hide_spines(ax)
        for yi, val in zip(y, plot["delta_ap_vs_full"]):
            ax.text(val, yi, f" {val:+.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
        out = FIG_DIR / "relation_memory_ablation_delta_ap.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outs.append(out)

    neigh_path = ROOT / "results" / "relation_temporal_context_case_study" / "temporal_context_neighbor_size_case.csv"
    if neigh_path.exists():
        df = pd.read_csv(neigh_path)
        df.to_csv(DATA_DIR / "temporal_context_neighbor_size_case.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=False)
        for idx, metric in enumerate(["test_mrr", "test_ap"]):
            ax = axes[idx]
            for task, g in df.groupby("task_name"):
                g = g.sort_values("recent_neighbor_size")
                ax.plot(g["recent_neighbor_size"], g[metric], marker="o", linewidth=2.0, label=compact_task_name(task))
                best_size = g["best_size"].dropna().iloc[0] if g["best_size"].notna().any() else None
                if best_size is not None:
                    ax.axvline(best_size, linestyle=":", linewidth=1.0, color=COLORS["muted"], alpha=0.6)
            ax.set_title(metric.replace("test_", "test ").upper())
            ax.set_xlabel("recent neighbor size")
            ax.set_ylabel(metric)
            hide_spines(ax)
        axes[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.05), ncol=2)
        fig.suptitle("Temporal Context / Neighbor Window", fontsize=15, fontweight="bold", y=1.08)
        out = FIG_DIR / "temporal_context_neighbor_window.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_edgebank_diagnostics() -> list[Path]:
    paths = sorted((ROOT / "results" / "baseline_audit_20260520" / "edgebank").glob("edgebank__*.csv"))
    if not paths:
        paths = sorted((ROOT / "results" / "edge_only_formal" / "baselines" / "metrics").glob("edgebank__*.csv"))
    rows = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        needed = {"pos_history_hit_rate", "all_zero_tie_rate", "mrr", "split"}
        if not needed.issubset(df.columns):
            continue
        task = p.stem.replace("edgebank__", "")
        df = df.copy()
        df["task_name"] = task
        df["source_csv"] = display_path(p)
        rows.append(df)
    if not rows:
        return []
    df = pd.concat(rows, ignore_index=True)
    df["pos_history_hit_rate"] = pd.to_numeric(df["pos_history_hit_rate"], errors="coerce")
    df["all_zero_tie_rate"] = pd.to_numeric(df["all_zero_tie_rate"], errors="coerce")
    df["mrr"] = pd.to_numeric(df["mrr"], errors="coerce")
    df.to_csv(DATA_DIR / "edgebank_diagnostics_long.csv", index=False)

    test = df[df["split"].astype(str).str.lower().eq("test")].copy()
    if test.empty:
        test = df.copy()
    test = test.sort_values(["task_name", "split"])
    y = np.arange(len(test))
    height = 0.38
    fig, ax = plt.subplots(figsize=(12.8, max(5.0, 0.38 * len(test) + 1.6)))
    ax.barh(y - height / 2, test["pos_history_hit_rate"], height=height, color=COLORS["green"], label="pos history hit rate")
    ax.barh(y + height / 2, test["all_zero_tie_rate"], height=height, color=COLORS["red"], alpha=0.82, label="all-zero tie rate")
    ax.set_yticks(y)
    ax.set_yticklabels([compact_task_name(t) for t in test["task_name"]], fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("rate")
    ax.set_title("EdgeBank Diagnostics: History Hits vs Degenerate Ties")
    hide_spines(ax)
    ax.legend(loc="lower right")
    out1 = FIG_DIR / "edgebank_history_hit_zero_tie_rates.png"
    fig.savefig(out1, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    sc = ax.scatter(
        test["pos_history_hit_rate"],
        test["mrr"],
        s=70 + 220 * test["all_zero_tie_rate"].fillna(0),
        c=test["all_zero_tie_rate"],
        cmap="rocket_r" if "rocket_r" in plt.colormaps() else "magma_r",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    for _, row in test.iterrows():
        ax.annotate(compact_task_name(row["task_name"]).replace(" transductive", " T").replace(" inductive", " I"), (row["pos_history_hit_rate"], row["mrr"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("pos history hit rate")
    ax.set_ylabel("EdgeBank test MRR")
    ax.set_title("EdgeBank MRR vs Historical Coverage")
    hide_spines(ax)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("all-zero tie rate")
    out2 = FIG_DIR / "edgebank_mrr_vs_history_hit.png"
    fig.savefig(out2, dpi=220)
    plt.close(fig)
    return [out1, out2]


def write_inventory(generated: list[Path]) -> Path:
    items = []
    checks = [
        ("memory_trace.pt", sorted((ROOT / "results").glob("**/memory_trace.pt"))),
        ("train_history.csv", sorted((ROOT / "results").glob("**/train_history.csv"))),
        ("best_config_sensitivity/job_plan.csv", [ROOT / "results" / "best_config_sensitivity" / "job_plan.csv"]),
        ("best_config_sensitivity/best_run_catalog.csv", [ROOT / "results" / "best_config_sensitivity" / "best_run_catalog.csv"]),
        ("relation_temporal_context_case_study/*.csv", sorted((ROOT / "results" / "relation_temporal_context_case_study").glob("*.csv"))),
        ("community_memory_case_study memory_evolution.csv", sorted((ROOT / "results" / "community_memory_case_study").glob("*/memory_evolution/memory_evolution.csv"))),
        ("baseline_audit_20260520/edgebank/*.csv", sorted((ROOT / "results" / "baseline_audit_20260520" / "edgebank").glob("*.csv"))),
        ("baseline_audit_20260520/dyglib/*.csv", sorted((ROOT / "results" / "baseline_audit_20260520" / "dyglib").glob("*.csv"))),
        ("referenceanalysis", sorted((ROOT / "referenceanalysis").glob("*"))),
    ]
    for name, paths in checks:
        real = [p for p in paths if p.exists()]
        items.append((name, len(real), real[:8]))

    lines = [
        "# Frontier Visual Analysis Inventory",
        "",
        "Generated only from existing result files in this workspace.",
        "",
        "## Available Data",
    ]
    for name, count, paths in items:
        lines.append(f"- {name}: {count} file(s)")
        for p in paths:
            lines.append(f"  - `{p.relative_to(ROOT)}`")
        if count > len(paths):
            lines.append(f"  - ... {count - len(paths)} more")
    lines += ["", "## Generated Figures"]
    for p in generated:
        lines.append(f"- `{p.relative_to(ROOT)}`")
    lines += ["", "## Data Limits"]
    if (DATA_DIR / "raw_relation_memory_fingerprints.csv").exists():
        lines.append(
            "- Raw `.pt` memory traces were read with a torch-enabled interpreter for `raw_relation_memory_fingerprints.png`; CSV memory summaries were also used for multi-task evolution curves."
        )
    else:
        lines.append(
            "- Raw `.pt` memory traces exist, but the active Python environment could not import `torch`; this run used the already materialized CSV summaries under `results/community_memory_case_study` instead of inventing memory values."
        )
    lines += [
        "- No per-edge prediction/error-case table was found by filename/content search, so no false-positive/false-negative case figure was generated.",
        "- Sensitivity figures use completed runs with readable `metrics.json` paths from `results/best_config_sensitivity/job_plan.csv`.",
        "- DyGFormer/TGAT audit files currently expose summary metrics only, not epoch-level loss/MRR histories; they were not drawn as epoch curves.",
    ]
    out = OUT_DIR / "README.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    setup_style()
    ensure_dirs()
    generated: list[Path] = []
    sensitivity = build_sensitivity_table()
    for out in [plot_sensitivity_gain(sensitivity), plot_sensitivity_heatmap(sensitivity)]:
        if out is not None:
            generated.append(out)
    generated.extend(plot_training_curves())
    generated.extend(plot_memory_evolution())
    generated.extend(plot_raw_relation_memory_fingerprints())
    generated.extend(plot_case_studies())
    generated.extend(plot_edgebank_diagnostics())
    readme = write_inventory(generated)
    print(f"generated {len(generated)} figures")
    print(readme)
    for p in generated:
        print(p)


if __name__ == "__main__":
    main()
