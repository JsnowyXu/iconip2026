#!/usr/bin/env python3
"""Standalone DHC memory artifact visualizer.

This script reads saved DHC run artifacts without modifying them and writes
derived CSV/PNG files for repeated 2D visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path("/home/xuxj/2026/dyn_relation_memory")
RESULTS_ROOT = PROJECT_ROOT / "results"

DATASET_PATTERNS = {
    "patent-us": ["patent_us"],
    "patent_us": ["patent_us"],
    "aminer_paper": ["aminer_paper"],
    "dblp": ["dblp"],
    "thgl_software": ["thgl_software"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "Visualize node/relation/community memory artifacts from DHC runs."
    )
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--results-root", type=Path, default=None)
    ap.add_argument("--dataset", default="", choices=["", *sorted(DATASET_PATTERNS)])
    ap.add_argument("--run-dir", type=Path, default=None, help="Run directory containing memory_trace.pt.")
    ap.add_argument("--trace", type=Path, default=None, help="Direct path to memory_trace.pt.")
    ap.add_argument("--final-state", type=Path, default=None, help="Optional path to memory_final_state.pt.")
    ap.add_argument("--data", type=Path, default=None, help="Optional task/unified .pt for node_type metadata.")
    ap.add_argument("--out-dir", type=Path, required=False, default=None)
    ap.add_argument("--list-runs", action="store_true", help="List candidate runs and exit.")
    ap.add_argument("--max-runs", type=int, default=200)
    ap.add_argument("--max-points", type=int, default=2000)
    ap.add_argument("--max-labels", type=int, default=80)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--memory-kinds", default="node,relation,community")
    return ap.parse_args()


def resolve_results_root(args: argparse.Namespace) -> Path:
    return args.results_root or args.project_root / "results"


def candidate_runs(results_root: Path, dataset: str) -> list[Path]:
    patterns = DATASET_PATTERNS.get(dataset, [dataset] if dataset else [""])
    out: list[Path] = []
    for trace in sorted(results_root.glob("**/memory_trace.pt")):
        s = str(trace.parent)
        if all(p not in s for p in patterns):
            continue
        if "community_memory_case_study" in s:
            continue
        out.append(trace.parent)
    return out


def choose_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def as_array(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pca_2d(x: np.ndarray, device: torch.device) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if x.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if x.shape[1] == 1:
        return np.column_stack([x[:, 0] - x[:, 0].mean(), np.zeros(x.shape[0])]).astype(np.float32)

    t = torch.from_numpy(x).to(device)
    t = t - t.mean(dim=0, keepdim=True)
    try:
        _, _, vh = torch.linalg.svd(t, full_matrices=False)
        z = t @ vh[:2].T
    except RuntimeError:
        t = t.cpu()
        _, _, vh = torch.linalg.svd(t, full_matrices=False)
        z = t @ vh[:2].T
    z = z.detach().cpu().numpy().astype(np.float32)
    if z.shape[1] == 1:
        z = np.column_stack([z[:, 0], np.zeros(z.shape[0], dtype=np.float32)])
    return z[:, :2]


def sample_rows(x: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return x[idx], idx


def metric_label(run_dir: Path) -> dict[str, str]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception:
        return {}
    names = list((metrics.get("test_relation_breakdown") or {}).keys())
    return {str(i): name for i, name in enumerate(names)}


def save_projection(
    out_dir: Path,
    kind: str,
    matrix: np.ndarray,
    epoch: int,
    device: torch.device,
    max_points: int,
    max_labels: int,
    dpi: int,
    labels: dict[str, str] | None = None,
) -> None:
    if matrix is None or matrix.ndim != 2 or matrix.size == 0:
        return
    sampled, original_idx = sample_rows(matrix.astype(np.float32), max_points)
    z = pca_2d(sampled, device)
    norms = np.linalg.norm(sampled, axis=1)
    df = pd.DataFrame(
        {
            "kind": kind,
            "epoch": int(epoch),
            "slot": original_idx,
            "x": z[:, 0],
            "y": z[:, 1],
            "norm": norms,
        }
    )
    if labels:
        df["label"] = [labels.get(str(i), f"slot_{i}") for i in original_idx]
    df.to_csv(out_dir / f"{kind}_final_projection.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(df["x"], df["y"], c=df["norm"], s=24, cmap="viridis", alpha=0.82)
    title_kind = kind.replace("_", " ").title()
    ax.set_title(f"{title_kind} memory PCA, epoch {epoch}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.18)
    fig.colorbar(sc, ax=ax, label="L2 norm")
    if len(df) <= max_labels:
        for _, r in df.iterrows():
            text = r.get("label", f"{int(r['slot'])}")
            ax.text(r["x"], r["y"], str(text), fontsize=7, alpha=0.75)
    fig.tight_layout()
    fig.savefig(out_dir / f"{kind}_final_projection.png", dpi=dpi)
    plt.close(fig)


def save_first_last_projection(
    out_dir: Path,
    kind: str,
    first: np.ndarray,
    last: np.ndarray,
    first_epoch: int,
    last_epoch: int,
    device: torch.device,
    max_points: int,
    dpi: int,
) -> None:
    if first is None or last is None or first.ndim != 2 or last.ndim != 2:
        return
    n = min(first.shape[0], last.shape[0], max_points)
    if n == 0:
        return
    idx = np.linspace(0, min(first.shape[0], last.shape[0]) - 1, n).round().astype(int)
    stacked = np.vstack([first[idx], last[idx]]).astype(np.float32)
    z = pca_2d(stacked, device)
    z0, z1 = z[:n], z[n:]
    pd.DataFrame(
        {
            "kind": kind,
            "slot": idx,
            "first_epoch": int(first_epoch),
            "last_epoch": int(last_epoch),
            "x_first": z0[:, 0],
            "y_first": z0[:, 1],
            "x_last": z1[:, 0],
            "y_last": z1[:, 1],
            "delta_norm": np.linalg.norm(last[idx] - first[idx], axis=1),
        }
    ).to_csv(out_dir / f"{kind}_first_last_projection.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(z0[:, 0], z0[:, 1], s=20, alpha=0.7, label=f"epoch {first_epoch}")
    ax.scatter(z1[:, 0], z1[:, 1], s=20, alpha=0.7, label=f"epoch {last_epoch}")
    for i in range(n):
        ax.plot([z0[i, 0], z1[i, 0]], [z0[i, 1], z1[i, 1]], color="#777777", alpha=0.18, linewidth=0.8)
    ax.set_title(f"{kind.title()} memory first vs last PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / f"{kind}_first_last_projection.png", dpi=dpi)
    plt.close(fig)


def memory_summary(snaps: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for pos, snap in enumerate(snaps):
        row: dict[str, float] = {"epoch": float(snap.get("epoch", pos + 1))}
        for key, label in [
            ("node_memory", "node"),
            ("rel_memory", "relation"),
            ("community_memory", "community"),
        ]:
            arr = as_array(snap.get(key))
            if arr is None or arr.ndim != 2 or arr.size == 0:
                continue
            norm = np.linalg.norm(arr, axis=1)
            row[f"{label}_norm_mean"] = float(norm.mean())
            row[f"{label}_norm_std"] = float(norm.std())
            row[f"{label}_slots"] = float(arr.shape[0])
            row[f"{label}_dim"] = float(arr.shape[1])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch")


def save_norm_plot(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    df.to_csv(out_dir / "memory_norm_evolution.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for c, label in [
        ("node_norm_mean", "node"),
        ("relation_norm_mean", "relation"),
        ("community_norm_mean", "community"),
    ]:
        if c in df:
            err = f"{label}_norm_std"
            ax.plot(df["epoch"], df[c], marker="o", label=label)
            if err in df:
                lo = df[c] - df[err]
                hi = df[c] + df[err]
                ax.fill_between(df["epoch"], lo, hi, alpha=0.12)
    ax.set_title("Memory norm evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean L2 norm")
    ax.legend()
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_norm_evolution.png", dpi=dpi)
    plt.close(fig)


def save_relation_slot_evolution(snaps: list[dict[str, Any]], out_dir: Path, dpi: int, labels: dict[str, str]) -> None:
    rows = []
    for pos, snap in enumerate(snaps):
        arr = as_array(snap.get("rel_memory"))
        if arr is None or arr.ndim != 2 or arr.size == 0:
            continue
        epoch = int(snap.get("epoch", pos + 1))
        norms = np.linalg.norm(arr, axis=1)
        for slot, norm in enumerate(norms):
            rows.append(
                {
                    "epoch": epoch,
                    "relation_slot": slot,
                    "relation_label": labels.get(str(slot), f"slot_{slot}"),
                    "relation_memory_norm": float(norm),
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "relation_slot_evolution.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for slot, sub in df.groupby("relation_slot"):
        label = labels.get(str(slot), f"slot_{slot}")
        ax.plot(sub["epoch"], sub["relation_memory_norm"], marker="o", linewidth=1.4, label=label)
    ax.set_title("Relation memory slot evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.18)
    if df["relation_slot"].nunique() <= 16:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "relation_slot_evolution.png", dpi=dpi)
    plt.close(fig)


def save_training_curves(run_dir: Path, out_dir: Path, dpi: int) -> None:
    hist_path = run_dir / "train_history.csv"
    if not hist_path.exists():
        return
    df = pd.read_csv(hist_path)
    df.to_csv(out_dir / "train_history.csv", index=False)
    metrics = [c for c in ["loss", "val_auc", "val_ap", "val_acc", "val_mrr"] if c in df.columns]
    if not metrics:
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.2, max(3.2, 1.8 * len(metrics))), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, c in zip(axes, metrics):
        ax.plot(df["epoch"], df[c], marker="o")
        ax.set_ylabel(c)
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Training curves", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=dpi)
    plt.close(fig)


def write_run_summary(run_dir: Path, out_dir: Path, device: torch.device, snaps: list[dict[str, Any]]) -> None:
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "device_used_for_pca": str(device),
        "num_snapshots": len(snaps),
    }
    for name in ["metrics.json", "model_stats.json"]:
        p = run_dir / name
        if p.exists():
            try:
                summary[name] = json.loads(p.read_text())
            except Exception:
                summary[name] = str(p)
    (out_dir / "run_visualization_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    results_root = resolve_results_root(args)

    if args.list_runs:
        runs = candidate_runs(results_root, args.dataset)[: args.max_runs]
        for run in runs:
            print(run)
        print(f"found {len(candidate_runs(results_root, args.dataset))} candidate runs")
        return

    if args.trace:
        trace_path = args.trace
        run_dir = trace_path.parent
    elif args.run_dir:
        run_dir = args.run_dir
        trace_path = run_dir / "memory_trace.pt"
    else:
        runs = candidate_runs(results_root, args.dataset)
        if not runs:
            raise SystemExit("No candidate memory_trace.pt found. Pass --run-dir or --trace.")
        run_dir = runs[0]
        trace_path = run_dir / "memory_trace.pt"

    if not trace_path.exists():
        raise SystemExit(f"Missing trace file: {trace_path}")

    out_dir = args.out_dir or run_dir / "memory_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    trace = load_torch(trace_path)
    snaps = trace.get("snapshots", [])
    if not snaps:
        raise SystemExit(f"No snapshots found in {trace_path}")

    kinds = {k.strip() for k in args.memory_kinds.split(",") if k.strip()}
    labels = metric_label(run_dir)
    first = snaps[0]
    last = snaps[-1]
    first_epoch = int(first.get("epoch", 1))
    last_epoch = int(last.get("epoch", len(snaps)))

    df = memory_summary(snaps)
    save_norm_plot(df, out_dir, args.dpi)
    save_relation_slot_evolution(snaps, out_dir, args.dpi, labels)
    save_training_curves(run_dir, out_dir, args.dpi)

    key_by_kind = {
        "node": "node_memory",
        "relation": "rel_memory",
        "community": "community_memory",
    }
    for kind, key in key_by_kind.items():
        if kind not in kinds:
            continue
        first_arr = as_array(first.get(key))
        last_arr = as_array(last.get(key))
        use_labels = labels if kind == "relation" else None
        if last_arr is not None:
            save_projection(
                out_dir,
                kind,
                last_arr,
                last_epoch,
                device,
                args.max_points,
                args.max_labels,
                args.dpi,
                use_labels,
            )
        if first_arr is not None and last_arr is not None:
            save_first_last_projection(
                out_dir,
                kind,
                first_arr,
                last_arr,
                first_epoch,
                last_epoch,
                device,
                args.max_points,
                args.dpi,
            )

    final_path = args.final_state or run_dir / "memory_final_state.pt"
    if final_path.exists():
        final_state = load_torch(final_path)
        for kind, key in key_by_kind.items():
            if kind not in kinds:
                continue
            arr = as_array(final_state.get(key))
            if arr is not None:
                save_projection(
                    out_dir,
                    f"{kind}_replayed",
                    arr,
                    last_epoch,
                    device,
                    args.max_points,
                    args.max_labels,
                    args.dpi,
                    labels if kind == "relation" else None,
                )

    write_run_summary(run_dir, out_dir, device, snaps)
    print(f"trace: {trace_path}")
    print(f"output: {out_dir}")
    print(f"device used for PCA: {device}")


if __name__ == "__main__":
    main()
