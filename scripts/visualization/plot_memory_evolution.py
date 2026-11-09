#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _save_first_last_heatmap(first_arr, last_arr, title: str, out_path: Path, max_rows: int = 48) -> None:
    if first_arr is None or last_arr is None:
        return
    first_arr = np.asarray(first_arr, dtype=np.float32)
    last_arr = np.asarray(last_arr, dtype=np.float32)
    if first_arr.ndim != 2 or last_arr.ndim != 2 or first_arr.size == 0 or last_arr.size == 0:
        return
    k = min(int(max_rows), first_arr.shape[0], last_arr.shape[0])
    first_plot = first_arr[:k]
    last_plot = last_arr[:k]
    vmax = float(max(np.abs(first_plot).max(), np.abs(last_plot).max(), 1e-6))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, max(3.0, 0.18 * k + 2.0)))
    for ax, arr, subtitle in zip(axes, [first_plot, last_plot], ["first", "last"]):
        im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{title} ({subtitle})")
        ax.set_xlabel("dimension")
        ax.set_ylabel("slot")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("Plot DHC memory evolution curves")
    ap.add_argument("--trace", required=True, type=str, help="Path to memory_trace.pt")
    ap.add_argument("--data", default="", type=str, help="Optional unified .pt for node_type lookup")
    ap.add_argument("--out-dir", required=True, type=str)
    args = ap.parse_args()

    trace_obj = torch.load(args.trace, map_location="cpu")
    snaps = trace_obj.get("snapshots", [])
    if len(snaps) == 0:
        raise RuntimeError(f"no snapshots found in {args.trace}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_type = None
    if args.data:
        data_obj = torch.load(args.data, map_location="cpu")
        node_type = data_obj.get("node_type", None)
        if node_type is not None:
            node_type = node_type.numpy().astype(np.int64)

    rows = []
    first_node_mem = None
    last_node_mem = None
    first_rel_mem = None
    last_rel_mem = None
    first_comm_mem = None
    last_comm_mem = None
    for s in snaps:
        ep = int(s.get("epoch", len(rows) + 1))
        node_mem = s.get("node_memory", None)
        rel_mem = s.get("rel_memory", None)
        comm_mem = s.get("community_memory", None)
        monitor_ids = s.get("monitor_node_ids", None)

        row = {"epoch": ep}
        if node_mem is not None:
            if first_node_mem is None:
                first_node_mem = np.asarray(node_mem, dtype=np.float32)
            last_node_mem = np.asarray(node_mem, dtype=np.float32)
            node_norm = np.linalg.norm(node_mem, axis=1)
            row["node_mem_norm_mean"] = float(node_norm.mean())
            row["node_mem_norm_std"] = float(node_norm.std())
            if node_type is not None and monitor_ids is not None:
                mids = np.asarray(monitor_ids, dtype=np.int64)
                for t in np.unique(node_type[mids]):
                    mask = node_type[mids] == t
                    if mask.any():
                        row[f"node_type_{int(t)}_norm_mean"] = float(node_norm[mask].mean())

        if rel_mem is not None:
            if first_rel_mem is None:
                first_rel_mem = np.asarray(rel_mem, dtype=np.float32)
            last_rel_mem = np.asarray(rel_mem, dtype=np.float32)
            rel_norm = np.linalg.norm(rel_mem, axis=1)
            row["rel_mem_norm_mean"] = float(rel_norm.mean())
            row["rel_mem_norm_std"] = float(rel_norm.std())

        if comm_mem is not None:
            if first_comm_mem is None:
                first_comm_mem = np.asarray(comm_mem, dtype=np.float32)
            last_comm_mem = np.asarray(comm_mem, dtype=np.float32)
            comm_norm = np.linalg.norm(comm_mem, axis=1)
            row["community_mem_norm_mean"] = float(comm_norm.mean())
            row["community_mem_norm_std"] = float(comm_norm.std())

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("epoch")
    df.to_csv(out_dir / "memory_evolution.csv", index=False)

    # Main trend plot
    plt.figure(figsize=(7, 4))
    for c, label in [
        ("node_mem_norm_mean", "node memory"),
        ("rel_mem_norm_mean", "relation memory"),
        ("community_mem_norm_mean", "community memory"),
    ]:
        if c in df.columns:
            plt.plot(df["epoch"], df[c], marker="o", label=label)
    plt.xlabel("epoch")
    plt.ylabel("L2 norm (mean)")
    plt.title("Memory Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "memory_evolution.png", dpi=180)
    plt.close()

    # Node type specific trend if available
    type_cols = [c for c in df.columns if c.startswith("node_type_") and c.endswith("_norm_mean")]
    if len(type_cols) > 0:
        plt.figure(figsize=(7, 4))
        for c in sorted(type_cols):
            plt.plot(df["epoch"], df[c], marker="o", label=c.replace("_norm_mean", ""))
        plt.xlabel("epoch")
        plt.ylabel("node memory L2 norm (mean)")
        plt.title("Node-type Memory Evolution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "memory_evolution_by_node_type.png", dpi=180)
        plt.close()

    _save_first_last_heatmap(first_node_mem, last_node_mem, "Node memory", out_dir / "node_memory_first_last_heatmap.png")
    _save_first_last_heatmap(first_rel_mem, last_rel_mem, "Relation memory", out_dir / "relation_memory_first_last_heatmap.png")
    _save_first_last_heatmap(first_comm_mem, last_comm_mem, "Community memory", out_dir / "community_memory_first_last_heatmap.png")

    print(f"saved memory evolution artifacts to {out_dir}")


if __name__ == "__main__":
    main()
