#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def component_sizes(num_nodes: int, src: np.ndarray, dst: np.ndarray, max_edges: int = 100000):
    parent = np.arange(num_nodes)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    e = min(max_edges, src.shape[0])
    for i in range(e):
        union(int(src[i]), int(dst[i]))

    roots = [find(i) for i in range(num_nodes)]
    c = Counter(roots)
    return np.array(sorted(c.values(), reverse=True))


def analyze_one(pt_path: Path, out_dir: Path, skip_plots: bool = False):
    obj = torch.load(pt_path, map_location="cpu")
    src = obj["edges"]["src"].numpy()
    dst = obj["edges"]["dst"].numpy()
    rel = obj["edges"]["rel"].numpy()
    ts = obj["edges"]["ts"].numpy()
    node_type = obj.get("node_type", torch.zeros(obj["num_nodes"], dtype=torch.long)).numpy()

    in_deg = np.bincount(dst, minlength=obj["num_nodes"])
    out_deg = np.bincount(src, minlength=obj["num_nodes"])
    deg = in_deg + out_deg

    name = obj.get("meta", {}).get("dataset_name", pt_path.stem)
    ddir = out_dir / name
    ddir.mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset": name,
        "num_nodes": int(obj["num_nodes"]),
        "num_edges": int(src.shape[0]),
        "num_relations": int(obj["num_rels"]),
        "num_node_types": int(obj.get("num_node_types", int(node_type.max()) + 1)),
        "time_min": float(ts.min()) if ts.size > 0 else 0.0,
        "time_max": float(ts.max()) if ts.size > 0 else 0.0,
        "time_span": float(ts.max() - ts.min()) if ts.size > 0 else 0.0,
        "edge_feat_dim": int(obj["edges"]["edge_feat"].shape[1]),
        "node_feat_dim": int(obj["node_feat"].shape[1]),
        "avg_degree": float(deg.mean()) if deg.size > 0 else 0.0,
        "max_degree": int(deg.max()) if deg.size > 0 else 0,
        "has_multiclass_label": obj.get("node_multiclass", None) is not None,
        "has_multilabel": obj.get("node_multilabel", None) is not None,
    }

    (ddir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    if not skip_plots:
        # Degree distribution
        plt.figure(figsize=(6, 4))
        plt.hist(np.log1p(deg), bins=50, color="#2a9d8f", alpha=0.85)
        plt.xlabel("log(1 + degree)")
        plt.ylabel("count")
        plt.title(f"Degree Distribution - {name}")
        plt.tight_layout()
        plt.savefig(ddir / "degree_distribution.png", dpi=160)
        plt.close()

        # Temporal density
        plt.figure(figsize=(6, 4))
        plt.hist(ts, bins=80, color="#e76f51", alpha=0.85)
        plt.xlabel("timestamp")
        plt.ylabel("edge count")
        plt.title(f"Temporal Density - {name}")
        plt.tight_layout()
        plt.savefig(ddir / "temporal_density.png", dpi=160)
        plt.close()

        # Relation distribution
        rel_count = np.bincount(rel, minlength=obj["num_rels"])
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(len(rel_count)), rel_count, color="#264653")
        plt.xlabel("relation id")
        plt.ylabel("edge count")
        plt.title(f"Relation Distribution - {name}")
        plt.tight_layout()
        plt.savefig(ddir / "relation_distribution.png", dpi=160)
        plt.close()

        # Node type distribution
        nt_count = np.bincount(node_type, minlength=int(node_type.max()) + 1)
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(len(nt_count)), nt_count, color="#f4a261")
        plt.xlabel("node type id")
        plt.ylabel("node count")
        plt.title(f"Node Type Distribution - {name}")
        plt.tight_layout()
        plt.savefig(ddir / "node_type_distribution.png", dpi=160)
        plt.close()

        # Community sketch by connected component size (top 20).
        comps = component_sizes(obj["num_nodes"], src, dst)
        topk = comps[:20]
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(len(topk)), topk, color="#457b9d")
        plt.xlabel("component rank")
        plt.ylabel("component size")
        plt.title(f"Community Sketch - {name}")
        plt.tight_layout()
        plt.savefig(ddir / "community_sketch.png", dpi=160)
        plt.close()

    return stats


def main():
    p = argparse.ArgumentParser("Analyze unified datasets and draw figures")
    p.add_argument("--inputs", nargs="+", required=True, help="List of unified .pt files")
    p.add_argument("--out-dir", type=str, default="data/reports")
    p.add_argument("--skip-plots", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for ip in args.inputs:
        all_stats.append(analyze_one(Path(ip), out_dir, skip_plots=args.skip_plots))

    df = pd.DataFrame(all_stats)
    df.to_csv(out_dir / "dataset_stats_summary.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
