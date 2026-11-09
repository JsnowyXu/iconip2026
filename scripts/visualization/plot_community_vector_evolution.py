#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def pca_fit_transform(x: np.ndarray, k: int = 2):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    # SVD-based PCA, no sklearn dependency.
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comp = vt[:k].T
    z = xc @ comp
    return z.astype(np.float32), mu.astype(np.float32), comp.astype(np.float32)


def pairwise_cos_mean(mat: np.ndarray) -> float:
    if mat.shape[0] <= 1:
        return 1.0
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    u = mat / n
    c = u @ u.T
    iu = np.triu_indices(c.shape[0], k=1)
    return float(c[iu].mean()) if iu[0].size > 0 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser("Visualize community-vector evolution from memory_trace.pt")
    ap.add_argument("--trace", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--max-communities-plot", type=int, default=32)
    args = ap.parse_args()

    trace = torch.load(args.trace, map_location="cpu")
    snaps = trace.get("snapshots", [])
    if not snaps:
        raise RuntimeError(f"no snapshots found in {args.trace}")

    rows = []
    cms = []
    eps = []
    for s in snaps:
        cm = s.get("community_memory", None)
        if cm is None:
            continue
        cm = np.asarray(cm, dtype=np.float32)
        if cm.ndim != 2 or cm.shape[0] == 0:
            continue
        ep = int(s.get("epoch", len(eps) + 1))
        eps.append(ep)
        cms.append(cm)

    if not cms:
        raise RuntimeError("community_memory not found in snapshots")

    stack = np.concatenate(cms, axis=0)
    z_all, mu, comp = pca_fit_transform(stack, k=2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recover per-epoch projected points.
    z_by_ep = []
    st = 0
    for cm in cms:
        ed = st + cm.shape[0]
        z_by_ep.append(z_all[st:ed])
        st = ed

    # Epoch-wise summary stats.
    prev_cm = None
    prev_cent = None
    for ep, cm, z in zip(eps, cms, z_by_ep):
        cent = cm.mean(axis=0)
        drift = float(np.linalg.norm(cm - prev_cm, ord="fro") / max(1, cm.shape[0])) if prev_cm is not None and prev_cm.shape == cm.shape else 0.0
        cent2 = z.mean(axis=0)
        cent_step = float(np.linalg.norm(cent2 - prev_cent)) if prev_cent is not None else 0.0
        rows.append(
            {
                "epoch": ep,
                "num_communities": int(cm.shape[0]),
                "community_norm_mean": float(np.linalg.norm(cm, axis=1).mean()),
                "community_norm_std": float(np.linalg.norm(cm, axis=1).std()),
                "pairwise_cos_mean": pairwise_cos_mean(cm),
                "pca_centroid_x": float(cent2[0]),
                "pca_centroid_y": float(cent2[1]),
                "drift_from_prev": drift,
                "centroid_step": cent_step,
            }
        )
        prev_cm = cm.copy()
        prev_cent = cent2.copy()

    pd.DataFrame(rows).to_csv(out_dir / "community_vector_stats.csv", index=False)

    # Figure 1: first vs last epoch projection.
    z0 = z_by_ep[0]
    z1 = z_by_ep[-1]
    k = min(int(args.max_communities_plot), z0.shape[0], z1.shape[0])

    plt.figure(figsize=(6.6, 5.4))
    plt.scatter(z0[:k, 0], z0[:k, 1], s=30, alpha=0.7, label=f"epoch {eps[0]}", color="#4e79a7")
    plt.scatter(z1[:k, 0], z1[:k, 1], s=30, alpha=0.7, label=f"epoch {eps[-1]}", color="#e15759")
    for i in range(k):
        plt.plot([z0[i, 0], z1[i, 0]], [z0[i, 1], z1[i, 1]], color="#999999", alpha=0.25, linewidth=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Community Vectors: First vs Last Epoch")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "community_vectors_first_last.png", dpi=180)
    plt.close()

    # Figure 2: centroid trajectory.
    d = pd.DataFrame(rows)
    plt.figure(figsize=(6.0, 4.8))
    plt.plot(d["pca_centroid_x"], d["pca_centroid_y"], marker="o", color="#2a9d8f")
    for _, r in d.iterrows():
        plt.text(r["pca_centroid_x"], r["pca_centroid_y"], str(int(r["epoch"])), fontsize=8)
    plt.xlabel("Centroid PC1")
    plt.ylabel("Centroid PC2")
    plt.title("Community Centroid Trajectory")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "community_centroid_trajectory.png", dpi=180)
    plt.close()

    # Figure 3: drift diagnostics.
    plt.figure(figsize=(7.0, 4.0))
    plt.plot(d["epoch"], d["drift_from_prev"], marker="o", label="drift_from_prev", color="#f28e2b")
    plt.plot(d["epoch"], d["centroid_step"], marker="s", label="centroid_step", color="#76b7b2")
    plt.xlabel("Epoch")
    plt.ylabel("Drift")
    plt.title("Community Evolution Drift")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "community_drift_curve.png", dpi=180)
    plt.close()

    print(f"saved community vector visualizations to {out_dir}")


if __name__ == "__main__":
    main()
