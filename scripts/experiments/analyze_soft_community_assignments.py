#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dhc.data.unified_dataset import load_unified_pt
from dhc.eval.evaluator import replay_history
from dhc.models.dhc_model import DHCModel


def load_model(run_dir: Path, data):
    args = {}
    ms_path = run_dir / "model_stats.json"
    if ms_path.exists():
        model_stats = json.loads(ms_path.read_text(encoding="utf-8"))
        args = model_stats.get("args", {}) if isinstance(model_stats, dict) else {}

    num_multiclass = 0
    num_multilabel = 0
    if data.node_multiclass is not None:
        num_multiclass = int(data.node_multiclass["labels"].max().item()) + 1
    if data.node_multilabel is not None:
        num_multilabel = int(data.node_multilabel["labels"].shape[1])

    model = DHCModel(
        num_nodes=data.num_nodes,
        num_rels=data.num_rels,
        num_node_types=data.num_node_types,
        node_feat=data.node_feat,
        node_type=data.node_type,
        edge_feat_dim=data.edge_feat.shape[1],
        mem_dim=int(args.get("mem_dim", 128)),
        rel_dim=int(args.get("rel_dim", 64)),
        time_dim=int(args.get("time_dim", 32)),
        msg_dim=int(args.get("msg_dim", 128)),
        community_dim=int(args.get("community_dim", 128)),
        num_communities=int(args.get("num_communities", 64)),
        num_multiclass=num_multiclass,
        num_multilabel=num_multilabel,
        use_relation_memory=not bool(args.get("no_relation_memory", False)),
        use_community=not bool(args.get("no_community", False)),
        community_assign_temperature=float(args.get("community_assign_temperature", 1.0)),
        community_assign_topk=int(args.get("community_assign_topk", 0)),
    ).to(data.node_feat.device)

    state = torch.load(run_dir / "best_model.pt", map_location=data.node_feat.device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def compute_assignment_stats(model, data, node_ids: torch.Tensor):
    feat = data.node_feat[node_ids]
    typ = model.node_type_emb(data.node_type[node_ids].clamp_min(0))
    mem = model.memory.get_node_memory(node_ids)
    h = model.node_encoder(torch.cat([feat, typ, mem], dim=-1))
    alpha = model.community.assign(h)

    eps = 1e-12
    ent = -(alpha * torch.log(alpha + eps)).sum(dim=1)
    ent_norm = ent / np.log(max(2, alpha.shape[1]))
    top1, top1_idx = torch.max(alpha, dim=1)

    rows = {
        "n_nodes": int(node_ids.numel()),
        "n_communities": int(alpha.shape[1]),
        "entropy_mean": float(ent.mean().item()),
        "entropy_norm_mean": float(ent_norm.mean().item()),
        "entropy_norm_std": float(ent_norm.std().item()),
        "top1_prob_mean": float(top1.mean().item()),
        "top1_prob_std": float(top1.std().item()),
        "effective_communities@0.5": float((alpha >= 0.5).sum(dim=1).float().mean().item()),
        "effective_communities@0.2": float((alpha >= 0.2).sum(dim=1).float().mean().item()),
    }

    dom_hist = torch.bincount(top1_idx, minlength=alpha.shape[1]).detach().cpu().numpy().astype(np.int64)
    return rows, alpha.detach().cpu().numpy(), dom_hist, top1.detach().cpu().numpy(), top1_idx.detach().cpu().numpy(), ent_norm.detach().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser("Analyze soft community assignment rationality")
    ap.add_argument("--data", required=True, type=str)
    ap.add_argument("--run-dir", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--max-nodes", type=int, default=5000)
    args = ap.parse_args()

    device = torch.device("cpu")
    data = load_unified_pt(args.data, device=device)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(run_dir, data)

    # Replay history to get temporal states before analysis.
    hist = torch.cat([data.train_idx, data.val_idx], dim=0)
    replay_history(model, data, history_idx=hist, batch_size=512, device=device)

    node_ids = torch.arange(data.num_nodes, device=device)
    if int(args.max_nodes) > 0 and node_ids.numel() > int(args.max_nodes):
        node_ids = node_ids[: int(args.max_nodes)]

    global_stats, alpha_np, dom_hist, top1_np, top1_idx_np, ent_norm_np = compute_assignment_stats(model, data, node_ids)

    # Node-type breakdown.
    ntype = data.node_type[node_ids].detach().cpu().numpy().astype(np.int64)
    rows = []
    assign_rows = []
    for t in sorted(np.unique(ntype).tolist()):
        mask = np.where(ntype == t)[0]
        if mask.size == 0:
            continue
        a = alpha_np[mask]
        eps = 1e-12
        ent = -(a * np.log(a + eps)).sum(axis=1)
        ent_norm = ent / np.log(max(2, a.shape[1]))
        top1 = a.max(axis=1)
        rows.append(
            {
                "node_type": int(t),
                "count": int(mask.size),
                "entropy_norm_mean": float(ent_norm.mean()),
                "top1_prob_mean": float(top1.mean()),
                "effective_communities@0.2": float((a >= 0.2).sum(axis=1).mean()),
            }
        )
        assign_rows.append(
            {
                "node_type": int(t),
                "count": int(mask.size),
                **{f"community_{j}": float(a[:, j].mean()) for j in range(a.shape[1])},
            }
        )

    pd.DataFrame([global_stats]).to_csv(out_dir / "soft_community_global_stats.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "soft_community_by_node_type.csv", index=False)
    pd.DataFrame({"community_id": np.arange(dom_hist.shape[0]), "dominant_count": dom_hist}).to_csv(
        out_dir / "soft_community_dominant_hist.csv", index=False
    )
    pd.DataFrame(
        {
            "node_id": node_ids.detach().cpu().numpy().astype(np.int64),
            "node_type": ntype,
            "dominant_community": top1_idx_np.astype(np.int64),
            "top1_prob": top1_np.astype(np.float32),
            "entropy_norm": ent_norm_np.astype(np.float32),
        }
    ).to_csv(out_dir / "soft_community_node_assignments.csv", index=False)

    plt.figure(figsize=(6.5, 4.2))
    plt.hist(top1_np, bins=30, color="#4e79a7", alpha=0.85)
    plt.xlabel("top-1 assignment probability")
    plt.ylabel("nodes")
    plt.title("Community assignment confidence")
    plt.tight_layout()
    plt.savefig(out_dir / "soft_community_top1_prob_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.5, 4.2))
    plt.hist(ent_norm_np, bins=30, color="#f28e2b", alpha=0.85)
    plt.xlabel("normalized entropy")
    plt.ylabel("nodes")
    plt.title("Community assignment entropy")
    plt.tight_layout()
    plt.savefig(out_dir / "soft_community_entropy_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.bar(np.arange(dom_hist.shape[0]), dom_hist, color="#59a14f")
    plt.xlabel("community id")
    plt.ylabel("dominant-node count")
    plt.title("Dominant community histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "soft_community_dominant_hist.png", dpi=180)
    plt.close()

    if assign_rows:
        assign_df = pd.DataFrame(assign_rows).sort_values("node_type")
        assign_df.to_csv(out_dir / "soft_community_mean_assignment_by_node_type.csv", index=False)
        heat = assign_df[[c for c in assign_df.columns if c.startswith("community_")]].to_numpy(dtype=np.float32)
        fig_h = max(3.0, 0.45 * heat.shape[0] + 1.5)
        plt.figure(figsize=(8.5, fig_h))
        plt.imshow(heat, aspect="auto", cmap="viridis")
        plt.yticks(np.arange(assign_df.shape[0]), assign_df["node_type"].tolist())
        plt.xlabel("community id")
        plt.ylabel("node type")
        plt.title("Mean community assignment by node type")
        plt.colorbar(shrink=0.85)
        plt.tight_layout()
        plt.savefig(out_dir / "soft_community_node_type_heatmap.png", dpi=180)
        plt.close()

    md = [
        "# Soft Community Assignment Analysis",
        "",
        "## Global Stats",
        "",
        pd.DataFrame([global_stats]).to_markdown(index=False),
        "",
        "## By Node Type",
        "",
        (pd.DataFrame(rows).to_markdown(index=False) if rows else "(empty)"),
        "",
        "## Notes",
        "",
        "- Lower normalized entropy indicates more decisive community assignment.",
        "- Moderate `effective_communities@0.2` supports soft (not one-hot) communities.",
        "- Dominant histogram checks whether communities collapse to a few prototypes.",
        "- Figures: `soft_community_top1_prob_hist.png`, `soft_community_entropy_hist.png`, `soft_community_dominant_hist.png`, `soft_community_node_type_heatmap.png`.",
    ]
    (out_dir / "soft_community_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
