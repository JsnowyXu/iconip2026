#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from best_edge_experiment_utils import FOCUS_DATASETS, FOCUS_TASK_TYPES, load_best_runs
from dhc.data.unified_dataset import load_unified_pt
from dhc.eval.evaluator import replay_history
from dhc.models.dhc_model import DHCModel


def ascii_safe_label(text: object, max_len: int = 36) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "unknown"
    clean = "".join(ch if ord(ch) < 128 else "_" for ch in raw)
    clean = " ".join(clean.split())
    if not clean.strip("_ "):
        clean = "node"
    if len(clean) > max_len:
        clean = clean[: max_len - 3] + "..."
    return clean


def load_model(run_dir: Path, data, device: torch.device) -> Tuple[DHCModel, Dict[str, object]]:
    model_stats = json.loads((run_dir / "model_stats.json").read_text(encoding="utf-8"))
    args = model_stats.get("args", {})

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
        use_recent_neighbors=not bool(args.get("no_recent_neighbors", False)),
        recent_neighbor_size=int(args.get("recent_neighbor_size", 10)),
        use_edge_type_features=not bool(args.get("no_edge_type_features", False)),
        neighbor_aggregation=str(args.get("neighbor_aggregation", "attn")),
    ).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, args


def build_node_label_maps(data_obj: Dict) -> Tuple[Dict[int, str], Dict[int, str]]:
    meta = data_obj.get("meta", {}) if isinstance(data_obj, dict) else {}
    extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
    node_type_map = extra.get("node_type_map", {}) if isinstance(extra, dict) else {}
    type_id_to_name = {}
    for name, idx in node_type_map.items():
        try:
            type_id_to_name[int(idx)] = str(name)
        except Exception:
            continue

    label_map: Dict[int, str] = {}
    for group_name in ["patent_attributes", "org_attributes", "codeIII_attributes"]:
        attrs = extra.get(group_name, {})
        node_ids = attrs.get("node_ids", [])
        if group_name == "patent_attributes":
            text = attrs.get("title", [])
            fallback = attrs.get("patent_id", [])
        elif group_name == "org_attributes":
            text = attrs.get("name", [])
            fallback = attrs.get("org_id", [])
        else:
            text = attrs.get("codeIII", [])
            fallback = attrs.get("description", [])
        for node_id, t, fb in zip(node_ids, text, fallback):
            t = str(t or "").strip()
            fb = str(fb or "").strip()
            label_map[int(node_id)] = t if t else (fb if fb else f"node_{int(node_id)}")

    return label_map, type_id_to_name


@torch.no_grad()
def precommunity_repr(model: DHCModel, node_ids: torch.Tensor) -> torch.Tensor:
    base_h = model._base_node_repr(node_ids)
    if model.use_recent_neighbors:
        query_ts = model.memory.get_node_last_update(node_ids)
        neigh_ctx = model._recent_neighbor_context(node_ids, base_h, query_ts)
        return model.node_fusion(torch.cat([base_h, neigh_ctx], dim=-1))
    return base_h


def batched_assignments(model: DHCModel, node_ids: torch.Tensor, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    hard_list: List[np.ndarray] = []
    maxprob_list: List[np.ndarray] = []
    for start in range(0, int(node_ids.numel()), batch_size):
        sub = node_ids[start : start + batch_size]
        h = precommunity_repr(model, sub)
        alpha = model.community.assign(h)
        hard_list.append(torch.argmax(alpha, dim=-1).detach().cpu().numpy().astype(np.int64))
        maxprob_list.append(torch.max(alpha, dim=-1).values.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(hard_list, axis=0), np.concatenate(maxprob_list, axis=0)


def _write_membership_artifacts(
    run_dir: Path,
    df: pd.DataFrame,
    out_dir: Path,
    top_members: int,
    size_prefix: str = "final",
) -> None:
    membership_name = f"{size_prefix}_node_membership.csv"
    sizes_name = f"{size_prefix}_community_sizes.csv"
    sizes_png = f"{size_prefix}_community_sizes.png"
    type_mix_name = f"{size_prefix}_community_type_mix.csv"
    top_members_name = f"top_{size_prefix}_community_members.csv" if size_prefix != "final" else "top_community_members.csv"

    df.to_csv(out_dir / membership_name, index=False)
    counts = df["community_id"].value_counts().sort_index()
    count_df = counts.rename_axis("community_id").reset_index(name="node_count")
    count_df.to_csv(out_dir / sizes_name, index=False)

    plt.figure(figsize=(8, 4.5))
    plt.bar(count_df["community_id"], count_df["node_count"])
    plt.xlabel("community id")
    plt.ylabel("assigned nodes")
    plt.title(f"{size_prefix.replace('_', ' ').title()} community sizes: {run_dir.name}")
    plt.tight_layout()
    plt.savefig(out_dir / sizes_png, dpi=180)
    plt.close()

    type_mix_df = (
        df.groupby(["community_id", "node_type_name"])
        .size()
        .rename("node_count")
        .reset_index()
        .sort_values(["community_id", "node_count"], ascending=[True, False])
    )
    type_mix_df.to_csv(out_dir / type_mix_name, index=False)

    top_rows: List[Dict[str, object]] = []
    for _, sub in df.groupby("community_id"):
        sub = sub.sort_values("assign_confidence", ascending=False).head(top_members)
        for _, row in sub.iterrows():
            top_rows.append(row.to_dict())
    pd.DataFrame(top_rows).to_csv(out_dir / top_members_name, index=False)


def render_final_membership_from_trace(
    run_dir: Path,
    data_obj: Dict,
    data,
    model: DHCModel,
    trace_obj: Dict[str, object],
    out_dir: Path,
    top_members: int,
) -> bool:
    snapshots = trace_obj.get("snapshots", []) if isinstance(trace_obj, dict) else []
    if not snapshots:
        return False
    last = snapshots[-1]
    monitor_node_ids = np.asarray(last.get("monitor_node_ids", []), dtype=np.int64)
    node_memory = np.asarray(last.get("node_memory", []), dtype=np.float32)
    community_memory = np.asarray(last.get("community_memory", []), dtype=np.float32)
    if monitor_node_ids.size == 0 or node_memory.size == 0 or community_memory.size == 0:
        return False

    label_map, type_id_to_name = build_node_label_maps(data_obj)
    node_ids_t = torch.from_numpy(monitor_node_ids).long().to(model.node_feat.device)
    mem_t = torch.from_numpy(node_memory).float().to(model.node_feat.device)
    comm_mem_t = torch.from_numpy(community_memory).float().to(model.node_feat.device)

    feat = model.node_feat[node_ids_t]
    typ = model.node_type_emb(model.node_type[node_ids_t].clamp_min(0))
    base_h = model.node_encoder(torch.cat([feat, typ, mem_t], dim=-1))
    q = model.community.node_to_comm(base_h)
    logits = (q @ comm_mem_t.t()) / math.sqrt(max(1, q.size(-1)))
    alpha = torch.softmax(logits, dim=-1)
    hard = torch.argmax(alpha, dim=-1).detach().cpu().numpy().astype(np.int64)
    conf = torch.max(alpha, dim=-1).values.detach().cpu().numpy().astype(np.float64)
    node_type = model.node_type[node_ids_t].detach().cpu().numpy().astype(np.int64)

    df = pd.DataFrame(
        {
            "node_id": monitor_node_ids.astype(np.int64),
            "node_type_id": node_type,
            "node_type_name": [type_id_to_name.get(int(x), f"type_{int(x)}") for x in node_type],
            "community_id": hard,
            "assign_confidence": conf,
            "node_label": [label_map.get(int(n), f"node_{int(n)}") for n in monitor_node_ids.tolist()],
        }
    )
    _write_membership_artifacts(run_dir, df, out_dir, top_members, size_prefix="final_monitor")
    return True


def render_final_membership(
    run_dir: Path,
    data_obj: Dict,
    data,
    model: DHCModel,
    out_dir: Path,
    top_members: int,
) -> None:
    node_ids = torch.arange(data.num_nodes, device=data.node_feat.device)
    hard, maxprob = batched_assignments(model, node_ids)
    label_map, type_id_to_name = build_node_label_maps(data_obj)
    node_type = data.node_type.detach().cpu().numpy().astype(np.int64)
    df = pd.DataFrame(
        {
            "node_id": node_ids.detach().cpu().numpy().astype(np.int64),
            "node_type_id": node_type,
            "node_type_name": [type_id_to_name.get(int(x), f"type_{int(x)}") for x in node_type],
            "community_id": hard,
            "assign_confidence": maxprob,
            "node_label": [label_map.get(int(n), f"node_{int(n)}") for n in node_ids.detach().cpu().numpy().tolist()],
        }
    )
    _write_membership_artifacts(run_dir, df, out_dir, top_members, size_prefix="final")


def render_monitor_evolution(run_dir: Path, data_obj: Dict, model: DHCModel, out_dir: Path) -> None:
    trace_path = run_dir / "memory_trace.pt"
    if not trace_path.exists():
        return
    trace_obj = torch.load(trace_path, map_location="cpu")
    snapshots = trace_obj.get("snapshots", [])
    if not snapshots:
        return

    label_map, type_id_to_name = build_node_label_maps(data_obj)
    rows: List[Dict[str, object]] = []
    prev_assign = None
    churn_rows: List[Dict[str, object]] = []
    for snap in snapshots:
        epoch = int(snap.get("epoch", 0))
        monitor_node_ids = np.asarray(snap.get("monitor_node_ids", []), dtype=np.int64)
        node_memory = np.asarray(snap.get("node_memory", []), dtype=np.float32)
        community_memory = np.asarray(snap.get("community_memory", []), dtype=np.float32)
        if monitor_node_ids.size == 0 or node_memory.size == 0 or community_memory.size == 0:
            continue

        monitor_ids_t = torch.from_numpy(monitor_node_ids).long().to(model.node_feat.device)
        mem_t = torch.from_numpy(node_memory).float().to(model.node_feat.device)
        feat = model.node_feat[monitor_ids_t]
        typ = model.node_type_emb(model.node_type[monitor_ids_t].clamp_min(0))
        base_h = model.node_encoder(torch.cat([feat, typ, mem_t], dim=-1))
        comm_mem = torch.from_numpy(community_memory).float().to(model.node_feat.device)

        q = model.community.node_to_comm(base_h)
        logits = (q @ comm_mem.t()) / math.sqrt(max(1, q.size(-1)))
        alpha = torch.softmax(logits, dim=-1)
        hard = torch.argmax(alpha, dim=-1).detach().cpu().numpy().astype(np.int64)
        conf = torch.max(alpha, dim=-1).values.detach().cpu().numpy().astype(np.float64)
        node_type_ids = model.node_type[monitor_ids_t].detach().cpu().numpy().astype(np.int64)
        for node_id, node_type_id, community_id, confidence in zip(monitor_node_ids, node_type_ids, hard, conf):
            rows.append(
                {
                    "epoch": epoch,
                    "node_id": int(node_id),
                    "node_type_id": int(node_type_id),
                    "node_type_name": type_id_to_name.get(int(node_type_id), f"type_{int(node_type_id)}"),
                    "community_id": int(community_id),
                    "assign_confidence": float(confidence),
                    "node_label": label_map.get(int(node_id), f"node_{int(node_id)}"),
                }
            )
        if prev_assign is not None:
            churn = float((hard != prev_assign).mean())
            churn_rows.append({"epoch": epoch, "membership_churn": churn})
        prev_assign = hard

    if not rows:
        return
    evo_df = pd.DataFrame(rows)
    evo_df.to_csv(out_dir / "monitor_assignment_evolution.csv", index=False)

    size_df = (
        evo_df.groupby(["epoch", "community_id"])
        .size()
        .rename("node_count")
        .reset_index()
        .sort_values(["epoch", "community_id"])
    )
    size_df.to_csv(out_dir / "monitor_community_sizes_by_epoch.csv", index=False)

    if churn_rows:
        churn_df = pd.DataFrame(churn_rows)
        churn_df.to_csv(out_dir / "monitor_assignment_churn.csv", index=False)
        plt.figure(figsize=(6.5, 4.0))
        plt.plot(churn_df["epoch"], churn_df["membership_churn"], marker="o")
        plt.xlabel("epoch")
        plt.ylabel("monitor-node community churn")
        plt.title(f"Community assignment churn: {run_dir.name}")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "monitor_assignment_churn.png", dpi=180)
        plt.close()

    heatmap = size_df.pivot(index="community_id", columns="epoch", values="node_count").fillna(0.0)
    if not heatmap.empty:
        plt.figure(figsize=(7.2, max(4.2, 0.18 * heatmap.shape[0] + 2.0)))
        plt.imshow(heatmap.values, aspect="auto", cmap="YlGnBu")
        plt.colorbar(label="monitor nodes")
        plt.yticks(range(len(heatmap.index)), [str(int(x)) for x in heatmap.index])
        plt.xticks(range(len(heatmap.columns)), [str(int(x)) for x in heatmap.columns])
        plt.xlabel("epoch")
        plt.ylabel("community id")
        plt.title(f"Monitor community occupancy heatmap: {run_dir.name}")
        plt.tight_layout()
        plt.savefig(out_dir / "monitor_community_heatmap.png", dpi=180)
        plt.close()

    drift_rows: List[Dict[str, object]] = []
    for node_id, sub in evo_df.sort_values(["node_id", "epoch"]).groupby("node_id"):
        community_ids = sub["community_id"].to_numpy(dtype=np.int64)
        if community_ids.size <= 1:
            num_changes = 0
        else:
            num_changes = int(np.sum(community_ids[1:] != community_ids[:-1]))
        drift_rows.append(
            {
                "node_id": int(node_id),
                "node_label": str(sub["node_label"].iloc[0]),
                "node_type_name": str(sub["node_type_name"].iloc[0]),
                "num_changes": num_changes,
                "num_epochs": int(sub["epoch"].nunique()),
                "first_community": int(sub["community_id"].iloc[0]),
                "last_community": int(sub["community_id"].iloc[-1]),
                "mean_confidence": float(sub["assign_confidence"].mean()),
            }
        )
    drift_df = pd.DataFrame(drift_rows).sort_values(
        ["num_changes", "mean_confidence", "node_id"],
        ascending=[False, True, True],
    )
    drift_df.to_csv(out_dir / "top_churn_nodes.csv", index=False)

    focus_nodes = drift_df.head(8)["node_id"].tolist()
    if focus_nodes:
        focus_df = evo_df[evo_df["node_id"].isin(focus_nodes)].copy()
        plt.figure(figsize=(8.0, 4.8))
        for node_id, sub in focus_df.sort_values(["node_id", "epoch"]).groupby("node_id"):
            label = ascii_safe_label(sub["node_label"].iloc[0])
            plt.plot(sub["epoch"], sub["community_id"], marker="o", linewidth=1.8, label=f"{label} ({int(node_id)})")
        plt.xlabel("epoch")
        plt.ylabel("community id")
        plt.title(f"High-churn member trajectories: {run_dir.name}")
        plt.grid(alpha=0.25)
        plt.legend(loc="best", fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "high_churn_member_trajectories.png", dpi=180)
        plt.close()

    top_epochs = sorted(evo_df["epoch"].unique().tolist())
    pick_epochs = [top_epochs[0], top_epochs[-1]] if len(top_epochs) > 1 else top_epochs
    focus = evo_df[evo_df["epoch"].isin(pick_epochs)].copy()
    if not focus.empty:
        pivot = (
            focus.groupby(["epoch", "community_id"])
            .size()
            .rename("node_count")
            .reset_index()
        )
        plt.figure(figsize=(7.5, 4.5))
        for epoch, sub in pivot.groupby("epoch"):
            plt.plot(sub["community_id"], sub["node_count"], marker="o", linewidth=2, label=f"epoch {int(epoch)}")
        plt.xlabel("community id")
        plt.ylabel("monitor nodes")
        plt.title(f"Monitor community sizes over time: {run_dir.name}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "monitor_community_size_evolution.png", dpi=180)
        plt.close()


def render_report(
    run_dir: Path,
    out_dir: Path,
    task_name: str,
    source_group: str,
    metrics_path: Path,
    membership_mode: str,
) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = metrics.get("test", {})
    lines = [
        "# Community Memory Case Study",
        "",
        f"- task: `{task_name}`",
        f"- source_group: `{source_group}`",
        f"- run_dir: `{run_dir}`",
        f"- metrics_path: `{metrics_path}`",
        f"- test_mrr: `{test.get('mrr')}`",
        f"- test_auc: `{test.get('auc')}`",
        f"- test_ap: `{test.get('ap')}`",
        f"- membership_mode: `{membership_mode}`",
        "",
        "Artifacts:",
        "- `final_node_membership.csv` / `final_monitor_node_membership.csv`",
        "- `final_community_sizes.csv` / `final_monitor_community_sizes.csv`",
        "- `final_community_type_mix.csv` / `final_monitor_community_type_mix.csv`",
        "- `top_community_members.csv` / `top_final_monitor_community_members.csv`",
        "- `monitor_assignment_evolution.csv`",
        "- `monitor_assignment_churn.csv` / `monitor_assignment_churn.png`",
        "- `monitor_community_sizes_by_epoch.csv` / `monitor_community_heatmap.png`",
        "- `top_churn_nodes.csv` / `high_churn_member_trajectories.png`",
        "- memory evolution plots from `plot_memory_evolution.py`",
        "- community vector plots from `plot_community_vector_evolution.py`",
        "",
        "Note:",
        "- Monitor-node epochwise assignments are estimated from snapshot node memories plus the saved community vectors.",
        "- `exact_replay` means final memberships were computed after replaying train+val history into the best checkpoint.",
        "- `monitor_snapshot` means final membership-style plots are derived from the last saved monitor-node snapshot for fast paper visualization.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser("Generate community-memory case-study artifacts from best runs")
    ap.add_argument("--project-root", type=Path, default=Path("/home/xuxj/2026/dyn_relation_memory"))
    ap.add_argument("--datasets", type=str, default=",".join(FOCUS_DATASETS))
    ap.add_argument("--task-types", type=str, default=",".join(FOCUS_TASK_TYPES))
    ap.add_argument("--top-members", type=int, default=8)
    ap.add_argument("--exact-replay", action="store_true", default=False)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/xuxj/2026/dyn_relation_memory/results/community_memory_case_study"),
    )
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    task_types = [x.strip() for x in args.task_types.split(",") if x.strip()]
    best_runs = load_best_runs(
        project_root=project_root,
        datasets=datasets,
        task_types=task_types,
        require_community_enabled=True,
        require_full_modules=True,
    )
    if not best_runs:
        raise SystemExit("No best full-module runs with community enabled found.")

    for best_run in best_runs:
        run_dir = best_run.run_dir
        task_out = out_root / best_run.task.task_name
        task_out.mkdir(parents=True, exist_ok=True)

        data = load_unified_pt(str(best_run.task.data_path), device=torch.device("cpu"))
        data_obj = torch.load(best_run.task.data_path, map_location="cpu")
        model, _ = load_model(run_dir=run_dir, data=data, device=torch.device("cpu"))
        trace_obj = {}
        mem_trace = run_dir / "memory_trace.pt"
        if mem_trace.exists():
            trace_obj = torch.load(mem_trace, map_location="cpu")

        membership_mode = "monitor_snapshot"
        used_snapshot = False
        if not bool(args.exact_replay) and trace_obj:
            used_snapshot = render_final_membership_from_trace(
                run_dir=run_dir,
                data_obj=data_obj,
                data=data,
                model=model,
                trace_obj=trace_obj,
                out_dir=task_out,
                top_members=int(args.top_members),
            )
        if not used_snapshot:
            hist_idx = torch.cat([data.train_idx, data.val_idx], dim=0)
            replay_history(model, data, history_idx=hist_idx, batch_size=512, device=torch.device("cpu"))
            render_final_membership(run_dir, data_obj, data, model, task_out, top_members=int(args.top_members))
            membership_mode = "exact_replay"

        render_monitor_evolution(run_dir, data_obj, model, task_out)

        if mem_trace.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "visualization" / "plot_memory_evolution.py"),
                    "--trace",
                    str(mem_trace),
                    "--out-dir",
                    str(task_out / "memory_evolution"),
                ],
                cwd=str(project_root),
                check=False,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(project_root / "scripts" / "visualization" / "plot_community_vector_evolution.py"),
                    "--trace",
                    str(mem_trace),
                    "--out-dir",
                    str(task_out / "community_vectors"),
                ],
                cwd=str(project_root),
                check=False,
            )

        render_report(
            run_dir=run_dir,
            out_dir=task_out,
            task_name=best_run.task.task_name,
            source_group=best_run.source_group,
            metrics_path=best_run.metrics_path,
            membership_mode=membership_mode,
        )
        print(f"[community-case-study] wrote {task_out}")

    print(out_root)


if __name__ == "__main__":
    main()
