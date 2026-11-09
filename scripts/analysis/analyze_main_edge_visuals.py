#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DATASETS = ("patent_us", "aminer_paper", "dblp", "thgl_software")


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    task_name: str
    task_path: Path
    preferred_run_dirs: tuple[Path, ...]


def relabel_dataset(name: str) -> str:
    return "patent-us" if name == "patent_us" else name


def safe_load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if x.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comp = vt[: min(2, vt.shape[0])].T
    z = x @ comp
    if z.shape[1] == 1:
        z = np.c_[z, np.zeros(z.shape[0])]
    return z[:, :2].astype(np.float32)


def relation_names(meta: dict[str, Any], num_rels: int) -> dict[int, str]:
    extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
    rel_map = extra.get("relation_map", {}) if isinstance(extra, dict) else {}
    inv = {int(v): str(k) for k, v in rel_map.items()} if isinstance(rel_map, dict) else {}
    dataset_name = str(meta.get("dataset_name", ""))
    if dataset_name.startswith("dblp"):
        inv.update({0: "author_write_paper", 1: "paper_cite_paper"})
    return {i: inv.get(i, f"rel_{i}") for i in range(num_rels)}


def node_type_names(meta: dict[str, Any], num_types: int) -> dict[int, str]:
    extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
    m = extra.get("node_type_map", {}) if isinstance(extra, dict) else {}
    inv = {int(v): str(k) for k, v in m.items()} if isinstance(m, dict) else {}
    dataset_name = str(meta.get("dataset_name", ""))
    if dataset_name.startswith("dblp"):
        inv.update({0: "paper", 1: "author"})
    return {i: inv.get(i, f"type_{i}") for i in range(num_types)}


def context_protocol(meta: dict[str, Any]) -> dict[str, Any]:
    extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
    if isinstance(extra, dict) and isinstance(extra.get("context_protocol"), dict):
        return extra["context_protocol"]
    return {}


def infer_task_rows(spec: TaskSpec, task: dict[str, Any]) -> list[dict[str, Any]]:
    meta = task.get("meta", {})
    edges = task["edges"]
    rel = as_numpy(edges["rel"]).astype(np.int64)
    src = as_numpy(edges["src"]).astype(np.int64)
    dst = as_numpy(edges["dst"]).astype(np.int64)
    labels = as_numpy(edges.get("edge_label", np.ones_like(rel))).astype(np.float32)
    node_type = as_numpy(task.get("node_type", np.full(int(task["num_nodes"]), -1))).astype(np.int64)
    rnames = relation_names(meta, int(task.get("num_rels", int(rel.max()) + 1)))
    nnames = node_type_names(meta, int(task.get("num_node_types", int(node_type.max()) + 1)))
    protocol = context_protocol(meta)
    target_rel_ids = protocol.get("target_original_rel_ids")
    if target_rel_ids is None and "target_original_rel_id" in protocol:
        target_rel_ids = [protocol["target_original_rel_id"]]
    if target_rel_ids is None:
        target_rel_id = meta.get("extra", {}).get("target_original_rel_id") if isinstance(meta.get("extra", {}), dict) else None
        target_rel_ids = [target_rel_id] if target_rel_id is not None else sorted(set(rel[labels > 0.5].tolist()))
    target_rel_ids = [int(x) for x in target_rel_ids]

    rows = []
    for rid in sorted(set(rel.tolist())):
        mask = rel == rid
        sup_mask = mask & (labels > 0.5)
        src_types = pd.Series(node_type[src[mask]]).value_counts().to_dict() if mask.any() else {}
        dst_types = pd.Series(node_type[dst[mask]]).value_counts().to_dict() if mask.any() else {}
        top_src_type = max(src_types, key=src_types.get) if src_types else -1
        top_dst_type = max(dst_types, key=dst_types.get) if dst_types else -1
        rows.append(
            {
                "dataset": spec.dataset,
                "dataset_label": relabel_dataset(spec.dataset),
                "task_name": spec.task_name,
                "task_path": str(spec.task_path),
                "relation_id": rid,
                "relation_name": rnames.get(rid, f"rel_{rid}"),
                "is_target_relation": rid in target_rel_ids,
                "edge_count": int(mask.sum()),
                "supervised_edge_count": int(sup_mask.sum()),
                "dominant_src_type": nnames.get(int(top_src_type), f"type_{top_src_type}"),
                "dominant_dst_type": nnames.get(int(top_dst_type), f"type_{top_dst_type}"),
                "task_phrase": f"{nnames.get(int(top_src_type), f'type_{top_src_type}')} -> {nnames.get(int(top_dst_type), f'type_{top_dst_type}')} {rnames.get(rid, f'rel_{rid}')}",
            }
        )
    return rows


def split_rows(spec: TaskSpec, task: dict[str, Any]) -> list[dict[str, Any]]:
    edges = task["edges"]
    rel = as_numpy(edges["rel"]).astype(np.int64)
    labels = as_numpy(edges.get("edge_label", np.ones_like(rel))).astype(np.float32)
    meta = task.get("meta", {})
    rnames = relation_names(meta, int(task.get("num_rels", int(rel.max()) + 1)))
    rows = []
    for split, idx in task.get("split", {}).items():
        idx = as_numpy(idx).astype(np.int64)
        for rid in sorted(set(rel[idx].tolist())):
            m = idx[rel[idx] == rid]
            rows.append(
                {
                    "dataset": spec.dataset,
                    "task_name": spec.task_name,
                    "split": split,
                    "relation_id": int(rid),
                    "relation_name": rnames.get(int(rid), f"rel_{rid}"),
                    "edge_count": int(len(m)),
                    "supervised_edge_count": int((labels[m] > 0.5).sum()),
                }
            )
    return rows


def find_tasks(root: Path, datasets: Iterable[str]) -> list[TaskSpec]:
    p = root / "data" / "processed"
    result = []
    candidates = {
        "patent_us": [
            (p / "tasks" / "patent_us_edge_org_codeIII_inductive.pt", root / "results" / "edge_only_formal" / "main" / "patent_us_edge_org_codeIII_inductive__seed42", root / "results" / "patent_dhc_formal" / "patent_us_edge_org_codeIII_inductive__seed42"),
            (p / "tasks" / "patent_us_edge_org_codeIII_transductive.pt", root / "results" / "edge_only_formal" / "main" / "patent_us_edge_org_codeIII_transductive__seed42", root / "results" / "patent_dhc_formal" / "patent_us_edge_org_codeIII_transductive__seed42"),
        ],
        "aminer_paper": [
            (p / "paper_tasks" / "aminer_paper_citation_edge_inductive.pt", root / "results" / "edge_only_formal" / "main" / "aminer_paper_citation_edge_inductive__seed42", root / "results" / "aminer_dhc_formal" / "aminer_paper_citation_edge_inductive__seed42"),
            (p / "paper_tasks" / "aminer_paper_citation_edge_transductive.pt", root / "results" / "edge_only_formal" / "main" / "aminer_paper_citation_edge_transductive__seed42", root / "results" / "aminer_dhc_formal" / "aminer_paper_citation_edge_transductive__seed42"),
        ],
        "dblp": [
            (p / "paper_tasks" / "dblp_edge_inductive.pt", root / "results" / "edge_only_formal" / "main" / "dblp_edge_inductive__seed42", root / "results" / "paper_dhc_formal" / "dblp_edge_inductive__full__s42"),
            (p / "paper_tasks" / "dblp_edge_transductive.pt", root / "results" / "edge_only_formal" / "main" / "dblp_edge_transductive__seed42", root / "results" / "paper_dhc_formal" / "dblp_edge_transductive__full__s42"),
        ],
        "thgl_software": [
            (p / "tgb2_tasks" / "thgl_software_edge_inductive.pt", root / "results" / "edge_only_formal" / "main" / "thgl_software_edge_inductive__seed42", root / "results" / "best_config_module_ablation" / "thgl_software_edge_inductive__full__s42"),
            (p / "tgb2_tasks" / "thgl_software_edge_transductive.pt", root / "results" / "edge_only_formal" / "main" / "thgl_software_edge_transductive__seed42", root / "results" / "best_config_module_ablation" / "thgl_software_edge_transductive__full__s42"),
        ],
    }
    for dataset in datasets:
        for task_path, *run_dirs in candidates.get(dataset, []):
            if task_path.exists():
                result.append(TaskSpec(dataset, task_path.stem, task_path, tuple(d for d in run_dirs if d.exists())))
    return result


def choose_run_dir(spec: TaskSpec) -> Path | None:
    for d in spec.preferred_run_dirs:
        if (d / "memory_trace.pt").exists():
            return d
    return None


def export_memory_projection(spec: TaskSpec, task: dict[str, Any], run_dir: Path, out_dir: Path, kind: str) -> bool:
    trace_path = run_dir / "memory_trace.pt"
    trace = safe_load_pt(trace_path)
    snaps = trace.get("snapshots", [])
    mats = []
    labels = []
    for snap in snaps:
        mat = snap.get(kind)
        if mat is None:
            continue
        mat = as_numpy(mat).astype(np.float32)
        if mat.ndim != 2 or mat.shape[0] == 0:
            continue
        ep = int(snap.get("epoch", len(mats) + 1))
        ids = snap.get("monitor_node_ids") if kind == "node_memory" else None
        ids = as_numpy(ids).astype(np.int64) if ids is not None else np.arange(mat.shape[0], dtype=np.int64)
        mats.append(mat)
        labels.extend((ep, int(i)) for i in ids[: mat.shape[0]])
    if not mats:
        return False

    stack = np.concatenate(mats, axis=0)
    z = pca_2d(stack)
    rows = []
    node_type = as_numpy(task.get("node_type", np.array([], dtype=np.int64))).astype(np.int64)
    ntype_names = node_type_names(task.get("meta", {}), int(task.get("num_node_types", 0)))
    rnames = relation_names(task.get("meta", {}), int(task.get("num_rels", 0)))
    for (ep, slot), xy, vec in zip(labels, z, stack):
        label = f"slot_{slot}"
        if kind == "node_memory" and slot < len(node_type):
            label = ntype_names.get(int(node_type[slot]), f"type_{int(node_type[slot])}")
        if kind == "rel_memory":
            label = rnames.get(slot, f"rel_{slot}")
        rows.append(
            {
                "dataset": spec.dataset,
                "task_name": spec.task_name,
                "run_dir": str(run_dir),
                "memory_kind": kind,
                "epoch": ep,
                "slot": slot,
                "label": label,
                "pc1": float(xy[0]),
                "pc2": float(xy[1]),
                "l2_norm": float(np.linalg.norm(vec)),
            }
        )
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{kind}_pca2d.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    epochs = sorted(df["epoch"].unique().tolist())
    scatter = ax.scatter(df["pc1"], df["pc2"], c=df["epoch"], cmap="viridis", s=22 if kind != "node_memory" else 34, alpha=0.72)
    if kind == "rel_memory":
        for slot, g in df.groupby("slot"):
            g = g.sort_values("epoch")
            ax.plot(g["pc1"], g["pc2"], color="#64748b", alpha=0.28, linewidth=0.8)
            ax.text(g.iloc[-1]["pc1"], g.iloc[-1]["pc2"], str(g.iloc[-1]["label"]), fontsize=8)
    ax.set_title(f"{spec.task_name}: {kind} all-epoch PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("epoch")
    fig.tight_layout()
    fig.savefig(out_dir / f"{kind}_pca2d_all_epochs.png", dpi=180)
    plt.close(fig)

    first_ep, last_ep = int(df["epoch"].min()), int(df["epoch"].max())
    plot = df[df["epoch"].isin([first_ep, last_ep])].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for ep, g in plot.groupby("epoch"):
        ax.scatter(g["pc1"], g["pc2"], s=30 if kind != "node_memory" else 42, alpha=0.78, label=f"epoch {ep}")
    if kind != "node_memory":
        for slot, g in plot.groupby("slot"):
            if len(g) == 2:
                g = g.sort_values("epoch")
                ax.plot(g["pc1"], g["pc2"], color="#8b8f96", alpha=0.35, linewidth=0.9)
                ax.text(g.iloc[-1]["pc1"], g.iloc[-1]["pc2"], str(g.iloc[-1]["label"]), fontsize=8)
    ax.set_title(f"{spec.task_name}: {kind} PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{kind}_pca2d_first_last.png", dpi=180)
    plt.close(fig)

    cent = df.groupby("epoch", as_index=False)[["pc1", "pc2", "l2_norm"]].mean()
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(cent["pc1"], cent["pc2"], marker="o", color="#2563eb")
    for _, row in cent.iterrows():
        ax.text(row["pc1"], row["pc2"], str(int(row["epoch"])), fontsize=8)
    ax.set_title(f"{spec.task_name}: {kind} centroid trajectory")
    ax.set_xlabel("Centroid PC1")
    ax.set_ylabel("Centroid PC2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{kind}_centroid_trajectory.png", dpi=180)
    plt.close(fig)

    norm = df.pivot_table(index="epoch", columns="label", values="l2_norm", aggfunc="mean").sort_index()
    norm.to_csv(out_dir / f"{kind}_norm_by_epoch.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    if kind == "node_memory":
        for label in norm.columns[: min(8, len(norm.columns))]:
            ax.plot(norm.index, norm[label], marker="o", linewidth=1.6, label=str(label))
        ax.set_title(f"{spec.task_name}: node memory norm by monitored type")
    elif kind == "rel_memory":
        for label in norm.columns:
            ax.plot(norm.index, norm[label], marker="o", linewidth=1.5, label=str(label))
        ax.set_title(f"{spec.task_name}: relation_type memory norm")
    else:
        top_cols = norm.iloc[-1].sort_values(ascending=False).head(12).index.tolist()
        for label in top_cols:
            ax.plot(norm.index, norm[label], linewidth=1.1, alpha=0.8)
        ax.set_title(f"{spec.task_name}: top community memory norms")
    ax.set_xlabel("epoch")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.2)
    if kind != "community_memory":
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{kind}_norm_by_epoch.png", dpi=180)
    plt.close(fig)
    return True


def mirror_relation_type_artifacts(task_out: Path) -> None:
    src = task_out / "rel_memory"
    dst = task_out / "relation_type"
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    mapping = {
        "rel_memory_pca2d.csv": "relation_type_pca2d.csv",
        "rel_memory_pca2d_all_epochs.png": "relation_type_pca2d_all_epochs.png",
        "rel_memory_pca2d_first_last.png": "relation_type_pca2d_first_last.png",
        "rel_memory_centroid_trajectory.png": "relation_type_centroid_trajectory.png",
        "rel_memory_norm_by_epoch.csv": "relation_type_norm_by_epoch.csv",
        "rel_memory_norm_by_epoch.png": "relation_type_norm_by_epoch.png",
    }
    for src_name, dst_name in mapping.items():
        sp = src / src_name
        if sp.exists():
            (dst / dst_name).write_bytes(sp.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_main_metrics(specs: list[TaskSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        run = choose_run_dir(spec)
        if run is None:
            continue
        metrics = read_json(run / "metrics.json")
        stats = read_json(run / "model_stats.json")
        hist_path = run / "train_history.csv"
        hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
        best_epoch = None
        best_val = None
        if not hist.empty and "val_mrr" in hist.columns:
            i = pd.to_numeric(hist["val_mrr"], errors="coerce").idxmax()
            best_epoch = int(hist.loc[i, "epoch"])
            best_val = float(hist.loc[i, "val_mrr"])
        test = metrics.get("test", {}) if isinstance(metrics.get("test"), dict) else {}
        rows.append(
            {
                "dataset": spec.dataset,
                "task_name": spec.task_name,
                "model": "DHC-main",
                "run_dir": str(run),
                "epochs_run": int(hist["epoch"].max()) if not hist.empty and "epoch" in hist else np.nan,
                "best_epoch": best_epoch,
                "best_val_mrr": best_val,
                "final_loss": float(hist["loss"].iloc[-1]) if not hist.empty and "loss" in hist else np.nan,
                "avg_epoch_sec": float(hist["epoch_sec"].mean()) if not hist.empty and "epoch_sec" in hist else np.nan,
                "total_sec": float(hist["epoch_sec"].sum()) if not hist.empty and "epoch_sec" in hist else np.nan,
                "test_mrr": test.get("mrr", np.nan),
                "test_auc": test.get("auc", np.nan),
                "test_ap": test.get("ap", np.nan),
                "total_params": stats.get("total_params", np.nan),
                "trainable_params": stats.get("trainable_params", np.nan),
            }
        )
    return rows


def parse_baselines(root: Path, datasets: set[str]) -> pd.DataFrame:
    rows = []
    manifest_path = root / "results" / "edge_only_formal" / "baselines" / "run_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    metrics_long = root / "results" / "edge_only_formal" / "baselines" / "metrics_long.csv"
    if metrics_long.exists():
        df = pd.read_csv(metrics_long)
        df = df[df["dataset"].isin(datasets) & df["model"].astype(str).str.lower().isin(["tgn", "tgn_edgetype", "repeatmixer", "sthn"])]
        for _, r in df.iterrows():
            rows.append(
                {
                    "dataset": r.get("dataset"),
                    "task_name": r.get("task_name"),
                    "model": r.get("model"),
                    "split": r.get("split"),
                    "test_mrr": r.get("test_mrr"),
                    "test_auc": r.get("test_auc"),
                    "test_ap": r.get("test_ap"),
                    "avg_epoch_sec": r.get("avg_epoch_sec"),
                    "total_sec": r.get("total_sec"),
                    "total_params": r.get("total_params"),
                    "trainable_params": r.get("trainable_params"),
                    "source_path": r.get("source_path"),
                }
            )
    for tami_root in [root / "results" / "tami_baseline_formal_20260508"]:
        if not tami_root.exists():
            continue
        for csv_path in sorted(tami_root.glob("*.csv")):
            if csv_path.name in {"summary.csv", "test_summary.csv", "run_manifest.csv"} or csv_path.name.startswith("_"):
                continue
            task = csv_path.stem
            dataset = next((d for d in datasets if task.startswith(d)), None)
            if dataset is None:
                continue
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "task_name": task,
                        "model": "tami",
                        "split": r.get("split"),
                        "test_mrr": r.get("mrr") if r.get("split") == "test" else np.nan,
                        "test_auc": r.get("auc") if r.get("split") == "test" else np.nan,
                        "test_ap": r.get("ap") if r.get("split") == "test" else np.nan,
                        "avg_epoch_sec": r.get("avg_epoch_sec"),
                        "total_sec": np.nan,
                        "total_params": r.get("total_params"),
                        "trainable_params": r.get("trainable_params"),
                        "source_path": str(csv_path),
                    }
                )
    if not manifest.empty:
        man = manifest[["name", "elapsed_sec"]].copy()
        for row in rows:
            src = str(row.get("source_path", ""))
            name = Path(src).stem
            hit = man[man["name"] == name]
            if not hit.empty:
                row["total_sec"] = float(hit.iloc[0]["elapsed_sec"])
    return pd.DataFrame(rows)


def _parse_key_value_epoch_lines(path: Path, model: str, dataset: str, task_name: str) -> list[dict[str, Any]]:
    rows = []
    pat = re.compile(
        r"epoch=(?P<epoch>\d+)\s+loss=(?P<loss>[-+0-9.eE]+)\s+"
        r"val_mrr=(?P<val_mrr>[-+0-9.eE]+)\s+val_auc=(?P<val_auc>[-+0-9.eE]+)\s+"
        r"val_ap=(?P<val_ap>[-+0-9.eE]+)\s+sec=(?P<epoch_sec>[-+0-9.eE]+)"
    )
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if not m:
            continue
        d = {k: float(v) for k, v in m.groupdict().items()}
        d["epoch"] = int(d["epoch"])
        rows.append({"dataset": dataset, "task_name": task_name, "model": model, "log_path": str(path), **d})
    return rows


def _parse_sthn_log(path: Path, dataset: str, task_name: str) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    epoch = None
    current: dict[str, Any] = {}
    epoch_pat = re.compile(r">>> Epoch\s+(\d+)")
    train_pat = re.compile(r"train mode with time\s+([-+0-9.eE]+), AUROC\s+([-+0-9.eE]+), AUPRC\s+([-+0-9.eE]+), loss\s+([-+0-9.eE]+)")
    valid_pat = re.compile(r"valid mode with time\s+([-+0-9.eE]+), AUROC\s+([-+0-9.eE]+), AUPRC\s+([-+0-9.eE]+), loss\s+([-+0-9.eE]+)")
    val_mrr_pat = re.compile(r"val:\s*mrr\s+([-+0-9.eE]+)", re.IGNORECASE)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = epoch_pat.search(line)
        if m:
            if current:
                rows.append(current)
            epoch = int(m.group(1))
            current = {"dataset": dataset, "task_name": task_name, "model": "sthn", "log_path": str(path), "epoch": epoch}
            continue
        if not current:
            continue
        m = train_pat.search(line)
        if m:
            current.update(
                {
                    "train_sec": float(m.group(1)),
                    "train_auc": float(m.group(2)),
                    "train_ap": float(m.group(3)),
                    "loss": float(m.group(4)),
                }
            )
            current["epoch_sec"] = current.get("train_sec", 0.0) + current.get("valid_sec", 0.0)
            continue
        m = valid_pat.search(line)
        if m:
            current.update(
                {
                    "valid_sec": float(m.group(1)),
                    "val_auc": float(m.group(2)),
                    "val_ap": float(m.group(3)),
                    "val_loss": float(m.group(4)),
                }
            )
            current["epoch_sec"] = current.get("train_sec", 0.0) + current.get("valid_sec", 0.0)
            continue
        m = val_mrr_pat.search(line)
        if m:
            current["val_mrr"] = float(m.group(1))
    if current:
        rows.append(current)
    return rows


def parse_baseline_epoch_logs(root: Path, datasets: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    log_root = root / "results" / "edge_only_formal" / "baselines" / "logs"
    sthn_metric_root = root / "results" / "edge_only_formal" / "baselines" / "metrics"
    for model in ["tgn", "tgn_edgetype", "repeatmixer"]:
        for path in sorted(log_root.glob(f"{model}__*.log")):
            task = path.stem.split("__", 1)[1]
            dataset = next((d for d in datasets if task.startswith(d)), None)
            if dataset is None:
                continue
            rows.extend(_parse_key_value_epoch_lines(path, model, dataset, task))
    for path in sorted(sthn_metric_root.glob("sthn__*.log")):
        task = path.stem.split("__", 1)[1]
        dataset = next((d for d in datasets if task.startswith(d)), None)
        if dataset is None:
            continue
        rows.extend(_parse_sthn_log(path, dataset, task))
    tami_log_root = root / "results" / "tami_baseline_formal_20260508" / "logs"
    for path in sorted(tami_log_root.glob("*.log")):
        task = path.stem
        dataset = next((d for d in datasets if task.startswith(d)), None)
        if dataset is None:
            continue
        rows.extend(_parse_key_value_epoch_lines(path, "tami", dataset, task))
    return pd.DataFrame(rows)


def export_baseline_epoch_curves(epoch_df: pd.DataFrame, out_root: Path) -> None:
    if epoch_df.empty:
        return
    epoch_df.to_csv(out_root / "baseline_epoch_curves.csv", index=False)
    for (dataset, task), gtask in epoch_df.groupby(["dataset", "task_name"]):
        out_dir = out_root / dataset / task / "performance"
        out_dir.mkdir(parents=True, exist_ok=True)
        gtask.to_csv(out_dir / "baseline_epoch_curves.csv", index=False)
        fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
        for model, g in gtask.groupby("model"):
            g = g.sort_values("epoch")
            if "loss" in g and g["loss"].notna().any():
                ax1.plot(g["epoch"], g["loss"], marker="o", linewidth=1.5, label=f"{model} loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        for model, g in gtask.groupby("model"):
            g = g.sort_values("epoch")
            if "val_mrr" in g and g["val_mrr"].notna().any():
                ax2.plot(g["epoch"], g["val_mrr"], marker="s", linestyle="--", linewidth=1.3, label=f"{model} val_mrr")
        ax2.set_ylabel("val MRR")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
        ax1.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_loss_val_mrr_curves.png", dpi=180)
        plt.close(fig)

        time_g = gtask[pd.to_numeric(gtask.get("epoch_sec"), errors="coerce").notna()].copy()
        if not time_g.empty:
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            for model, g in time_g.groupby("model"):
                g = g.sort_values("epoch")
                ax.plot(g["epoch"], g["epoch_sec"], marker="o", linewidth=1.5, label=f"{model} epoch_sec")
            ax.set_title(f"{task}: baseline epoch time")
            ax.set_xlabel("epoch")
            ax.set_ylabel("seconds / epoch")
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_dir / "baseline_epoch_time_curves.png", dpi=180)
            plt.close(fig)


def edge_semantic_importance(root: Path, datasets: set[str]) -> pd.DataFrame:
    rows = []
    for base in [root / "results" / "edge_memory_ablation", root / "results" / "best_config_module_ablation"]:
        if not base.exists():
            continue
        for d in sorted(base.glob("*")):
            if not d.is_dir() or "__" not in d.name:
                continue
            dataset = next((x for x in datasets if d.name.startswith(x)), None)
            if dataset is None:
                continue
            metrics = read_json(d / "metrics.json")
            if not metrics:
                continue
            task = d.name.split("__")[0]
            variant = "__".join(d.name.split("__")[1:])
            test = metrics.get("test", {}) if isinstance(metrics.get("test"), dict) else {}
            rows.append(
                {
                    "dataset": dataset,
                    "task_name": task,
                    "variant": variant,
                    "test_mrr": test.get("mrr", np.nan),
                    "test_auc": test.get("auc", np.nan),
                    "test_ap": test.get("ap", np.nan),
                    "source_dir": str(d),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    full = df[df["variant"].str.contains("full", regex=False)].rename(columns={"test_mrr": "full_test_mrr"})
    key = ["dataset", "task_name"]
    full = full[key + ["full_test_mrr"]].drop_duplicates(key)
    df = df.merge(full, on=key, how="left")
    df["delta_mrr_vs_full"] = df["test_mrr"] - df["full_test_mrr"]
    return df


def export_training_curves(specs: list[TaskSpec], out_root: Path) -> None:
    for spec in specs:
        run = choose_run_dir(spec)
        if run is None or not (run / "train_history.csv").exists():
            continue
        hist = pd.read_csv(run / "train_history.csv")
        out_dir = out_root / spec.dataset / spec.task_name / "performance"
        out_dir.mkdir(parents=True, exist_ok=True)
        hist.to_csv(out_dir / "main_train_history.csv", index=False)
        fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
        if "loss" in hist:
            ax1.plot(hist["epoch"], hist["loss"], marker="o", label="loss", color="#1f2937")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        for c, color in [("val_mrr", "#2563eb"), ("val_auc", "#059669"), ("val_ap", "#dc2626")]:
            if c in hist:
                ax2.plot(hist["epoch"], hist[c], marker="s", label=c, color=color, alpha=0.8)
        ax2.set_ylabel("validation metric")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        ax1.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "main_loss_metric_curve.png", dpi=180)
        plt.close(fig)

        if "epoch_sec" in hist:
            d = hist.copy()
            d["cum_sec"] = pd.to_numeric(d["epoch_sec"], errors="coerce").fillna(0).cumsum()
            fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
            ax1.plot(d["epoch"], d["epoch_sec"], marker="o", color="#059669", label="epoch_sec")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("seconds / epoch")
            ax2 = ax1.twinx()
            ax2.plot(d["epoch"], d["cum_sec"], marker="s", color="#7c3aed", label="cumulative_sec")
            ax2.set_ylabel("cumulative seconds")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="best")
            ax1.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_dir / "main_epoch_time_curve.png", dpi=180)
            plt.close(fig)


def export_task_relation_plots(task_df: pd.DataFrame, split_df: pd.DataFrame, out_root: Path) -> None:
    if task_df.empty:
        return
    for (dataset, task), g in task_df.groupby(["dataset", "task_name"]):
        out_dir = out_root / dataset / task / "task_definition"
        out_dir.mkdir(parents=True, exist_ok=True)
        g.to_csv(out_dir / "task_relation_counts.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        colors = ["#dc2626" if bool(x) else "#2563eb" for x in g["is_target_relation"]]
        ax.bar(g["relation_name"], g["edge_count"], color=colors, alpha=0.86, label="all context edges")
        ax.bar(g["relation_name"], g["supervised_edge_count"], color="#f97316", alpha=0.62, label="supervised target positives")
        ax.set_title(f"{task}: relation composition")
        ax.set_xlabel("relation type")
        ax.set_ylabel("edge count")
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "relation_composition.png", dpi=180)
        plt.close(fig)

        sg = split_df[(split_df["dataset"] == dataset) & (split_df["task_name"] == task)].copy()
        if sg.empty:
            continue
        sg.to_csv(out_dir / "relation_split_counts.csv", index=False)
        piv = sg.pivot_table(index="relation_name", columns="split", values="supervised_edge_count", aggfunc="sum", fill_value=0)
        for col in ["train", "val", "test"]:
            if col not in piv.columns:
                piv[col] = 0
        piv = piv[["train", "val", "test"]]
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        piv.plot(kind="bar", stacked=True, ax=ax, color=["#2563eb", "#f59e0b", "#10b981"])
        ax.set_title(f"{task}: supervised positives by split")
        ax.set_xlabel("relation type")
        ax.set_ylabel("supervised edge count")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "supervised_edges_by_split.png", dpi=180)
        plt.close(fig)


def export_performance_plots(perf: pd.DataFrame, out_root: Path) -> None:
    if perf.empty:
        return
    perf.to_csv(out_root / "performance_summary.csv", index=False)
    test_rows = perf.copy()
    if "split" in test_rows.columns:
        test_rows = test_rows[(test_rows["split"].isna()) | (test_rows["split"] == "test")]
    test_rows = test_rows[pd.to_numeric(test_rows["test_mrr"], errors="coerce").notna()].copy()
    if test_rows.empty:
        return
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric, fname, ylabel in [
        ("test_mrr", "overview_test_mrr_by_model.png", "test MRR"),
        ("avg_epoch_sec", "overview_avg_epoch_sec_by_model.png", "seconds / epoch"),
    ]:
        dd = test_rows[pd.to_numeric(test_rows.get(metric), errors="coerce").notna()].copy()
        if dd.empty:
            continue
        dd[metric] = pd.to_numeric(dd[metric], errors="coerce")
        piv = dd.pivot_table(index="task_name", columns="model", values=metric, aggfunc="max")
        fig, ax = plt.subplots(figsize=(13.0, 6.2))
        piv.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(fname.replace("_", " ").replace(".png", ""))
        ax.set_xlabel("task")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=180)
        plt.close(fig)

    for (dataset, task), g in test_rows.groupby(["dataset", "task_name"]):
        out_dir = out_root / dataset / task / "performance"
        out_dir.mkdir(parents=True, exist_ok=True)
        gg = g.copy()
        gg["test_mrr"] = pd.to_numeric(gg["test_mrr"], errors="coerce")
        gg["avg_epoch_sec"] = pd.to_numeric(gg.get("avg_epoch_sec"), errors="coerce")
        gg = gg.sort_values("test_mrr", ascending=False)
        gg.to_csv(out_dir / "test_performance_by_model.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        ax.bar(gg["model"].astype(str), gg["test_mrr"], color="#2563eb", alpha=0.86)
        ax.set_ylim(0, min(1.0, max(0.1, float(gg["test_mrr"].max()) * 1.12)))
        ax.set_title(f"{task}: test MRR by model")
        ax.set_xlabel("model")
        ax.set_ylabel("test MRR")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "test_mrr_by_model.png", dpi=180)
        plt.close(fig)

        time_g = gg[pd.to_numeric(gg["avg_epoch_sec"], errors="coerce").notna()].copy()
        if not time_g.empty:
            fig, ax = plt.subplots(figsize=(8.2, 4.6))
            ax.bar(time_g["model"].astype(str), time_g["avg_epoch_sec"], color="#059669", alpha=0.86)
            ax.set_title(f"{task}: average epoch time")
            ax.set_xlabel("model")
            ax.set_ylabel("seconds / epoch")
            ax.tick_params(axis="x", labelrotation=25)
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_dir / "avg_epoch_sec_by_model.png", dpi=180)
            plt.close(fig)


def export_importance_plots(importance: pd.DataFrame, out_root: Path) -> None:
    if importance.empty:
        return
    for (dataset, task), g in importance.groupby(["dataset", "task_name"]):
        out_dir = out_root / dataset / task / "edge_semantics"
        out_dir.mkdir(parents=True, exist_ok=True)
        g = g.copy()
        g.to_csv(out_dir / "edge_semantic_importance.csv", index=False)
        if "delta_mrr_vs_full" not in g or g["delta_mrr_vs_full"].isna().all():
            continue
        g = g.sort_values("delta_mrr_vs_full")
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        colors = ["#dc2626" if v < 0 else "#2563eb" for v in g["delta_mrr_vs_full"]]
        ax.bar(g["variant"], g["delta_mrr_vs_full"], color=colors, alpha=0.86)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title(f"{task}: ablation delta vs full")
        ax.set_xlabel("variant")
        ax.set_ylabel("MRR delta")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "ablation_delta_mrr_vs_full.png", dpi=180)
        plt.close(fig)


def write_case_study_notes(out_root: Path, task_rows: pd.DataFrame, importance: pd.DataFrame, perf: pd.DataFrame) -> None:
    lines = [
        "# Case Study Candidates",
        "",
        "These notes are generated from task metadata, relation counts, memory traces, ablations, and performance tables.",
        "",
    ]
    for dataset, g in task_rows.groupby("dataset"):
        lines.append(f"## {relabel_dataset(dataset)}")
        target = g[g["is_target_relation"]]
        target_phrase = "; ".join(target["task_phrase"].drop_duplicates().astype(str).tolist()[:4]) or "target relation from task metadata"
        lines.append(f"- Link prediction target: {target_phrase}.")
        imp = importance[(importance["dataset"] == dataset) & (~importance["delta_mrr_vs_full"].isna())].copy() if not importance.empty else pd.DataFrame()
        if not imp.empty:
            neg = imp[imp["delta_mrr_vs_full"] < 0].sort_values("delta_mrr_vs_full").head(3)
            if neg.empty:
                best = imp.sort_values("delta_mrr_vs_full", ascending=False).head(3)
                msg = ", ".join(f"{r.variant}: {r.delta_mrr_vs_full:+.4f} MRR" for r in best.itertuples())
                lines.append(f"- Edge/relation semantics evidence: ablations did not reduce MRR in the available runs; largest gains vs full are {msg}.")
            else:
                drops = ", ".join(f"{r.variant}: {r.delta_mrr_vs_full:+.4f} MRR" for r in neg.itertuples())
                lines.append(f"- Edge/relation semantics evidence: strongest negative effects vs full are {drops}.")
        p = perf[(perf["dataset"] == dataset) & (perf.get("split", "test") == "test")] if not perf.empty and "split" in perf.columns else pd.DataFrame()
        if not p.empty:
            top = p.sort_values("test_mrr", ascending=False).head(3)
            winners = ", ".join(f"{r.model}={r.test_mrr:.4f}" for r in top.itertuples() if pd.notna(r.test_mrr))
            lines.append(f"- Performance contrast for evidence paragraph: {winners}.")
        if dataset == "patent_us":
            lines.append("- Suggested narrative: company/org memory is shaped by patent-org and org-coorg context before predicting org-codeIII links, so assignee-side collaboration or publication-like activity can be used as evidence for IPC-code affinity shifts.")
        elif dataset == "aminer_paper":
            lines.append("- Suggested narrative: citation links are predicted while author-write-paper, venue, and coauthor context act as semantic channels that move paper and relation memories over time.")
        elif dataset == "dblp":
            lines.append("- Suggested narrative: paper-paper citation prediction can be framed as temporal topic diffusion under heterogeneous paper/venue context.")
        elif dataset == "thgl_software":
            lines.append("- Suggested narrative: many software relation types provide a clean edge-semantics stress test; relation-memory trajectories reveal which edge slots drift toward the target link semantics.")
        lines.append("")
    (out_root / "case_study_candidates.md").write_text("\n".join(lines), encoding="utf-8")


def write_commands(out_root: Path, project_root: Path, datasets: list[str]) -> None:
    py = "/home/xuxj/.conda/envs/pytorch/bin/python"
    lines = [
        "# Reproduction Commands",
        "",
        "Run from `/home/xuxj/2026/dyn_relation_memory`.",
        "",
        "All datasets:",
        "",
        f"`{py} scripts/analysis/analyze_main_edge_visuals.py --project-root {project_root} --datasets {','.join(datasets)}`",
        "",
        "Single dataset examples:",
        "",
    ]
    for ds in datasets:
        lines.append(f"- `{py} scripts/analysis/analyze_main_edge_visuals.py --project-root {project_root} --datasets {ds} --out-root results/main_edge_visual_analysis_single_{ds}`")
    (out_root / "commands.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(out_root: Path, datasets: list[str]) -> None:
    lines = [
        "# Main Edge Visual Analysis Index",
        "",
        "Generated artifacts are organized as `dataset/task/{node_memory,rel_memory,relation_type,community_memory,task_definition,performance,edge_semantics}`.",
        "",
        "Top-level tables:",
        "- `task_definitions.csv`: concrete link prediction target and context relations.",
        "- `relation_split_counts.csv`: relation counts by train/val/test split.",
        "- `dynamic_memory_availability.csv`: confirms node/relation_type/community dynamic traces.",
        "- `performance_summary.csv`: DHC-main and baseline metrics/time/parameter summary.",
        "- `baseline_epoch_curves.csv`: parsed baseline loss/val/time curves.",
        "- `edge_semantic_importance.csv`: ablation deltas for edge/relation semantics.",
        "",
        "Per-task visual files:",
        "- `*_pca2d.csv` and `*_pca2d_all_epochs.png`: two-dimensional memory trajectories.",
        "- `*_pca2d_first_last.png`: first-vs-last epoch semantic movement.",
        "- `*_centroid_trajectory.png`: memory centroid evolution.",
        "- `*_norm_by_epoch.csv/png`: memory magnitude evolution over epochs.",
        "- `relation_composition.png` and `supervised_edges_by_split.png`: task semantics.",
        "- `main_loss_metric_curve.png`, `baseline_loss_val_mrr_curves.png`, `main_epoch_time_curve.png`, `baseline_epoch_time_curves.png`, `test_mrr_by_model.png`, `avg_epoch_sec_by_model.png`: performance and time evidence.",
        "- `figures/overview_test_mrr_by_model.png` and `figures/overview_avg_epoch_sec_by_model.png`: cross-task overview plots.",
        "",
        "Datasets:",
    ]
    for dataset in datasets:
        lines.append(f"- `{dataset}`")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset names")
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()

    root = args.project_root.resolve()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    out_root = (args.out_root or (root / "results" / "main_edge_visual_analysis")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    specs = find_tasks(root, datasets)
    task_rows, split_counts, rel_dynamic = [], [], []
    for spec in specs:
        task = safe_load_pt(spec.task_path)
        task_rows.extend(infer_task_rows(spec, task))
        split_counts.extend(split_rows(spec, task))
        run = choose_run_dir(spec)
        task_out = out_root / spec.dataset / spec.task_name
        if run is not None:
            for kind in ["node_memory", "rel_memory", "community_memory"]:
                ok = export_memory_projection(spec, task, run, task_out / kind, kind)
                rel_dynamic.append(
                    {
                        "dataset": spec.dataset,
                        "task_name": spec.task_name,
                        "run_dir": str(run),
                        "memory_kind": kind,
                        "has_dynamic_trace": ok,
                    }
                )
            mirror_relation_type_artifacts(task_out)

    task_df = pd.DataFrame(task_rows)
    split_df = pd.DataFrame(split_counts)
    task_df.to_csv(out_root / "task_definitions.csv", index=False)
    split_df.to_csv(out_root / "relation_split_counts.csv", index=False)
    pd.DataFrame(rel_dynamic).to_csv(out_root / "dynamic_memory_availability.csv", index=False)
    export_task_relation_plots(task_df, split_df, out_root)

    main_perf = pd.DataFrame(flatten_main_metrics(specs))
    base_perf = parse_baselines(root, set(datasets))
    base_epochs = parse_baseline_epoch_logs(root, set(datasets))
    perf = pd.concat([main_perf, base_perf], ignore_index=True, sort=False)
    export_performance_plots(perf, out_root)
    importance = edge_semantic_importance(root, set(datasets))
    importance.to_csv(out_root / "edge_semantic_importance.csv", index=False)
    export_importance_plots(importance, out_root)
    export_training_curves(specs, out_root)
    export_baseline_epoch_curves(base_epochs, out_root)
    write_case_study_notes(out_root, task_df, importance, perf)
    write_commands(out_root, root, datasets)
    write_index(out_root, datasets)

    print(f"saved analysis artifacts to {out_root}")


if __name__ == "__main__":
    main()
