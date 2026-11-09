#!/usr/bin/env python3
import argparse
import ast
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


DATASET_SPECS = [
    {
        "name": "patent_us",
        "short": "us",
        "csv": "all-us.csv",
        "ipc": "IPCSubClass_US.csv",
        "cls": "clsIII-reInd.csv",
        "org_name_cols": ["orgname"],
        "title_cols": ["title"],
        "summary_cols": ["summary"],
        "date_cols": ["publicationDate", "date_time", "date_parsed"],
    },
    {
        "name": "patent_cn",
        "short": "cn",
        "csv": "all-cn.csv",
        "ipc": "IPCSubClass.csv",
        "cls": "clsIII-reInd.csv",
        "org_name_cols": ["applicantName", "orgname"],
        "title_cols": ["title"],
        "summary_cols": ["summary"],
        "date_cols": ["publicationDate"],
    },
    {
        "name": "patent_in",
        "short": "in",
        "csv": "all-in.csv",
        "ipc": "IPCSubClass_US.csv",
        "cls": "clsIII-reInd.csv",
        "org_name_cols": ["orgname"],
        "title_cols": ["title"],
        "summary_cols": ["summary"],
        "date_cols": ["publicationDate", "date_time", "date_parsed"],
    },
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def load_ipc_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, engine="python")
    except Exception:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return pd.DataFrame({"Class Name": []})
        header = lines[0].strip().lstrip("\ufeff") or "Class Name"
        rows = [{"Class Name": line.strip()} for line in lines[1:] if line.strip()]
        return pd.DataFrame(rows, columns=[header])


def split_multi_value(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(split_multi_value(item))
        return out

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            return split_multi_value(parsed)
        except Exception:
            pass

    text = text.replace("|", ";")
    parts = [p.strip().strip("'").strip('"') for p in text.split(";")]
    return [p for p in parts if p and p.lower() not in {"nan", "none", "null"}]


def normalize_org_name(raw: str) -> str:
    text = str(raw).strip().strip("'").strip('"')
    return " ".join(text.split())


def parse_publication_ts(value) -> Tuple[float, str]:
    if value is None:
        return 0.0, ""
    if isinstance(value, float) and math.isnan(value):
        return 0.0, ""

    raw = str(value).strip()
    if not raw:
        return 0.0, ""

    dt = None
    for candidate in [raw]:
        try:
            dt = pd.to_datetime(candidate, errors="raise")
            break
        except Exception:
            dt = None
    if dt is None and raw.isdigit() and len(raw) == 8:
        try:
            dt = datetime.strptime(raw, "%Y%m%d")
        except Exception:
            dt = None
    if dt is None:
        try:
            dt = pd.to_datetime(raw, errors="coerce")
        except Exception:
            dt = None
    if dt is None or pd.isna(dt):
        return 0.0, raw
    py_dt = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
    return float(py_dt.timestamp()), py_dt.strftime("%Y-%m-%d")


def sort_old_indices(indices: Iterable[int], ts: np.ndarray) -> np.ndarray:
    idx = np.asarray(list(indices), dtype=np.int64).reshape(-1)
    idx = idx[(idx >= 0) & (idx < ts.shape[0])]
    if idx.size == 0:
        return np.array([], dtype=np.int64)
    idx = np.unique(idx)
    order = np.lexsort((idx, ts[idx]))
    return idx[order].astype(np.int64)


def temporal_partition_sorted(sorted_old: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_old = np.asarray(sorted_old, dtype=np.int64).reshape(-1)
    if sorted_old.size == 0:
        z = np.array([], dtype=np.int64)
        return z, z, z
    n = sorted_old.size
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = max(1, n - n_train - 1)
    train_old = sorted_old[:n_train]
    val_old = sorted_old[n_train : n_train + n_val]
    test_old = sorted_old[n_train + n_val :]
    if test_old.size == 0 and val_old.size > 0:
        test_old = val_old[-1:]
        val_old = val_old[:-1]
    if val_old.size == 0 and train_old.size > 1:
        val_old = train_old[-1:]
        train_old = train_old[:-1]
    return train_old.astype(np.int64), val_old.astype(np.int64), test_old.astype(np.int64)


def map_old_to_new(old_indices: np.ndarray, old_to_new: np.ndarray) -> np.ndarray:
    arr = np.asarray(old_indices, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    arr = arr[(arr >= 0) & (arr < old_to_new.shape[0])]
    mapped = old_to_new[arr]
    mapped = mapped[mapped >= 0]
    if mapped.size == 0:
        return np.array([], dtype=np.int64)
    return np.unique(mapped).astype(np.int64)


def remove_indices(base: np.ndarray, to_remove: np.ndarray, ts: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.int64).reshape(-1)
    to_remove = np.asarray(to_remove, dtype=np.int64).reshape(-1)
    if base.size == 0 or to_remove.size == 0:
        return sort_old_indices(base, ts)
    mask = ~np.isin(base, to_remove)
    return sort_old_indices(base[mask], ts)


def export_tgn(obj: Dict, out_dir: Path, name: str) -> None:
    ensure_dir(out_dir)
    src = obj["edges"]["src"].numpy()
    dst = obj["edges"]["dst"].numpy()
    ts = obj["edges"]["ts"].numpy()
    edge_feat = obj["edges"]["edge_feat"].numpy()

    with (out_dir / f"{name}.csv").open("w", encoding="utf-8") as f:
        f.write("u,i,ts,label,f0,f1,f2,f3\n")
        for i in range(len(src)):
            f0, f1, f2, f3 = [float(x) for x in edge_feat[i, :4]]
            f.write(f"{int(src[i])},{int(dst[i])},{float(ts[i])},1,{f0},{f1},{f2},{f3}\n")

    df = pd.DataFrame({"u": src + 1, "i": dst + 1, "ts": ts, "label": 1, "idx": range(1, len(src) + 1)})
    df.to_csv(out_dir / f"ml_{name}.csv", index=False)
    np.save(out_dir / f"ml_{name}.npy", np.vstack([np.zeros((1, edge_feat.shape[1]), dtype=np.float32), edge_feat]))
    node_feat = obj["node_feat"].numpy().astype(np.float32)
    np.save(out_dir / f"ml_{name}_node.npy", np.vstack([np.zeros((1, node_feat.shape[1]), dtype=np.float32), node_feat]))


def export_tgb2(obj: Dict, out_dir: Path, name: str) -> None:
    ensure_dir(out_dir)
    pd.DataFrame(
        {
            "u": obj["edges"]["src"].numpy(),
            "i": obj["edges"]["dst"].numpy(),
            "ts": obj["edges"]["ts"].numpy(),
            "edge_type": obj["edges"]["rel"].numpy(),
            "label": 1,
            "idx": range(obj["edges"]["src"].shape[0]),
        }
    ).to_csv(out_dir / f"{name}_edgelist.csv", index=False)
    pd.DataFrame({"node_id": range(obj["num_nodes"]), "node_type": obj["node_type"].numpy()}).to_csv(
        out_dir / f"{name}_nodetype.csv", index=False
    )


def export_baseline_bundle(obj: Dict, out_root: Path, name: str) -> None:
    export_tgn(obj, out_root / "tgn", name)
    export_tgb2(obj, out_root / "tgb2", name)


def component_sizes(num_nodes: int, src: np.ndarray, dst: np.ndarray, max_edges: int = 100000) -> np.ndarray:
    parent = np.arange(num_nodes)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(min(max_edges, src.shape[0])):
        union(int(src[i]), int(dst[i]))

    roots = [find(i) for i in range(num_nodes)]
    counts = Counter(roots)
    return np.array(sorted(counts.values(), reverse=True))


def analyze_one(obj: Dict, out_dir: Path) -> Dict:
    src = obj["edges"]["src"].numpy()
    dst = obj["edges"]["dst"].numpy()
    rel = obj["edges"]["rel"].numpy()
    ts = obj["edges"]["ts"].numpy()
    node_type = obj["node_type"].numpy()

    in_deg = np.bincount(dst, minlength=obj["num_nodes"])
    out_deg = np.bincount(src, minlength=obj["num_nodes"])
    deg = in_deg + out_deg

    name = obj.get("meta", {}).get("dataset_name", "dataset")
    ddir = ensure_dir(out_dir / name)
    stats = {
        "dataset": name,
        "num_nodes": int(obj["num_nodes"]),
        "num_edges": int(src.shape[0]),
        "num_relations": int(obj["num_rels"]),
        "num_node_types": int(obj.get("num_node_types", int(node_type.max()) + 1)),
        "time_min": float(ts.min()) if ts.size > 0 else 0.0,
        "time_max": float(ts.max()) if ts.size > 0 else 0.0,
        "edge_feat_dim": int(obj["edges"]["edge_feat"].shape[1]),
        "node_feat_dim": int(obj["node_feat"].shape[1]),
        "avg_degree": float(deg.mean()) if deg.size > 0 else 0.0,
        "max_degree": int(deg.max()) if deg.size > 0 else 0,
    }
    (ddir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.figure(figsize=(6, 4))
    plt.hist(np.log1p(deg), bins=50, color="#2a9d8f", alpha=0.85)
    plt.xlabel("log(1 + degree)")
    plt.ylabel("count")
    plt.title(f"Degree Distribution - {name}")
    plt.tight_layout()
    plt.savefig(ddir / "degree_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(ts, bins=80, color="#e76f51", alpha=0.85)
    plt.xlabel("timestamp")
    plt.ylabel("edge count")
    plt.title(f"Temporal Density - {name}")
    plt.tight_layout()
    plt.savefig(ddir / "temporal_density.png", dpi=160)
    plt.close()

    rel_count = np.bincount(rel, minlength=obj["num_rels"])
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(rel_count)), rel_count, color="#264653")
    plt.xlabel("relation id")
    plt.ylabel("edge count")
    plt.title(f"Relation Distribution - {name}")
    plt.tight_layout()
    plt.savefig(ddir / "relation_distribution.png", dpi=160)
    plt.close()

    nt_count = np.bincount(node_type, minlength=int(node_type.max()) + 1)
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(nt_count)), nt_count, color="#f4a261")
    plt.xlabel("node type id")
    plt.ylabel("node count")
    plt.title(f"Node Type Distribution - {name}")
    plt.tight_layout()
    plt.savefig(ddir / "node_type_distribution.png", dpi=160)
    plt.close()

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


def build_task_view(
    obj: Dict,
    out_name: str,
    train_old: np.ndarray,
    val_old: np.ndarray,
    test_old: np.ndarray,
    supervised_old: np.ndarray,
    extra_info: Dict,
) -> Dict:
    src_all = obj["edges"]["src"].numpy().astype(np.int64)
    dst_all = obj["edges"]["dst"].numpy().astype(np.int64)
    ts_all = obj["edges"]["ts"].numpy().astype(np.float32)
    rel_all = obj["edges"]["rel"].numpy().astype(np.int64)
    edge_feat_all = obj["edges"]["edge_feat"].numpy().astype(np.float32)

    # Held-out events take precedence when entity cutoffs overlap in time.
    # Preserve the original artifacts separately; this correction changes rebuilt splits.
    original_sizes = (len(train_old), len(val_old), len(test_old))
    test_old = sort_old_indices(np.unique(test_old), ts_all)
    val_old = sort_old_indices(np.setdiff1d(val_old, test_old), ts_all)
    train_old = sort_old_indices(np.setdiff1d(train_old, np.union1d(val_old, test_old)), ts_all)
    split_repair = {"priority": "test_then_validation_then_train",
                    "removed_train_entries": original_sizes[0] - len(train_old),
                    "removed_val_entries": original_sizes[1] - len(val_old),
                    "removed_test_duplicates": original_sizes[2] - len(test_old)}

    selected_old = np.concatenate([train_old, val_old, test_old], axis=0).astype(np.int64)
    selected_old = sort_old_indices(selected_old, ts_all)

    old_to_new = np.full((ts_all.shape[0],), -1, dtype=np.int64)
    old_to_new[selected_old] = np.arange(selected_old.shape[0], dtype=np.int64)

    train_new = map_old_to_new(train_old, old_to_new)
    val_new = map_old_to_new(val_old, old_to_new)
    test_new = map_old_to_new(test_old, old_to_new)
    edge_label = np.isin(selected_old, supervised_old).astype(np.float32)

    meta = dict(obj.get("meta", {}))
    extra = dict(meta.get("extra", {}))
    extra["task_view_name"] = out_name
    extra["split_overlap_repair"] = split_repair
    extra["context_protocol"] = {
        "enabled": True,
        "supervision_mode": "edge_label_mask",
        "preserve_global_rel_ids": True,
        "selected_edges": int(selected_old.size),
        "supervised_edges": int(edge_label.sum()),
        "train_window_edges": int(train_new.size),
        "val_window_edges": int(val_new.size),
        "test_window_edges": int(test_new.size),
    }
    extra.update(extra_info)
    meta["dataset_name"] = out_name
    meta["extra"] = extra

    return {
        "num_nodes": int(obj["num_nodes"]),
        "num_rels": int(obj["num_rels"]),
        "num_node_types": int(obj["num_node_types"]),
        "node_feat": obj["node_feat"],
        "node_type": obj["node_type"],
        "edges": {
            "src": torch.from_numpy(src_all[selected_old].astype(np.int64)),
            "dst": torch.from_numpy(dst_all[selected_old].astype(np.int64)),
            "ts": torch.from_numpy(ts_all[selected_old].astype(np.float32)),
            "rel": torch.from_numpy(rel_all[selected_old].astype(np.int64)),
            "edge_feat": torch.from_numpy(edge_feat_all[selected_old].astype(np.float32)),
            "edge_label": torch.from_numpy(edge_label.astype(np.float32)),
        },
        "split": {
            "train": torch.from_numpy(train_new),
            "val": torch.from_numpy(val_new),
            "test": torch.from_numpy(test_new),
        },
        "node_multiclass": obj.get("node_multiclass"),
        "node_multilabel": obj.get("node_multilabel"),
        "meta": meta,
    }


def build_task_views(unified: Dict) -> Dict[str, Dict]:
    rel_map = unified["meta"]["extra"]["relation_map"]
    target_rel = int(rel_map["org_codeIII"])
    src = unified["edges"]["src"].numpy().astype(np.int64)
    dst = unified["edges"]["dst"].numpy().astype(np.int64)
    ts = unified["edges"]["ts"].numpy().astype(np.float32)
    rel = unified["edges"]["rel"].numpy().astype(np.int64)
    n_edges = rel.shape[0]

    all_sorted = sort_old_indices(np.arange(n_edges, dtype=np.int64), ts)
    target_old = sort_old_indices(np.where(rel == target_rel)[0], ts)
    tr_tgt, va_tgt, te_tgt = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)

    train_cut = float(ts[tr_tgt[-1]]) if tr_tgt.size > 0 else float(ts[all_sorted[-1]])
    val_cut = float(ts[va_tgt[-1]]) if va_tgt.size > 0 else train_cut

    tr_old = all_sorted[ts[all_sorted] <= train_cut]
    va_old = all_sorted[(ts[all_sorted] > train_cut) & (ts[all_sorted] <= val_cut)]
    te_old = all_sorted[ts[all_sorted] > val_cut]
    trans_view = build_task_view(
        unified,
        out_name=f"{unified['meta']['dataset_name']}_edge_org_codeIII_transductive",
        train_old=tr_old,
        val_old=va_old,
        test_old=te_old,
        supervised_old=target_old,
        extra_info={
            "target_original_rel_id": target_rel,
            "target_relation_name": "org_codeIII",
            "protocol": "full_hetero_context_transductive",
        },
    )

    org_first_ts = {}
    org_nodes = src[target_old]
    for edge_id, org_node in zip(target_old.tolist(), org_nodes.tolist()):
        org_first_ts.setdefault(int(org_node), float(ts[int(edge_id)]))
    org_sorted = sorted(org_first_ts.items(), key=lambda x: (x[1], x[0]))
    org_ids = np.array([k for k, _ in org_sorted], dtype=np.int64)
    if org_ids.size >= 3:
        n_org = org_ids.size
        n_train_org = max(1, int(n_org * 0.70))
        n_val_org = max(1, int(n_org * 0.15))
        if n_train_org + n_val_org >= n_org:
            n_train_org = max(1, n_org - 2)
            n_val_org = max(1, n_org - n_train_org - 1)
        train_orgs = org_ids[:n_train_org]
        val_orgs = org_ids[n_train_org : n_train_org + n_val_org]
        test_orgs = org_ids[n_train_org + n_val_org :]
        if test_orgs.size == 0 and val_orgs.size > 0:
            test_orgs = val_orgs[-1:]
            val_orgs = val_orgs[:-1]
    else:
        train_orgs = org_ids
        val_orgs = np.array([], dtype=np.int64)
        test_orgs = np.array([], dtype=np.int64)

    target_src = src[target_old]
    ind_train_target = target_old[np.isin(target_src, train_orgs)]
    ind_val_target = target_old[np.isin(target_src, val_orgs)]
    ind_test_target = target_old[np.isin(target_src, test_orgs)]
    if ind_val_target.size == 0 or ind_test_target.size == 0:
        ind_train_target, ind_val_target, ind_test_target = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)
        unseen_orgs = np.array([], dtype=np.int64)
    else:
        unseen_orgs = np.concatenate([val_orgs, test_orgs], axis=0).astype(np.int64)

    train_cut_ind = float(ts[ind_train_target[-1]]) if ind_train_target.size > 0 else float(ts[all_sorted[-1]])
    val_cut_ind = float(ts[ind_val_target[-1]]) if ind_val_target.size > 0 else train_cut_ind
    involve_unseen = np.isin(src, unseen_orgs) | np.isin(dst, unseen_orgs) if unseen_orgs.size > 0 else np.zeros(n_edges, dtype=bool)
    ind_tr_old = all_sorted[(ts[all_sorted] <= train_cut_ind) & (~involve_unseen[all_sorted])]
    ind_va_old = all_sorted[(ts[all_sorted] > train_cut_ind) & (ts[all_sorted] <= val_cut_ind)]
    ind_te_old = all_sorted[ts[all_sorted] > val_cut_ind]
    if ind_val_target.size > 0 and ind_va_old.size == 0:
        ind_va_old = sort_old_indices(ind_val_target, ts)
        ind_tr_old = remove_indices(ind_tr_old, ind_va_old, ts)
    if ind_test_target.size > 0 and ind_te_old.size == 0:
        ind_te_old = sort_old_indices(ind_test_target, ts)
        ind_tr_old = remove_indices(ind_tr_old, ind_te_old, ts)
    ind_view = build_task_view(
        unified,
        out_name=f"{unified['meta']['dataset_name']}_edge_org_codeIII_inductive",
        train_old=ind_tr_old,
        val_old=ind_va_old,
        test_old=ind_te_old,
        supervised_old=np.concatenate([ind_train_target, ind_val_target, ind_test_target], axis=0),
        extra_info={
            "target_original_rel_id": target_rel,
            "target_relation_name": "org_codeIII",
            "protocol": "full_hetero_context_inductive",
            "unseen_org_node_ids": unseen_orgs.tolist(),
        },
    )

    node_view = build_task_view(
        unified,
        out_name=f"{unified['meta']['dataset_name']}_node_patent_multilabel",
        train_old=tr_old,
        val_old=va_old,
        test_old=te_old,
        supervised_old=np.array([], dtype=np.int64),
        extra_info={"node_task": "patent_multilabel_codeIII", "node_task_view": True},
    )
    return {
        "edge_org_codeIII_transductive": trans_view,
        "edge_org_codeIII_inductive": ind_view,
        "node_patent_multilabel": node_view,
    }


def build_unified_dataset(spec: Dict, dataset_root: Path, feature_dim: int) -> Tuple[Dict, Dict]:
    ds_dir = dataset_root / spec["name"]
    print(f"[prepare] start {spec['name']}", flush=True)
    df = pd.read_csv(ds_dir / spec["csv"])
    cls_df = pd.read_csv(ds_dir / spec["cls"])
    ipc_df = load_ipc_table(ds_dir / spec["ipc"])

    title_col = choose_first(df.columns, spec["title_cols"])
    summary_col = choose_first(df.columns, spec["summary_cols"])
    date_col = choose_first(df.columns, spec["date_cols"])
    org_name_col = choose_first(df.columns, spec["org_name_cols"])

    ipc_desc_by_code: Dict[str, str] = {}
    if {"Class", "Name"}.issubset(ipc_df.columns):
        for _, row in ipc_df.iterrows():
            code = str(row["Class"]).strip()
            desc = str(row["Name"]).strip()
            ipc_desc_by_code[code] = desc
    else:
        col = ipc_df.columns[0]
        for _, row in ipc_df.iterrows():
            text = str(row[col]).strip()
            code = text.split()[0].strip().strip('"')
            desc = text[len(code):].strip().strip('"')
            ipc_desc_by_code[code] = desc or text

    code_meta_by_reid: Dict[str, Dict] = {}
    for _, row in cls_df.iterrows():
        codeiiiid = str(row["codeIIIId"]).strip()
        codeiii = str(row["codeIII"]).strip()
        reid = str(row["codeIIIIdreId"]).strip()
        desc = ipc_desc_by_code.get(codeiii, "")
        code_meta_by_reid[reid] = {
            "codeIIIId": codeiiiid,
            "codeIIIIdreId": reid,
            "codeIII": codeiii,
            "description": desc,
        }

    node_key_to_id: Dict[str, int] = {}
    node_type: List[int] = []
    node_text: List[str] = []
    node_name: List[str] = []
    patent_nodes: List[int] = []
    patent_raw_ids: List[str] = []
    patent_titles: List[str] = []
    patent_summaries: List[str] = []
    patent_dates: List[str] = []
    patent_ts: List[float] = []
    patent_code_ids: Dict[int, List[str]] = {}
    patent_label_ids: Dict[int, List[str]] = {}
    patent_node_to_idx: Dict[int, int] = {}
    org_nodes: List[int] = []
    org_raw_ids: List[str] = []
    org_names: List[str] = []
    code_nodes: List[int] = []
    code_raw_ids: List[str] = []
    code_reindexed_ids: List[str] = []
    code_names: List[str] = []
    code_descs: List[str] = []

    def get_node(key: str, ntype: int, text: str, display_name: str, registry: str, raw_id: str, extra_text: str = "") -> int:
        if key in node_key_to_id:
            return node_key_to_id[key]
        nid = len(node_type)
        node_key_to_id[key] = nid
        node_type.append(ntype)
        full_text = " ".join([t for t in [text, extra_text] if t]).strip()
        node_text.append(full_text)
        node_name.append(display_name)
        if registry == "patent":
            patent_nodes.append(nid)
            patent_node_to_idx[nid] = len(patent_nodes) - 1
            patent_raw_ids.append(raw_id)
            patent_titles.append(text)
            patent_summaries.append(extra_text)
            patent_dates.append("")
            patent_ts.append(0.0)
            patent_code_ids[nid] = []
            patent_label_ids[nid] = []
        elif registry == "org":
            org_nodes.append(nid)
            org_raw_ids.append(raw_id)
            org_names.append(display_name)
        elif registry == "code":
            code_nodes.append(nid)
            code_raw_ids.append(raw_id)
            code_reindexed_ids.append(key.split(":", 1)[1])
            code_names.append(display_name)
            code_descs.append(extra_text)
        return nid

    rel_map = {"patent_org": 0, "patent_codeIII": 1, "org_codeIII": 2, "org_coorg": 3}
    src_list: List[int] = []
    dst_list: List[int] = []
    ts_list: List[float] = []
    rel_list: List[int] = []

    row_stats = Counter()
    org_pair_counter = Counter()
    org_code_counter = Counter()
    for _, row in df.iterrows():
        patent_raw = str(row["patentId"]).strip()
        patent_title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""
        patent_summary = str(row[summary_col]).strip() if summary_col and pd.notna(row[summary_col]) else ""
        patent_ts_val, patent_date_text = parse_publication_ts(row[date_col] if date_col else "")
        org_ids = split_multi_value(row["orgID"])
        code_ids = split_multi_value(row["codeIIIId"])
        org_names_row = split_multi_value(row[org_name_col]) if org_name_col else []
        class_codes_row = split_multi_value(row["classCode"]) if "classCode" in df.columns else []

        org_ids = [x for x in org_ids if x]
        code_ids = [x for x in code_ids if x in code_meta_by_reid]
        if not org_ids or not code_ids:
            row_stats["rows_skipped_missing_org_or_code"] += 1
            continue

        patent_node = get_node(
            key=f"patent:{patent_raw}",
            ntype=0,
            text=patent_title,
            display_name=f"patent:{patent_raw}",
            registry="patent",
            raw_id=patent_raw,
            extra_text=patent_summary,
        )
        pat_idx = patent_node_to_idx[patent_node]
        patent_dates[pat_idx] = patent_date_text
        patent_ts[pat_idx] = patent_ts_val

        org_node_ids = []
        for i, org_raw in enumerate(dict.fromkeys(org_ids)):
            org_name = normalize_org_name(org_names_row[i]) if i < len(org_names_row) else f"org:{org_raw}"
            org_node = get_node(
                key=f"org:{org_raw}",
                ntype=1,
                text=org_name,
                display_name=org_name,
                registry="org",
                raw_id=str(org_raw),
            )
            org_node_ids.append(org_node)

        code_node_ids = []
        for code_reid in dict.fromkeys(code_ids):
            meta = code_meta_by_reid[code_reid]
            code_name = meta["codeIII"]
            code_desc = meta["description"]
            code_node = get_node(
                key=f"code:{code_reid}",
                ntype=2,
                text=code_name,
                display_name=code_name,
                registry="code",
                raw_id=str(meta["codeIIIId"]),
                extra_text=code_desc,
            )
            code_node_ids.append(code_node)
            patent_code_ids[patent_node].append(code_reid)
            patent_label_ids[patent_node].append(meta["codeIIIId"])

        patent_code_ids[patent_node] = sorted(set(patent_code_ids[patent_node]))
        patent_label_ids[patent_node] = sorted(set(patent_label_ids[patent_node]), key=lambda x: int(x))

        for org_node in org_node_ids:
            src_list.append(patent_node)
            dst_list.append(org_node)
            ts_list.append(patent_ts_val)
            rel_list.append(rel_map["patent_org"])
        row_stats["patent_org_edges"] += len(org_node_ids)

        for code_node in code_node_ids:
            src_list.append(patent_node)
            dst_list.append(code_node)
            ts_list.append(patent_ts_val)
            rel_list.append(rel_map["patent_codeIII"])
        row_stats["patent_codeIII_edges"] += len(code_node_ids)

        for org_node in org_node_ids:
            for code_node in code_node_ids:
                src_list.append(org_node)
                dst_list.append(code_node)
                ts_list.append(patent_ts_val)
                rel_list.append(rel_map["org_codeIII"])
                org_code_counter[(org_node, code_node)] += 1
        row_stats["org_codeIII_edges"] += len(org_node_ids) * len(code_node_ids)

        for i in range(len(org_node_ids)):
            for j in range(i + 1, len(org_node_ids)):
                a, b = org_node_ids[i], org_node_ids[j]
                src_list.extend([a, b])
                dst_list.extend([b, a])
                ts_list.extend([patent_ts_val, patent_ts_val])
                rel_list.extend([rel_map["org_coorg"], rel_map["org_coorg"]])
                org_pair_counter[(a, b)] += 1
        row_stats["org_coorg_edges"] += max(0, len(org_node_ids) * (len(org_node_ids) - 1))
        row_stats["rows_processed"] += 1
        row_stats["class_code_text_nonempty"] += int(len(class_codes_row) > 0)

    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)
    ts = np.asarray(ts_list, dtype=np.float32)
    rel = np.asarray(rel_list, dtype=np.int64)
    node_type_arr = np.asarray(node_type, dtype=np.int64)

    if ts.size == 0:
        raise RuntimeError(f"{spec['name']} produced no edges")

    order = np.lexsort((np.arange(ts.size), ts))
    src = src[order]
    dst = dst[order]
    ts = ts[order]
    rel = rel[order]

    in_deg = np.bincount(dst, minlength=len(node_type))
    out_deg = np.bincount(src, minlength=len(node_type))
    deg = in_deg + out_deg
    first_ts = np.full(len(node_type), ts.max(), dtype=np.float32)
    last_ts = np.zeros(len(node_type), dtype=np.float32)
    for s, d, t in zip(src, dst, ts):
        first_ts[s] = min(first_ts[s], t)
        first_ts[d] = min(first_ts[d], t)
        last_ts[s] = max(last_ts[s], t)
        last_ts[d] = max(last_ts[d], t)
    ts_min, ts_max = float(ts.min()), float(ts.max())
    ts_span = max(ts_max - ts_min, 1.0)
    first_ts_norm = ((first_ts - ts_min) / ts_span).astype(np.float32)
    last_ts_norm = ((last_ts - ts_min) / ts_span).astype(np.float32)

    texts = [t if t else n for t, n in zip(node_text, node_name)]
    text_dim = max(8, feature_dim - 8)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=text_dim, lowercase=False)
    text_feat = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    if text_feat.shape[1] < text_dim:
        pad = np.zeros((text_feat.shape[0], text_dim - text_feat.shape[1]), dtype=np.float32)
        text_feat = np.concatenate([text_feat, pad], axis=1)

    struct_feat = np.stack(
        [
            np.log1p(in_deg) / max(np.log1p(in_deg).max(), 1.0),
            np.log1p(out_deg) / max(np.log1p(out_deg).max(), 1.0),
            np.log1p(deg) / max(np.log1p(deg).max(), 1.0),
            first_ts_norm,
            last_ts_norm,
            (node_type_arr == 0).astype(np.float32),
            (node_type_arr == 1).astype(np.float32),
            (node_type_arr == 2).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    node_feat = np.concatenate([text_feat, struct_feat], axis=1).astype(np.float32)

    patent_node_arr = np.asarray(patent_nodes, dtype=np.int64)
    label_id_order = sorted({cid for labels in patent_label_ids.values() for cid in labels}, key=lambda x: int(x))
    label_meta_by_id = {}
    for meta in code_meta_by_reid.values():
        label_meta_by_id.setdefault(
            meta["codeIIIId"],
            {"codeIII": meta["codeIII"], "description": meta["description"]},
        )
    label_id_to_col = {cid: i for i, cid in enumerate(label_id_order)}
    label_matrix = np.zeros((len(patent_nodes), len(label_id_order)), dtype=np.float32)
    for i, pnode in enumerate(patent_nodes):
        labels = [label_id_to_col[cid] for cid in patent_label_ids.get(pnode, []) if cid in label_id_to_col]
        if labels:
            label_matrix[i, labels] = 1.0

    rel_norm = rel.astype(np.float32) / max(int(rel.max()), 1)
    ts_norm = ((ts - ts_min) / ts_span).astype(np.float32)
    edge_feat = np.stack(
        [
            ts_norm,
            rel_norm,
            node_type_arr[src].astype(np.float32) / 2.0,
            node_type_arr[dst].astype(np.float32) / 2.0,
        ],
        axis=1,
    ).astype(np.float32)

    all_sorted = sort_old_indices(np.arange(ts.shape[0], dtype=np.int64), ts)
    target_old = sort_old_indices(np.where(rel == rel_map["org_codeIII"])[0], ts)
    trans_tr, trans_va, trans_te = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)
    org_first_ts = {}
    for edge_id in target_old.tolist():
        org_first_ts.setdefault(int(src[edge_id]), float(ts[edge_id]))
    org_sorted = sorted(org_first_ts.items(), key=lambda x: (x[1], x[0]))
    org_nodes_sorted = np.array([k for k, _ in org_sorted], dtype=np.int64)
    if org_nodes_sorted.size >= 3:
        n_org = org_nodes_sorted.size
        n_train_org = max(1, int(n_org * 0.70))
        n_val_org = max(1, int(n_org * 0.15))
        if n_train_org + n_val_org >= n_org:
            n_train_org = max(1, n_org - 2)
            n_val_org = max(1, n_org - n_train_org - 1)
        train_orgs = org_nodes_sorted[:n_train_org]
        val_orgs = org_nodes_sorted[n_train_org : n_train_org + n_val_org]
        test_orgs = org_nodes_sorted[n_train_org + n_val_org :]
    else:
        train_orgs = org_nodes_sorted
        val_orgs = np.array([], dtype=np.int64)
        test_orgs = np.array([], dtype=np.int64)

    train_org_set = set(train_orgs.tolist())
    val_org_set = set(val_orgs.tolist())
    test_org_set = set(test_orgs.tolist())
    ind_train = [eid for eid in target_old.tolist() if int(src[eid]) in train_org_set]
    ind_val = [eid for eid in target_old.tolist() if int(src[eid]) in val_org_set]
    ind_test = [eid for eid in target_old.tolist() if int(src[eid]) in test_org_set]

    train_cut = float(ts[trans_tr[-1]]) if trans_tr.size > 0 else float(ts[all_sorted[-1]])
    val_cut = float(ts[trans_va[-1]]) if trans_va.size > 0 else train_cut
    base_tr = all_sorted[ts[all_sorted] <= train_cut]
    base_va = all_sorted[(ts[all_sorted] > train_cut) & (ts[all_sorted] <= val_cut)]
    base_te = all_sorted[ts[all_sorted] > val_cut]

    unified = {
        "num_nodes": int(len(node_type)),
        "num_rels": int(len(rel_map)),
        "num_node_types": 3,
        "node_feat": torch.from_numpy(node_feat.astype(np.float32)),
        "node_type": torch.from_numpy(node_type_arr.astype(np.int64)),
        "edges": {
            "src": torch.from_numpy(src.astype(np.int64)),
            "dst": torch.from_numpy(dst.astype(np.int64)),
            "ts": torch.from_numpy(ts.astype(np.float32)),
            "rel": torch.from_numpy(rel.astype(np.int64)),
            "edge_feat": torch.from_numpy(edge_feat.astype(np.float32)),
            "edge_label": torch.ones(src.shape[0], dtype=torch.float32),
        },
        "split": {
            "train": torch.from_numpy(base_tr),
            "val": torch.from_numpy(base_va),
            "test": torch.from_numpy(base_te),
        },
        "node_multiclass": None,
        "node_multilabel": {
            "node_ids": torch.from_numpy(patent_node_arr.astype(np.int64)),
            "labels": torch.from_numpy(label_matrix.astype(np.float32)),
            "label_names": [
                {
                    "codeIIIId": cid,
                    "codeIII": label_meta_by_id[cid]["codeIII"],
                    "description": label_meta_by_id[cid]["description"],
                }
                for cid in label_id_order
            ],
        },
        "meta": {
            "dataset_name": spec["name"],
            "extra": {
                "relation_map": rel_map,
                "node_type_map": {"patent": 0, "org": 1, "codeIII": 2},
                "label_names": [
                    {
                        "codeIIIId": cid,
                        "codeIII": label_meta_by_id[cid]["codeIII"],
                        "description": label_meta_by_id[cid]["description"],
                    }
                    for cid in label_id_order
                ],
                "patent_attributes": {
                    "node_ids": patent_nodes,
                    "patent_id": patent_raw_ids,
                    "title": patent_titles,
                    "summary": patent_summaries,
                    "publication_date": patent_dates,
                    "publication_ts": patent_ts,
                },
                "org_attributes": {
                    "node_ids": org_nodes,
                    "org_id": org_raw_ids,
                    "name": org_names,
                },
                "codeIII_attributes": {
                    "node_ids": code_nodes,
                    "codeIIIId": code_raw_ids,
                    "codeIIIIdreId": code_reindexed_ids,
                    "codeIII": code_names,
                    "description": code_descs,
                },
                "task_splits": {
                    "org_codeIII_transductive": {
                        "edge_indices_train": trans_tr.tolist(),
                        "edge_indices_val": trans_va.tolist(),
                        "edge_indices_test": trans_te.tolist(),
                    },
                    "org_codeIII_inductive": {
                        "edge_indices_train": ind_train,
                        "edge_indices_val": ind_val,
                        "edge_indices_test": ind_test,
                        "unseen_org_node_ids": np.concatenate([val_orgs, test_orgs], axis=0).astype(np.int64).tolist(),
                    },
                },
                "hetero_main_benchmark_ok": True,
            },
        },
    }

    counts = {
        "rows_total": int(len(df)),
        "rows_processed": int(row_stats["rows_processed"]),
        "rows_skipped_missing_org_or_code": int(row_stats["rows_skipped_missing_org_or_code"]),
        "num_patent_nodes": int(len(patent_nodes)),
        "num_org_nodes": int(len(org_nodes)),
        "num_codeIII_nodes": int(len(code_nodes)),
        "num_edges_patent_org": int((rel == rel_map["patent_org"]).sum()),
        "num_edges_patent_codeIII": int((rel == rel_map["patent_codeIII"]).sum()),
        "num_edges_org_codeIII": int((rel == rel_map["org_codeIII"]).sum()),
        "num_edges_org_coorg": int((rel == rel_map["org_coorg"]).sum()),
        "num_unique_org_code_pairs": int(len(org_code_counter)),
        "num_unique_org_org_pairs": int(len(org_pair_counter)),
        "num_multilabel_classes": int(len(label_id_order)),
        "num_labeled_patents": int(label_matrix.shape[0]),
    }
    print(
        f"[prepare] finished {spec['name']} nodes={unified['num_nodes']} edges={unified['edges']['src'].shape[0]} labels={unified['node_multilabel']['labels'].shape[1]}",
        flush=True,
    )
    return unified, counts


def write_process_md(path: Path, dataset_name: str, python_bin: str, gpu_summary: str, commands: List[str], counts: Dict, outputs: List[Path]) -> None:
    lines = [
        f"# {dataset_name} Process Log",
        "",
        "## Scope",
        "- Project root: `/home/xuxj/2026/dyn_relation_memory`",
        "- Dataset root: `/home/xuxj/2026/datasets`",
        "- Baseline root: `/home/xuxj/2026/baselines/src`",
        f"- Python env: `{python_bin}`",
        f"- GPU snapshot: `{gpu_summary}`",
        "",
        "## Commands",
    ]
    for cmd in commands:
        lines.append(f"- `{cmd}`")
    lines.extend(["", "## Main Counts"])
    for key, value in counts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Outputs"])
    for op in outputs:
        lines.append(f"- `{op}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(project_root: Path, dataset_summaries: List[Dict]) -> None:
    lines = ["# patent Dataset Note", "", "## Main Processing Steps", ""]
    for item in dataset_summaries:
        lines.append(f"### {item['name']}")
        for step in item["steps"]:
            lines.append(f"{step}")
        lines.append("")

    lines.extend(["## Unified Graph Statistics", ""])
    for item in dataset_summaries:
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append("| Item | Value |")
        lines.append("| --- | ---: |")
        for key, value in item["stats_rows"]:
            lines.append(f"| {key} | {value} |")
        lines.append("")
        lines.append("| Node type id | Node type name | Count |")
        lines.append("| --- | --- | ---: |")
        for nid, name, count in item["node_rows"]:
            lines.append(f"| {nid} | {name} | {count} |")
        lines.append("")
        lines.append("| Relation id | Relation name | Count |")
        lines.append("| --- | --- | ---: |")
        for rid, name, count in item["rel_rows"]:
            lines.append(f"| {rid} | {name} | {count} |")
        lines.append("")
        lines.append("| Label id | Label name | Positive count |")
        lines.append("| --- | --- | ---: |")
        for lid, name, count in item["label_rows"][:30]:
            lines.append(f"| {lid} | {name} | {count} |")
        if len(item["label_rows"]) > 30:
            lines.append(f"| ... | ... | {len(item['label_rows']) - 30} more labels omitted |")
        lines.append("")
        lines.append("Task views:")
        for task in item["task_files"]:
            lines.append(f"- `{task}`")
        lines.append("")
        lines.append("Analysis outputs:")
        for fig in item["figures"]:
            lines.append(f"- `{fig}`")
        lines.append("")

    lines.extend(
        [
            "## DHC Representation Usage Check",
            "- Edge prediction does use other edge types as temporal context.",
            "- The patent task views keep the full heterogeneous stream and only restrict supervision with `edge_label` on `org_codeIII`.",
            "- In `dhc/models/dhc_model.py`, link prediction consumes `rel_emb`, relation memory, edge features, and time encoding, not only node states.",
            "- Node multilabel classification already replays the temporal edge stream before calling `predict_node_heads(...)`.",
            "- `predict_node_heads(...)` uses `_node_repr(...)`, which fuses node memory, recent-neighbor context, relation-aware neighbor messages, and community context.",
            "- Current conclusion: edge-type and edge-feature information is already used for both edge prediction and node classification; no immediate DHC patch is required before preprocessing and first-round experiments.",
            "",
        ]
    )
    (project_root / "patent.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--dataset-root", default=str(Path(__file__).resolve().parents[2] / "raw_datasets"))
    ap.add_argument("--python-bin", default=__import__("sys").executable)
    ap.add_argument("--feature-dim", type=int, default=128)
    ap.add_argument("--gpu-summary", default="GPU 0 preferred after nvidia-smi check")
    ap.add_argument("--datasets", default="", help="Comma-separated dataset names, e.g. patent_us,patent_cn")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    processed_root = ensure_dir(project_root / "data" / "processed")
    task_root = ensure_dir(processed_root / "tasks")
    adapter_root = ensure_dir(project_root / "data" / "interim" / "baseline_adapters")
    report_root = ensure_dir(project_root / "data" / "reports")

    commands = [
        "nvidia-smi",
        f"{args.python_bin} scripts/experiments/run_patent_preprocessing.py --project-root {project_root} --dataset-root {dataset_root} --python-bin {args.python_bin}",
    ]

    dataset_summaries = []
    stats_rows = []
    selected = {x.strip() for x in str(args.datasets).split(",") if x.strip()}

    for spec in DATASET_SPECS:
        if selected and spec["name"] not in selected:
            continue
        unified, counts = build_unified_dataset(spec, dataset_root, args.feature_dim)
        unified_path = processed_root / f"{spec['name']}_unified.pt"
        torch.save(unified, unified_path)
        print(f"[save] unified {unified_path}", flush=True)

        task_views = build_task_views(unified)
        task_paths = []
        for suffix, task_obj in task_views.items():
            out_path = task_root / f"{spec['name']}_{suffix}.pt"
            torch.save(task_obj, out_path)
            export_baseline_bundle(task_obj, adapter_root / out_path.stem, out_path.stem)
            task_paths.append(out_path)
            print(f"[save] task {out_path}", flush=True)

        analyze_stats = [analyze_one(unified, report_root)]
        for task_obj in task_views.values():
            analyze_stats.append(analyze_one(task_obj, report_root))
        stats_rows.extend(analyze_stats)
        print(f"[analyze] completed {spec['name']}", flush=True)

        rel = unified["edges"]["rel"].numpy()
        rel_map = unified["meta"]["extra"]["relation_map"]
        rel_inv = {int(v): k for k, v in rel_map.items()}
        node_type_map = unified["meta"]["extra"]["node_type_map"]
        node_type_inv = {int(v): k for k, v in node_type_map.items()}
        node_type_arr = unified["node_type"].numpy()
        labels = unified["node_multilabel"]["labels"].numpy()
        label_names = unified["meta"]["extra"]["label_names"]
        dataset_summaries.append(
            {
                "name": spec["name"],
                "steps": [
                    "1. Read the full raw CSV together with `clsIII-reInd.csv` and the IPC subclass table.",
                    "2. Split multi-valued `orgID` and `codeIIIId` fields, and align organization names when present.",
                    "3. Build a heterogeneous temporal graph with node types `patent`, `org`, and `codeIII`.",
                    "4. Add relation types `patent_org`, `patent_codeIII`, `org_codeIII`, and co-assignee `org_coorg` edges.",
                    "5. Initialize node text features from patent text, organization names, and IPC subclass descriptions.",
                    "6. Export unified `.pt`, task-view `.pt`, TGN-style adapters, and TGB2/STHN adapters.",
                    "7. Draw dataset analysis figures and save per-task statistics for reruns.",
                ],
                "stats_rows": [
                    ("Nodes", unified["num_nodes"]),
                    ("Edges", unified["edges"]["src"].shape[0]),
                    ("Node types", unified["num_node_types"]),
                    ("Relation types", unified["num_rels"]),
                    ("Node feature dim", unified["node_feat"].shape[1]),
                    ("Edge feature dim", unified["edges"]["edge_feat"].shape[1]),
                    ("Labeled patents", unified["node_multilabel"]["node_ids"].shape[0]),
                    ("Multilabel classes", unified["node_multilabel"]["labels"].shape[1]),
                ],
                "node_rows": [(nid, node_type_inv[nid], int((node_type_arr == nid).sum())) for nid in sorted(node_type_inv)],
                "rel_rows": [(rid, rel_inv[rid], int((rel == rid).sum())) for rid in sorted(rel_inv)],
                "label_rows": [
                    (
                        i,
                        f"{label_names[i]['codeIIIId']} / {label_names[i]['codeIII']}",
                        int(labels[:, i].sum()),
                    )
                    for i in range(labels.shape[1])
                ],
                "task_files": [str(p) for p in task_paths],
                "figures": [
                    str(report_root / spec["name"] / "degree_distribution.png"),
                    str(report_root / spec["name"] / "temporal_density.png"),
                    str(report_root / spec["name"] / "relation_distribution.png"),
                    str(report_root / spec["name"] / "node_type_distribution.png"),
                    str(report_root / spec["name"] / "community_sketch.png"),
                ],
            }
        )

        outputs = [unified_path] + task_paths + [
            adapter_root / p.stem / "tgn" / f"ml_{p.stem}.csv" for p in task_paths
        ] + [
            adapter_root / p.stem / "tgb2" / f"{p.stem}_edgelist.csv" for p in task_paths
        ]
        write_process_md(
            path=project_root / f"{spec['name']}_process.md",
            dataset_name=spec["name"],
            python_bin=args.python_bin,
            gpu_summary=args.gpu_summary,
            commands=commands,
            counts=counts,
            outputs=outputs,
        )

    pd.DataFrame(stats_rows).drop_duplicates(subset=["dataset"]).to_csv(report_root / "dataset_stats_summary.csv", index=False)
    write_summary_md(project_root, dataset_summaries)
    print("[done] patent preprocessing pipeline completed", flush=True)


if __name__ == "__main__":
    main()
