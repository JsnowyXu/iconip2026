#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def sort_old_indices(indices: np.ndarray, ts: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
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
    tr = sorted_old[:n_train]
    va = sorted_old[n_train : n_train + n_val]
    te = sorted_old[n_train + n_val :]
    if te.size == 0 and va.size > 0:
        te = va[-1:]
        va = va[:-1]
    if va.size == 0 and tr.size > 1:
        va = tr[-1:]
        tr = tr[:-1]
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


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


def pick_relation_id(meta_extra: Dict, rel_name: str) -> int:
    rel_map = meta_extra.get("relation_map", {}) if isinstance(meta_extra, dict) else {}
    if not isinstance(rel_map, dict):
        return -1
    if rel_name in rel_map:
        return int(rel_map[rel_name])
    for k, v in rel_map.items():
        if str(k).lower() == rel_name.lower():
            return int(v)
    return -1


def build_edge_view(
    obj: Dict,
    out_name: str,
    train_target_old: np.ndarray,
    val_target_old: np.ndarray,
    test_target_old: np.ndarray,
    context_keep_mask: np.ndarray = None,
    extra_info: Dict = None,
) -> Dict:
    src_all = obj["edges"]["src"].numpy().astype(np.int64)
    dst_all = obj["edges"]["dst"].numpy().astype(np.int64)
    ts_all = obj["edges"]["ts"].numpy().astype(np.float32)
    rel_all = obj["edges"]["rel"].numpy().astype(np.int64)
    raw_num_rels = int(obj.get("num_rels", int(rel_all.max()) + 1 if rel_all.size > 0 else 1))

    all_old_sorted = sort_old_indices(np.arange(ts_all.shape[0], dtype=np.int64), ts_all)
    keep_mask = np.ones(ts_all.shape[0], dtype=bool) if context_keep_mask is None else np.asarray(context_keep_mask, dtype=bool)
    keep_mask = keep_mask.reshape(-1)
    if keep_mask.shape[0] != ts_all.shape[0]:
        raise ValueError("context_keep_mask length mismatch")

    train_cut = float(ts_all[train_target_old[-1]]) if train_target_old.size > 0 else float(ts_all[all_old_sorted[0]])
    val_cut = float(ts_all[val_target_old[-1]]) if val_target_old.size > 0 else train_cut

    train_old = all_old_sorted[keep_mask[all_old_sorted] & (ts_all[all_old_sorted] <= train_cut)]
    val_old = all_old_sorted[keep_mask[all_old_sorted] & (ts_all[all_old_sorted] > train_cut) & (ts_all[all_old_sorted] <= val_cut)]
    assigned = np.zeros(ts_all.shape[0], dtype=bool)
    assigned[train_old] = True
    assigned[val_old] = True
    test_old = all_old_sorted[~assigned[all_old_sorted]]
    selected_old = np.concatenate([train_old, val_old, test_old], axis=0).astype(np.int64)
    selected_old = sort_old_indices(selected_old, ts_all)

    old_to_new = np.full((ts_all.shape[0],), -1, dtype=np.int64)
    old_to_new[selected_old] = np.arange(selected_old.shape[0], dtype=np.int64)

    src = src_all[selected_old]
    dst = dst_all[selected_old]
    ts = ts_all[selected_old]
    rel = rel_all[selected_old]
    rel_vals = sorted(np.unique(rel).tolist()) if rel.size > 0 else [0]
    num_rels = max(1, raw_num_rels)

    supervised_old = sort_old_indices(np.concatenate([train_target_old, val_target_old, test_target_old], axis=0), ts_all)
    edge_label = np.isin(selected_old, supervised_old).astype(np.float32)

    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6) if ts.size > 0 else np.array([], dtype=np.float32)
    rel_norm = rel / max(1, num_rels - 1) if rel.size > 0 else np.array([], dtype=np.float32)
    edge_feat = np.stack([ts_norm, rel_norm], axis=1).astype(np.float32) if ts.size > 0 else np.zeros((0, 2), dtype=np.float32)

    train_new = map_old_to_new(train_old, old_to_new)
    val_new = map_old_to_new(val_old, old_to_new)
    test_new = map_old_to_new(test_old, old_to_new)

    meta = dict(obj.get("meta", {}))
    extra = dict(meta.get("extra", {}))
    extra.pop("task_splits", None)
    extra["task_view_name"] = out_name
    extra["source_rel_ids"] = rel_vals
    extra["context_protocol"] = {
        "enabled": True,
        "supervision_mode": "edge_label_mask",
        "preserve_global_rel_ids": True,
        "selected_edges": int(selected_old.size),
        "supervised_edges": int(edge_label.sum()),
        "train_window_edges": int(train_new.size),
        "val_window_edges": int(val_new.size),
        "test_window_edges": int(test_new.size),
        "train_supervised_edges": int(np.isin(train_old, supervised_old).sum()),
        "val_supervised_edges": int(np.isin(val_old, supervised_old).sum()),
        "test_supervised_edges": int(np.isin(test_old, supervised_old).sum()),
    }
    if isinstance(extra_info, dict):
        extra["context_protocol"].update(extra_info)
    meta["dataset_name"] = out_name
    meta["extra"] = extra

    return {
        "num_nodes": int(obj["num_nodes"]),
        "num_rels": int(num_rels),
        "num_node_types": int(obj.get("num_node_types", 1)),
        "node_feat": obj["node_feat"],
        "node_type": obj["node_type"],
        "edges": {
            "src": torch.from_numpy(src.astype(np.int64)),
            "dst": torch.from_numpy(dst.astype(np.int64)),
            "ts": torch.from_numpy(ts.astype(np.float32)),
            "rel": torch.from_numpy(rel.astype(np.int64)),
            "edge_feat": torch.from_numpy(edge_feat.astype(np.float32)),
            "edge_label": torch.from_numpy(edge_label.astype(np.float32)),
        },
        "split": {
            "train": torch.from_numpy(train_new),
            "val": torch.from_numpy(val_new),
            "test": torch.from_numpy(test_new),
        },
        "node_multiclass": obj.get("node_multiclass", None),
        "node_multilabel": obj.get("node_multilabel", None),
        "meta": meta,
    }


def build_sorted_full_view(obj: Dict, out_name: str) -> Dict:
    ts_all = obj["edges"]["ts"].numpy().astype(np.float32)
    all_old_sorted = sort_old_indices(np.arange(ts_all.shape[0], dtype=np.int64), ts_all)
    train_old, val_old, test_old = temporal_partition_sorted(all_old_sorted, train_ratio=0.70, val_ratio=0.15)
    return build_edge_view(
        obj=obj,
        out_name=out_name,
        train_target_old=train_old,
        val_target_old=val_old,
        test_target_old=test_old,
        extra_info={"node_task_view": True, "target_kind": "all_relations"},
    )


def sanitize_old_indices(ids: List[int], num_edges: int, ts: np.ndarray) -> np.ndarray:
    arr = np.asarray(ids, dtype=np.int64).reshape(-1)
    arr = arr[(arr >= 0) & (arr < num_edges)]
    return sort_old_indices(arr, ts)


def build_patent_task_views(obj: Dict, out_dir: Path, name_prefix: str):
    meta_extra = obj.get("meta", {}).get("extra", {})
    rel_patent_ipc = pick_relation_id(meta_extra, "patent_ipc")
    rel_assignee_ipc = pick_relation_id(meta_extra, "assignee_ipc")

    src = obj["edges"]["src"].numpy().astype(np.int64)
    dst = obj["edges"]["dst"].numpy().astype(np.int64)
    ts = obj["edges"]["ts"].numpy().astype(np.float32)
    rel = obj["edges"]["rel"].numpy().astype(np.int64)
    n_edges = rel.shape[0]
    results = []

    if rel_patent_ipc >= 0:
        target_old = sort_old_indices(np.where(rel == rel_patent_ipc)[0].astype(np.int64), ts)
        tr_old, va_old, te_old = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)
        v = build_edge_view(
            obj=obj,
            out_name=f"{name_prefix}_edge_patent_ipc",
            train_target_old=tr_old,
            val_target_old=va_old,
            test_target_old=te_old,
            extra_info={
                "target_original_rel_id": int(rel_patent_ipc),
                "protocol": "full_hetero_context_transductive",
            },
        )
        op = out_dir / f"{name_prefix}_edge_patent_ipc.pt"
        torch.save(v, op)
        results.append(op)

    task_splits = meta_extra.get("task_splits", {}) if isinstance(meta_extra, dict) else {}
    if rel_assignee_ipc >= 0:
        target_old = sort_old_indices(np.where(rel == rel_assignee_ipc)[0].astype(np.int64), ts)

        trans_info = task_splits.get("assignee_ipc_transductive", {}) if isinstance(task_splits, dict) else {}
        if isinstance(trans_info, dict) and len(trans_info) > 0:
            tr_old = sanitize_old_indices(trans_info.get("edge_indices_train", []), n_edges, ts)
            va_old = sanitize_old_indices(trans_info.get("edge_indices_val", []), n_edges, ts)
            te_old = sanitize_old_indices(trans_info.get("edge_indices_test", []), n_edges, ts)
        else:
            tr_old, va_old, te_old = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)
        v = build_edge_view(
            obj=obj,
            out_name=f"{name_prefix}_edge_assignee_ipc_transductive",
            train_target_old=tr_old,
            val_target_old=va_old,
            test_target_old=te_old,
            extra_info={
                "target_original_rel_id": int(rel_assignee_ipc),
                "protocol": "full_hetero_context_transductive",
            },
        )
        op = out_dir / f"{name_prefix}_edge_assignee_ipc_transductive.pt"
        torch.save(v, op)
        results.append(op)

        ind_info = task_splits.get("assignee_ipc_inductive", {}) if isinstance(task_splits, dict) else {}
        if isinstance(ind_info, dict) and len(ind_info) > 0:
            tr_old = sanitize_old_indices(ind_info.get("edge_indices_train", []), n_edges, ts)
            va_old = sanitize_old_indices(ind_info.get("edge_indices_val", []), n_edges, ts)
            te_old = sanitize_old_indices(ind_info.get("edge_indices_test", []), n_edges, ts)
            unseen_nodes = np.asarray(ind_info.get("unseen_assignee_node_ids", []), dtype=np.int64).reshape(-1)
        else:
            tr_old, va_old, te_old = temporal_partition_sorted(target_old, train_ratio=0.80, val_ratio=0.20)
            unseen_nodes = np.array([], dtype=np.int64)
        keep_mask = ~(np.isin(src, unseen_nodes) | np.isin(dst, unseen_nodes)) if unseen_nodes.size > 0 else np.ones(n_edges, dtype=bool)
        v = build_edge_view(
            obj=obj,
            out_name=f"{name_prefix}_edge_assignee_ipc_inductive",
            train_target_old=tr_old,
            val_target_old=va_old,
            test_target_old=te_old,
            context_keep_mask=keep_mask,
            extra_info={
                "target_original_rel_id": int(rel_assignee_ipc),
                "protocol": "full_hetero_context_inductive",
                "unseen_assignee_node_ids": unseen_nodes.tolist(),
            },
        )
        op = out_dir / f"{name_prefix}_edge_assignee_ipc_inductive.pt"
        torch.save(v, op)
        results.append(op)

    node_view = build_sorted_full_view(obj, out_name=f"{name_prefix}_node_patent_multilabel")
    node_meta = dict(node_view.get("meta", {}))
    node_extra = dict(node_meta.get("extra", {}))
    node_extra["node_task"] = "patent_multilabel_ipc_subclass"
    node_meta["dataset_name"] = f"{name_prefix}_node_patent_multilabel"
    node_meta["extra"] = node_extra
    node_view["meta"] = node_meta
    op = out_dir / f"{name_prefix}_node_patent_multilabel.pt"
    torch.save(node_view, op)
    results.append(op)

    return results


def main():
    ap = argparse.ArgumentParser("Build task-specific unified views from processed datasets")
    ap.add_argument("--input", required=True, type=str, help="Path to unified .pt")
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--name-prefix", default="", type=str)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(in_path, map_location="cpu")
    prefix = args.name_prefix.strip() or in_path.stem
    outputs = build_patent_task_views(obj, out_dir=out_dir, name_prefix=prefix)
    print("generated task views:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
