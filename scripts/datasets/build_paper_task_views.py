#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple

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


def inductive_partition_target(target_old: np.ndarray, src: np.ndarray, ts: np.ndarray, rng: np.random.RandomState):
    target_old = sort_old_indices(target_old, ts)
    if target_old.size == 0:
        z = np.array([], dtype=np.int64)
        return z, z, z, np.array([], dtype=np.int64)

    target_src = src[target_old]
    src_nodes, src_counts = np.unique(target_src, return_counts=True)
    if src_nodes.size == 0:
        z = np.array([], dtype=np.int64)
        return z, z, z, np.array([], dtype=np.int64)

    target_holdout_edges = max(1, int(0.10 * target_old.size))
    min_base_edges = max(64, int(0.10 * target_old.size))
    perm = rng.permutation(src_nodes.size)
    unseen_nodes = []
    heldout_edges = 0
    for idx in perm.tolist():
        if heldout_edges >= target_holdout_edges:
            break
        node = int(src_nodes[idx])
        count = int(src_counts[idx])
        remain_after = int(target_old.size - (heldout_edges + count))
        if remain_after < min_base_edges:
            continue
        unseen_nodes.append(node)
        heldout_edges += count

    if len(unseen_nodes) == 0:
        smallest = int(src_nodes[np.argmin(src_counts)])
        if int(target_old.size - src_counts.min()) >= min_base_edges:
            unseen_nodes = [smallest]

    unseen = np.sort(np.asarray(unseen_nodes, dtype=np.int64))
    test_mask = np.isin(target_src, unseen)
    test_old = target_old[test_mask]
    base_old = target_old[~test_mask]
    if test_old.size == 0 or base_old.size < min_base_edges:
        # Fall back to a pure temporal split when the source-held-out protocol
        # would leave too little supervision for a stable train/val partition.
        base_old = target_old
        test_old = np.array([], dtype=np.int64)
        unseen = np.array([], dtype=np.int64)
    base_old = sort_old_indices(base_old, ts)
    cut = min(max(1, int(0.80 * base_old.size)), max(1, base_old.size - 1))
    train_old = base_old[:cut]
    val_old = base_old[cut:]
    if val_old.size < max(1, min(64, base_old.size // 20)):
        train_old, val_old, test_old = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)
        unseen = np.array([], dtype=np.int64)
    return train_old, val_old, test_old.astype(np.int64), unseen


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


def remove_indices(base: np.ndarray, remove: np.ndarray, ts: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.int64).reshape(-1)
    remove = np.asarray(remove, dtype=np.int64).reshape(-1)
    if base.size == 0 or remove.size == 0:
        return sort_old_indices(base, ts)
    keep = ~np.isin(base, remove)
    return sort_old_indices(base[keep], ts)


def build_edge_view(
    obj: Dict,
    out_name: str,
    target_old: np.ndarray,
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

    explicit_val_old = sort_old_indices(val_target_old, ts_all)
    explicit_test_old = sort_old_indices(test_target_old, ts_all)
    if explicit_val_old.size > 0:
        train_old = remove_indices(train_old, explicit_val_old, ts_all)
        test_old = remove_indices(test_old, explicit_val_old, ts_all)
        val_old = sort_old_indices(np.concatenate([val_old, explicit_val_old]), ts_all)
    if explicit_test_old.size > 0:
        train_old = remove_indices(train_old, explicit_test_old, ts_all)
        val_old = remove_indices(val_old, explicit_test_old, ts_all)
        test_old = sort_old_indices(np.concatenate([test_old, explicit_test_old]), ts_all)

    # Inductive task windows can become empty after filtering context edges that
    # touch unseen nodes. When that happens, fall back to the explicit target
    # split so early stopping and checkpoint selection remain meaningful.
    if val_old.size == 0 and np.asarray(val_target_old).size > 0:
        val_old = sort_old_indices(val_target_old, ts_all)
        train_old = remove_indices(train_old, val_old, ts_all)
        test_old = remove_indices(test_old, val_old, ts_all)
    if test_old.size == 0 and np.asarray(test_target_old).size > 0:
        test_old = sort_old_indices(test_target_old, ts_all)
        train_old = remove_indices(train_old, test_old, ts_all)
        val_old = remove_indices(val_old, test_old, ts_all)

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
    supervised_mask = np.isin(selected_old, supervised_old)
    edge_label = supervised_mask.astype(np.float32)

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
        "target_original_rel_ids": sorted(np.unique(rel_all[supervised_old]).tolist()) if supervised_old.size > 0 else [],
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
            "src": torch.from_numpy(src),
            "dst": torch.from_numpy(dst),
            "ts": torch.from_numpy(ts),
            "rel": torch.from_numpy(rel),
            "edge_feat": torch.from_numpy(edge_feat),
            "edge_label": torch.from_numpy(edge_label),
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


def build_sorted_full_view(obj: Dict, out_name: str) -> Dict:
    ts_all = obj["edges"]["ts"].numpy().astype(np.float32)
    all_old_sorted = sort_old_indices(np.arange(ts_all.shape[0], dtype=np.int64), ts_all)
    train_old, val_old, test_old = temporal_partition_sorted(all_old_sorted, train_ratio=0.70, val_ratio=0.15)
    return build_edge_view(
        obj=obj,
        out_name=out_name,
        target_old=all_old_sorted,
        train_target_old=train_old,
        val_target_old=val_old,
        test_target_old=test_old,
        context_keep_mask=np.ones(ts_all.shape[0], dtype=bool),
        extra_info={"node_task_view": True, "target_kind": "all_relations"},
    )


def choose_target_relation(rel: np.ndarray, target_rel: int = None) -> int:
    if target_rel is not None and int(target_rel) >= 0:
        return int(target_rel)
    vals, cnt = np.unique(rel, return_counts=True)
    if vals.size == 0:
        return 0
    best_rel = int(vals[np.argmax(cnt)])
    return best_rel


def build(input_pt: Path, out_dir: Path, prefix: str, target_rel: int = None):
    obj = torch.load(input_pt, map_location="cpu")
    rel = obj["edges"]["rel"].numpy().astype(np.int64)
    ts = obj["edges"]["ts"].numpy().astype(np.float32)
    src = obj["edges"]["src"].numpy().astype(np.int64)
    dst = obj["edges"]["dst"].numpy().astype(np.int64)

    target_rel_id = choose_target_relation(rel, target_rel=target_rel)
    target_old = sort_old_indices(np.where(rel == int(target_rel_id))[0].astype(np.int64), ts)
    if target_old.size == 0:
        raise RuntimeError(f"no edges found for target relation id={target_rel_id}")

    trans_tr_old, trans_va_old, trans_te_old = temporal_partition_sorted(target_old, train_ratio=0.70, val_ratio=0.15)
    trans_view = build_edge_view(
        obj=obj,
        out_name=f"{prefix}_edge_transductive",
        target_old=target_old,
        train_target_old=trans_tr_old,
        val_target_old=trans_va_old,
        test_target_old=trans_te_old,
        extra_info={
            "target_original_rel_id": int(target_rel_id),
            "protocol": "full_hetero_context_transductive",
        },
    )

    ind_tr_old, ind_va_old, ind_te_old, unseen_nodes = inductive_partition_target(
        target_old=target_old,
        src=src,
        ts=ts,
        rng=np.random.RandomState(42),
    )
    keep_mask = ~(np.isin(src, unseen_nodes) | np.isin(dst, unseen_nodes))
    ind_view = build_edge_view(
        obj=obj,
        out_name=f"{prefix}_edge_inductive",
        target_old=target_old,
        train_target_old=ind_tr_old,
        val_target_old=ind_va_old,
        test_target_old=ind_te_old,
        context_keep_mask=keep_mask,
        extra_info={
            "target_original_rel_id": int(target_rel_id),
            "protocol": "full_hetero_context_inductive",
            "unseen_source_node_ids": unseen_nodes.tolist(),
        },
    )

    node_view = build_sorted_full_view(obj, out_name=f"{prefix}_node_multilabel")

    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"{prefix}_edge_transductive.pt"
    p2 = out_dir / f"{prefix}_edge_inductive.pt"
    p3 = out_dir / f"{prefix}_node_multilabel.pt"
    torch.save(trans_view, p1)
    torch.save(ind_view, p2)
    torch.save(node_view, p3)
    return [p1, p2, p3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--name-prefix", required=True, type=str)
    ap.add_argument("--target-rel", type=int, default=-1)
    args = ap.parse_args()

    outs = build(
        input_pt=Path(args.input),
        out_dir=Path(args.out_dir),
        prefix=args.name_prefix,
        target_rel=None if args.target_rel < 0 else int(args.target_rel),
    )
    print("generated:")
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()
