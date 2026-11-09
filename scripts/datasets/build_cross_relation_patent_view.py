#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def temporal_split(ts: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ts.size == 0:
        z = np.array([], dtype=np.int64)
        return z, z, z
    order = np.argsort(ts)
    n = ts.size
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = max(1, n - n_train - 1)
    tr = np.sort(order[:n_train])
    va = np.sort(order[n_train : n_train + n_val])
    te = np.sort(order[n_train + n_val :])
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def build_view(
    obj: Dict,
    target_rel_id: int,
    aux_rel_id: int,
    bridge_rel_id: int,
    out_name: str,
    aux_mode: str = "train_time",
    context_only_aux: bool = True,
) -> Dict:
    src = obj["edges"]["src"].numpy().astype(np.int64)
    dst = obj["edges"]["dst"].numpy().astype(np.int64)
    ts = obj["edges"]["ts"].numpy().astype(np.float32)
    rel = obj["edges"]["rel"].numpy().astype(np.int64)

    target_old = np.where(rel == int(target_rel_id))[0].astype(np.int64)
    aux_old = np.where(rel == int(aux_rel_id))[0].astype(np.int64)
    bridge_old = np.where(rel == int(bridge_rel_id))[0].astype(np.int64)
    if target_old.size == 0:
        raise RuntimeError(f"no edges found for target relation id={target_rel_id}")

    tr_l, va_l, te_l = temporal_split(ts[target_old])
    target_tr_old = target_old[tr_l]
    target_va_old = target_old[va_l]
    target_te_old = target_old[te_l]

    if aux_mode == "all":
        aux_train_old = aux_old
        bridge_train_old = bridge_old
    elif aux_mode == "train_time":
        cutoff = float(ts[target_tr_old].max()) if target_tr_old.size > 0 else float(ts[target_old].min())
        aux_train_old = aux_old[ts[aux_old] <= cutoff]
        bridge_train_old = bridge_old[ts[bridge_old] <= cutoff]
    else:
        raise ValueError(f"unknown aux_mode={aux_mode}")

    selected_old = np.unique(np.concatenate([target_old, aux_train_old, bridge_train_old], axis=0))
    old_to_new = np.full((src.shape[0],), -1, dtype=np.int64)
    old_to_new[selected_old] = np.arange(selected_old.shape[0], dtype=np.int64)

    src_new = src[selected_old]
    dst_new = dst[selected_old]
    ts_new = ts[selected_old]
    rel_old_new = rel[selected_old]

    rel_vals = sorted(np.unique(rel_old_new).tolist())
    rel_map_new = {int(r): i for i, r in enumerate(rel_vals)}
    rel_new = np.array([rel_map_new[int(r)] for r in rel_old_new], dtype=np.int64)
    num_rels = max(1, len(rel_vals))

    edge_label = np.ones((src_new.shape[0],), dtype=np.float32)
    if context_only_aux:
        is_target = rel_old_new == int(target_rel_id)
        edge_label[~is_target] = 0.0

    ts_norm = (ts_new - ts_new.min()) / (ts_new.max() - ts_new.min() + 1e-6) if ts_new.size > 0 else np.array([], dtype=np.float32)
    rel_norm = rel_new / max(1, num_rels - 1) if rel_new.size > 0 else np.array([], dtype=np.float32)
    edge_feat = np.stack([ts_norm, rel_norm], axis=1).astype(np.float32) if ts_new.size > 0 else np.zeros((0, 2), dtype=np.float32)

    train_old = np.unique(np.concatenate([target_tr_old, aux_train_old, bridge_train_old], axis=0))
    val_old = np.unique(target_va_old)
    test_old = np.unique(target_te_old)

    train_new = np.unique(old_to_new[train_old][old_to_new[train_old] >= 0]).astype(np.int64)
    val_new = np.unique(old_to_new[val_old][old_to_new[val_old] >= 0]).astype(np.int64)
    test_new = np.unique(old_to_new[test_old][old_to_new[test_old] >= 0]).astype(np.int64)

    meta = dict(obj.get("meta", {}))
    extra = dict(meta.get("extra", {}))
    extra["task_view_name"] = out_name
    extra["cross_relation_boost"] = {
        "enabled": True,
        "target_original_rel_id": int(target_rel_id),
        "aux_original_rel_id": int(aux_rel_id),
        "bridge_original_rel_id": int(bridge_rel_id),
        "aux_mode": aux_mode,
        "target_edges": int(target_old.size),
        "aux_edges_total": int(aux_old.size),
        "bridge_edges_total": int(bridge_old.size),
        "aux_edges_in_train": int(aux_train_old.size),
        "bridge_edges_in_train": int(bridge_train_old.size),
        "selected_edges": int(selected_old.size),
        "new_relation_map": {str(k): int(v) for k, v in rel_map_new.items()},
        "context_only_aux": bool(context_only_aux),
    }
    extra["source_rel_ids"] = rel_vals
    meta["dataset_name"] = out_name
    meta["extra"] = extra

    out = {
        "num_nodes": int(obj["num_nodes"]),
        "num_rels": int(num_rels),
        "num_node_types": int(obj.get("num_node_types", 1)),
        "node_feat": obj["node_feat"],
        "node_type": obj["node_type"],
        "edges": {
            "src": torch.from_numpy(src_new),
            "dst": torch.from_numpy(dst_new),
            "ts": torch.from_numpy(ts_new),
            "rel": torch.from_numpy(rel_new),
            "edge_feat": torch.from_numpy(edge_feat),
            "edge_label": torch.from_numpy(edge_label),
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Build cross-relation enhancement view for patent->IPC prediction")
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--name-prefix", required=True, type=str)
    ap.add_argument("--target-rel-name", type=str, default="patent_ipc")
    ap.add_argument("--aux-rel-name", type=str, default="assignee_ipc")
    ap.add_argument("--bridge-rel-name", type=str, default="patent_assignee")
    ap.add_argument("--aux-mode", type=str, default="train_time", choices=["train_time", "all"])
    ap.add_argument(
        "--context-only-aux",
        action="store_true",
        default=False,
        help="If set, aux+bridge edges are used as memory context only (edge_label=0) and excluded from link loss.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(in_path, map_location="cpu")
    extra = obj.get("meta", {}).get("extra", {})
    rel_map = extra.get("relation_map", {})
    if not isinstance(rel_map, dict):
        raise RuntimeError("relation_map missing in meta.extra")

    if args.target_rel_name not in rel_map:
        raise RuntimeError(f"target relation '{args.target_rel_name}' not found in relation_map={rel_map}")
    if args.aux_rel_name not in rel_map:
        raise RuntimeError(f"aux relation '{args.aux_rel_name}' not found in relation_map={rel_map}")
    if args.bridge_rel_name not in rel_map:
        raise RuntimeError(f"bridge relation '{args.bridge_rel_name}' not found in relation_map={rel_map}")

    target_rel_id = int(rel_map[args.target_rel_name])
    aux_rel_id = int(rel_map[args.aux_rel_name])
    bridge_rel_id = int(rel_map[args.bridge_rel_name])

    out_name = f"{args.name_prefix}_edge_patent_ipc_plus_assignee_ipc"
    out_obj = build_view(
        obj=obj,
        target_rel_id=target_rel_id,
        aux_rel_id=aux_rel_id,
        bridge_rel_id=bridge_rel_id,
        out_name=out_name,
        aux_mode=args.aux_mode,
        context_only_aux=args.context_only_aux,
    )
    out_pt = out_dir / f"{out_name}.pt"
    torch.save(out_obj, out_pt)

    print(f"saved: {out_pt}")
    print("split sizes:", {k: int(v.numel()) for k, v in out_obj["split"].items()})
    print("relations in view:", sorted(torch.unique(out_obj["edges"]["rel"]).tolist()))
    print("cross_relation_boost:", out_obj["meta"]["extra"].get("cross_relation_boost", {}))


if __name__ == "__main__":
    main()
