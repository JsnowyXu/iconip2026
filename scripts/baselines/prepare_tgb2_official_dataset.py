#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


def build_split_override(obj, kept_old_indices: np.ndarray):
    split = obj.get("split")
    if not isinstance(split, dict):
        return None

    kept_old_indices = np.asarray(kept_old_indices, dtype=np.int64).reshape(-1)
    if kept_old_indices.size == 0:
        return None

    inv = np.full(int(kept_old_indices.max()) + 1, -1, dtype=np.int64)
    inv[kept_old_indices] = np.arange(kept_old_indices.size, dtype=np.int64)

    out = {}
    num_edges = int(kept_old_indices.size)
    for key in ("train", "val", "test"):
        idx = split.get(key)
        if idx is None:
            continue
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < inv.shape[0])]
        mapped = inv[idx]
        mapped = mapped[mapped >= 0]
        mask = np.zeros(num_edges, dtype=bool)
        if mapped.size > 0:
            mask[np.unique(mapped)] = True
        out[key] = mask

    if {"train", "val", "test"}.issubset(out.keys()):
        return out
    return None


def export_raw_files(obj, out_dir: Path, dataset_name: str, max_edges: int, use_rank_ts: bool):
    src = obj["edges"]["src"].numpy().astype(np.int64)
    dst = obj["edges"]["dst"].numpy().astype(np.int64)
    ts = obj["edges"]["ts"].numpy().astype(np.float64)
    rel = obj["edges"]["rel"].numpy().astype(np.int64)
    node_type = obj["node_type"].numpy().astype(np.int64)

    order = np.argsort(ts)
    if max_edges > 0:
        order = order[: min(max_edges, len(order))]
    order = np.sort(order)

    src = src[order]
    dst = dst[order]
    ts = ts[order]
    rel = rel[order]
    rel_vals = sorted(np.unique(rel).tolist()) if rel.size > 0 else [0]
    rel_map = {int(r): i for i, r in enumerate(rel_vals)}
    rel = np.array([rel_map[int(r)] for r in rel], dtype=np.int64)

    active_nodes = np.unique(np.concatenate([src, dst], axis=0))

    out_dir.mkdir(parents=True, exist_ok=True)
    edge_csv = out_dir / f"{dataset_name}_edgelist.csv"
    node_csv = out_dir / f"{dataset_name}_nodetype.csv"

    if use_rank_ts:
        # Keep temporal order but avoid quantile ties that can collapse val/test masks.
        ts_out = np.arange(1, len(ts) + 1, dtype=np.int64)
    else:
        ts_out = ts.astype(np.int64)

    with edge_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "head", "tail", "relation_type"])
        for s, d, t, r in zip(src, dst, ts_out, rel):
            w.writerow([int(t), int(s), int(d), int(r)])

    with node_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "type"])
        for nid in active_nodes:
            if 0 <= nid < len(node_type):
                ntype = int(node_type[nid])
            else:
                ntype = 0
            w.writerow([int(nid), ntype])

    split_override = build_split_override(obj, order)
    return edge_csv, node_csv, split_override


def cleanup_cache(dataset_dir: Path, dataset_name: str):
    # Remove cached processed artifacts and negative samples so the updated files are used.
    remove_names = [
        f"ml_{dataset_name}.pkl",
        f"ml_{dataset_name}_edge.pkl",
        f"ml_{dataset_name}_nodeType.pkl",
        f"{dataset_name}_val_ns.pkl",
        f"{dataset_name}_test_ns.pkl",
        f"{dataset_name}_task_split.npy",
    ]
    for name in remove_names:
        p = dataset_dir / name
        if p.exists():
            p.unlink()


def generate_ns_with_tgb2(tgb2_root: Path, dataset_name: str):
    sys.path.insert(0, str(tgb2_root))

    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    from tgb.linkproppred.thg_negative_generator import THGNegativeEdgeGenerator
    from tgb.utils.info import DATA_EVAL_METRIC_DICT, DATA_NS_STRATEGY_DICT, DATA_VERSION_DICT

    if dataset_name not in DATA_EVAL_METRIC_DICT:
        DATA_EVAL_METRIC_DICT[dataset_name] = "mrr"
    if dataset_name not in DATA_NS_STRATEGY_DICT:
        DATA_NS_STRATEGY_DICT[dataset_name] = "node-type-filtered"
    if dataset_name not in DATA_VERSION_DICT:
        DATA_VERSION_DICT[dataset_name] = 1

    ds = PyGLinkPropPredDataset(name=dataset_name, root="datasets", download=False)
    data = ds.get_TemporalData()
    val_data = data[ds.val_mask]
    test_data = data[ds.test_mask]

    first_node_id = int(data.src.min().item())
    first_node_id = min(first_node_id, int(data.dst.min().item()))
    last_node_id = int(data.src.max().item())
    last_node_id = max(last_node_id, int(data.dst.max().item()))

    gen = THGNegativeEdgeGenerator(
        dataset_name=dataset_name,
        first_node_id=first_node_id,
        last_node_id=last_node_id,
        node_type=ds.node_type,
        strategy="node-type-filtered",
        num_neg_e=20,
        edge_data=data,
    )
    partial_path = str(ds.dataset.root)
    gen.generate_negative_samples(pos_edges=val_data, split_mode="val", partial_path=partial_path)
    gen.generate_negative_samples(pos_edges=test_data, split_mode="test", partial_path=partial_path)

    return {
        "num_nodes": int(ds.num_nodes),
        "num_edges": int(ds.num_edges),
        "num_rels": int(ds.num_rels),
        "val_edges": int(ds.val_mask.sum().item()),
        "test_edges": int(ds.test_mask.sum().item()),
        "root": str(ds.dataset.root),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--tgb2-root", required=True, type=str)
    ap.add_argument("--dataset-name", default="thgl-github", type=str)
    ap.add_argument("--max-edges", default=10000, type=int)
    ap.add_argument("--use-rank-ts", action="store_true", default=True)
    args = ap.parse_args()

    tgb2_root = Path(args.tgb2_root).resolve()
    if not tgb2_root.exists():
        raise FileNotFoundError(f"tgb2 root not found: {tgb2_root}")
    dataset_dir = tgb2_root / "tgb" / "datasets" / args.dataset_name.replace("-", "_")

    obj = torch.load(args.input, map_location="cpu")
    cleanup_cache(dataset_dir=dataset_dir, dataset_name=args.dataset_name)
    edge_csv, node_csv, split_override = export_raw_files(
        obj=obj,
        out_dir=dataset_dir,
        dataset_name=args.dataset_name,
        max_edges=args.max_edges,
        use_rank_ts=args.use_rank_ts,
    )
    if split_override is not None:
        np.save(dataset_dir / f"{args.dataset_name}_task_split.npy", split_override, allow_pickle=True)
    stats = generate_ns_with_tgb2(tgb2_root=tgb2_root, dataset_name=args.dataset_name)

    print(f"Prepared official TGB2 dataset at: {dataset_dir}")
    print(f"edgelist: {edge_csv}")
    print(f"nodetype: {node_csv}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
