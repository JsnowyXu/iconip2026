#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from prepare_dataset import build_default_multilabel, build_degree_features, to_unified


def maybe_factorize(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.integer):
        return series.to_numpy(dtype=np.int64)
    codes, _ = pd.factorize(series.astype(str), sort=True)
    return codes.astype(np.int64)


def load_thg_dataset(edgelist_csv: Path, nodetype_csv: Path, dataset_name: str, max_edges: int) -> Dict:
    edges = pd.read_csv(edgelist_csv)
    required_edge_cols = {"timestamp", "head", "tail", "relation_type"}
    missing = required_edge_cols.difference(edges.columns)
    if missing:
        raise ValueError(f"missing columns in {edgelist_csv}: {sorted(missing)}")

    edges = edges.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if max_edges > 0 and len(edges) > max_edges:
        edges = edges.iloc[:max_edges].copy()

    head_raw = edges["head"]
    tail_raw = edges["tail"]
    rel_raw = edges["relation_type"]
    ts = edges["timestamp"].to_numpy(dtype=np.float32)

    if np.issubdtype(head_raw.dtype, np.integer) and np.issubdtype(tail_raw.dtype, np.integer):
        all_nodes = pd.Index(
            np.unique(np.concatenate([head_raw.to_numpy(dtype=np.int64), tail_raw.to_numpy(dtype=np.int64)]))
        )
        node_to_new = {int(node_id): idx for idx, node_id in enumerate(all_nodes.tolist())}
        src = head_raw.map(node_to_new).to_numpy(dtype=np.int64)
        dst = tail_raw.map(node_to_new).to_numpy(dtype=np.int64)
        original_node_ids = all_nodes.to_numpy(dtype=np.int64)
    else:
        all_nodes = pd.Index(pd.concat([head_raw.astype(str), tail_raw.astype(str)], axis=0).unique())
        node_to_new = {str(node_id): idx for idx, node_id in enumerate(all_nodes.tolist())}
        src = head_raw.astype(str).map(node_to_new).to_numpy(dtype=np.int64)
        dst = tail_raw.astype(str).map(node_to_new).to_numpy(dtype=np.int64)
        original_node_ids = np.arange(len(all_nodes), dtype=np.int64)

    if np.issubdtype(rel_raw.dtype, np.integer):
        rel_vals = np.sort(rel_raw.astype(np.int64).unique())
        rel_map = {int(r): idx for idx, r in enumerate(rel_vals.tolist())}
        rel = rel_raw.map(rel_map).to_numpy(dtype=np.int64)
        rel_original = rel_vals.astype(np.int64)
    else:
        rel_codes, rel_uniques = pd.factorize(rel_raw.astype(str), sort=True)
        rel = rel_codes.astype(np.int64)
        rel_original = rel_uniques.to_numpy()

    node_types = pd.read_csv(nodetype_csv)
    required_node_cols = {"node_id", "type"}
    missing = required_node_cols.difference(node_types.columns)
    if missing:
        raise ValueError(f"missing columns in {nodetype_csv}: {sorted(missing)}")

    type_lookup = {}
    for row in node_types.itertuples(index=False):
        type_lookup[row.node_id] = int(row.type)
        type_lookup[str(row.node_id)] = int(row.type)

    node_type = np.zeros(len(all_nodes), dtype=np.int64)
    missing_type_count = 0
    for new_id, original_id in enumerate(all_nodes.tolist()):
        if original_id in type_lookup:
            node_type[new_id] = int(type_lookup[original_id])
        elif str(original_id) in type_lookup:
            node_type[new_id] = int(type_lookup[str(original_id)])
        else:
            missing_type_count += 1
            node_type[new_id] = 0

    node_feat = build_degree_features(len(all_nodes), src, dst, ts, node_type, dim=128)
    node_multilabel = build_default_multilabel(len(all_nodes), src, dst, ts)
    node_multiclass = node_type.copy()

    return {
        "name": dataset_name,
        "src": src,
        "dst": dst,
        "ts": ts,
        "rel": rel,
        "node_type": node_type,
        "node_feat": node_feat,
        "node_multiclass": node_multiclass,
        "node_multiclass_node_ids": np.arange(len(all_nodes), dtype=np.int64),
        "node_multilabel": node_multilabel,
        "node_multilabel_node_ids": np.arange(len(all_nodes), dtype=np.int64),
        "meta_extra": {
            "source": "TGB2.0",
            "original_edgelist_csv": str(edgelist_csv),
            "original_nodetype_csv": str(nodetype_csv),
            "max_edges_applied": int(max_edges),
            "num_original_rows_loaded": int(len(edges)),
            "num_unique_relations": int(len(np.unique(rel))),
            "num_unique_node_types": int(len(np.unique(node_type))),
            "missing_node_type_count": int(missing_type_count),
            "original_node_ids_sample": original_node_ids[:10].tolist(),
            "original_relation_values_sample": rel_original[:10].tolist() if hasattr(rel_original, "tolist") else list(rel_original[:10]),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", required=True, type=str)
    ap.add_argument("--edgelist-csv", required=True, type=str)
    ap.add_argument("--nodetype-csv", required=True, type=str)
    ap.add_argument("--out-pt", required=True, type=str)
    ap.add_argument("--max-edges", type=int, default=120000)
    args = ap.parse_args()

    dataset_obj = load_thg_dataset(
        edgelist_csv=Path(args.edgelist_csv),
        nodetype_csv=Path(args.nodetype_csv),
        dataset_name=args.dataset_name,
        max_edges=int(args.max_edges),
    )
    out_pt = Path(args.out_pt)
    to_unified(dataset_obj, out_pt)
    print(out_pt)


if __name__ == "__main__":
    main()
