#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ROOT_REPEATMIXER = WORKSPACE_ROOT / "baselines" / "src" / "RepeatMixer-main"
root_str = str(ROOT_REPEATMIXER)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from models.EdgeBank import edge_bank_link_prediction
from utils.DataLoader import Data


def auc_ap(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        order = np.argsort(y_score)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(y_score)) + 1
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    prec = tp / np.maximum(tp + fp, 1)
    ap = (prec * (y == 1)).sum() / max((y == 1).sum(), 1)
    return float(auc), float(ap)


def mrr_from_pos_neg(pos_scores, neg_scores):
    pos = np.asarray(pos_scores, dtype=np.float64).reshape(-1, 1)
    neg = np.asarray(neg_scores, dtype=np.float64)
    ranks = 1 + (neg >= pos).sum(axis=1)
    return float((1.0 / np.maximum(ranks, 1)).mean())


def to_data(obj, idx):
    idx = np.asarray(idx, dtype=np.int64)
    edges = obj["edges"]
    src_all = edges["src"].numpy()
    dst_all = edges["dst"].numpy()
    ts_all = edges["ts"].numpy()
    labels_all = edges["edge_label"].numpy()

    src = src_all[idx]
    dst = dst_all[idx]
    ts = ts_all[idx]
    labels = labels_all[idx]
    edge_ids = idx.astype(np.int64) + 1

    if "raw_edge_ids" in edges:
        raw_edge_ids_all = edges["raw_edge_ids"].numpy().astype(np.int64)
    else:
        # Match RepeatMixer's convention: same (src, dst) pairs share one raw id.
        _, raw_edge_ids_all = np.unique(np.stack([src_all, dst_all], axis=1), axis=0, return_inverse=True)
        raw_edge_ids_all = raw_edge_ids_all.astype(np.int64) + 1
    raw_edge_ids = raw_edge_ids_all[idx]

    return Data(
        src_node_ids=src,
        dst_node_ids=dst,
        node_interact_times=ts,
        edge_ids=edge_ids,
        labels=labels,
        raw_edge_ids=raw_edge_ids,
    )


def evaluate(obj, train_idx, eval_idx, eval_num_negs, mode, rng):
    labels_all = obj["edges"]["edge_label"].numpy().astype(np.float32)
    ts_all = obj["edges"]["ts"].numpy().astype(np.float64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    eval_idx = np.asarray(eval_idx, dtype=np.int64)
    eval_idx = eval_idx[labels_all[eval_idx] > 0.5]

    # EdgeBank memory is built from observed positive history only. Some
    # inductive task splits are node-based rather than strictly chronological,
    # so trim history to interactions available before this eval window.
    if eval_idx.size > 0:
        min_eval_time = float(ts_all[eval_idx].min())
        train_idx = train_idx[(labels_all[train_idx] > 0.5) & (ts_all[train_idx] <= min_eval_time)]
    else:
        train_idx = train_idx[labels_all[train_idx] > 0.5]

    train = to_data(obj, train_idx)
    eval_data = to_data(obj, eval_idx)
    num_nodes = int(obj["num_nodes"])

    if len(eval_data.src_node_ids) == 0:
        return {
            "mrr": 0.0,
            "auc": 0.5,
            "ap": 0.0,
            "eval_pos_edges": 0,
            "history_pos_edges": int(len(train.src_node_ids)),
            "pos_history_hits": 0,
            "pos_history_hit_rate": 0.0,
            "all_zero_tie_rate": 0.0,
        }

    neg_src = np.repeat(eval_data.src_node_ids, eval_num_negs)
    neg_dst = rng.randint(0, num_nodes, len(eval_data.src_node_ids) * eval_num_negs)
    pos_concat, neg_flat = edge_bank_link_prediction(
        history_data=train,
        positive_edges=(eval_data.src_node_ids, eval_data.dst_node_ids),
        negative_edges=(neg_src, neg_dst),
        edge_bank_memory_mode=mode,
        time_window_mode="fixed_proportion",
        time_window_proportion=0.15,
    )
    pos_concat = np.asarray(pos_concat).reshape(-1)
    neg_concat = np.asarray(neg_flat).reshape(len(eval_data.src_node_ids), eval_num_negs)
    y_true = np.concatenate(
        [np.ones(len(pos_concat), dtype=np.int64), np.zeros(neg_concat.size, dtype=np.int64)]
    )
    y_score = np.concatenate([pos_concat, neg_concat.reshape(-1)])
    auc, ap = auc_ap(np.array(y_true), np.array(y_score)) if len(y_true) > 0 else (0.5, 0.0)
    mrr = mrr_from_pos_neg(pos_concat, neg_concat) if pos_concat.size > 0 else 0.0
    all_zero = (pos_concat.reshape(-1, 1) == 0.0) & (neg_concat == 0.0)
    return {
        "mrr": mrr,
        "auc": auc,
        "ap": ap,
        "eval_pos_edges": int(len(eval_data.src_node_ids)),
        "history_pos_edges": int(len(train.src_node_ids)),
        "pos_history_hits": int((pos_concat > 0.5).sum()),
        "pos_history_hit_rate": float((pos_concat > 0.5).mean()) if pos_concat.size > 0 else 0.0,
        "all_zero_tie_rate": float(all_zero.all(axis=1).mean()) if pos_concat.size > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-num-negs", type=int, default=20)
    ap.add_argument("--memory-mode", default="unlimited_memory", choices=["unlimited_memory", "time_window_memory", "repeat_threshold_memory"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    obj = torch.load(args.data, map_location="cpu")
    train_idx = obj["split"]["train"].numpy()
    val_idx = obj["split"]["val"].numpy()
    test_idx = obj["split"]["test"].numpy()

    val_metrics = evaluate(obj, train_idx, val_idx, args.eval_num_negs, args.memory_mode, rng)
    hist_idx = np.concatenate([train_idx, val_idx], axis=0)
    test_metrics = evaluate(obj, hist_idx, test_idx, args.eval_num_negs, args.memory_mode, rng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "split": "val",
                "model": "EdgeBank",
                "mrr": val_metrics["mrr"],
                "auc": val_metrics["auc"],
                "ap": val_metrics["ap"],
                "memory_mode": args.memory_mode,
                "seed": args.seed,
                **{k: v for k, v in val_metrics.items() if k not in {"mrr", "auc", "ap"}},
            },
            {
                "split": "test",
                "model": "EdgeBank",
                "mrr": test_metrics["mrr"],
                "auc": test_metrics["auc"],
                "ap": test_metrics["ap"],
                "memory_mode": args.memory_mode,
                "seed": args.seed,
                **{k: v for k, v in test_metrics.items() if k not in {"mrr", "auc", "ap"}},
            },
        ]
    )
    df.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
