#!/usr/bin/env python3
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ROOT_TGB2 = WORKSPACE_ROOT / "baselines" / "src" / "TGB2-main" / "TGB2-main"
ROOT_DYGLIB = WORKSPACE_ROOT / "baselines" / "src" / "DyGLib_TGB-master" / "DyGLib_TGB-master"
for root in (ROOT_TGB2, ROOT_DYGLIB):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.modules import MergeLayer
from utils.DataLoader import Data, get_idx_data_loader
from utils.utils import NegativeEdgeSampler, get_neighbor_sampler


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


def build_data_objects(obj, max_edges: int, train_ratio: float, val_ratio: float):
    src_all = obj["edges"]["src"].numpy().astype(np.int64)
    dst_all = obj["edges"]["dst"].numpy().astype(np.int64)
    ts_all = obj["edges"]["ts"].numpy().astype(np.float64)
    edge_feat_all = obj["edges"]["edge_feat"].numpy().astype(np.float32)

    def sanitize(ids):
        arr = np.asarray(ids, dtype=np.int64).reshape(-1)
        arr = arr[(arr >= 0) & (arr < src_all.shape[0])]
        if arr.size == 0:
            return arr
        return arr[np.argsort(ts_all[arr], kind="mergesort")]

    def proportional_take(splits, budget):
        sizes = {k: int(v.size) for k, v in splits.items()}
        total = sum(sizes.values())
        if budget <= 0 or total <= budget:
            return splits
        nonempty = [k for k, v in sizes.items() if v > 0]
        if not nonempty:
            return splits
        alloc = {k: int(budget * sizes[k] / total) for k in splits}
        if budget >= len(nonempty):
            for k in nonempty:
                alloc[k] = max(1, alloc[k])
        remain = budget - sum(min(sizes[k], alloc[k]) for k in splits)
        alloc = {k: min(sizes[k], alloc[k]) for k in splits}
        while remain > 0:
            cands = [k for k in nonempty if alloc[k] < sizes[k]]
            if not cands:
                break
            cands.sort(key=lambda k: (sizes[k] - alloc[k], sizes[k]), reverse=True)
            alloc[cands[0]] += 1
            remain -= 1
        return {k: splits[k][: alloc[k]] for k in splits}

    split_obj = obj.get("split", {})
    has_split = isinstance(split_obj, dict) and all(k in split_obj for k in ("train", "val", "test"))
    if has_split:
        raw_splits = {
            "train": sanitize(split_obj["train"].numpy()),
            "val": sanitize(split_obj["val"].numpy()),
            "test": sanitize(split_obj["test"].numpy()),
        }
        keep_splits = proportional_take(raw_splits, int(max_edges))
        order = np.sort(np.concatenate([v for v in keep_splits.values() if v.size > 0], axis=0))
        old_to_new_edge = np.full(src_all.shape[0], -1, dtype=np.int64)
        old_to_new_edge[order] = np.arange(order.shape[0], dtype=np.int64)
        train_idx = old_to_new_edge[keep_splits["train"]]
        val_idx = old_to_new_edge[keep_splits["val"]]
        test_idx = old_to_new_edge[keep_splits["test"]]
    else:
        order = np.argsort(ts_all)
        if max_edges > 0:
            order = order[: min(max_edges, len(order))]
        order = np.sort(order)
        n = order.shape[0]
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = np.arange(0, n_train, dtype=np.int64)
        val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
        test_idx = np.arange(n_train + n_val, n, dtype=np.int64)

    src = src_all[order]
    dst = dst_all[order]
    ts = ts_all[order]
    edge_feat = edge_feat_all[order]

    # Reindex to active nodes to keep smoke validation fast on large sparse IDs.
    active_nodes = np.unique(np.concatenate([src, dst], axis=0))
    src = np.searchsorted(active_nodes, src) + 1
    dst = np.searchsorted(active_nodes, dst) + 1

    node_feat_all = obj["node_feat"].numpy().astype(np.float32)
    node_feat = node_feat_all[active_nodes]
    edge_ids = np.arange(src.shape[0], dtype=np.int64) + 1
    labels = np.ones(src.shape[0], dtype=np.float32)

    node_raw = np.vstack([np.zeros((1, node_feat.shape[1]), dtype=np.float32), node_feat])
    edge_raw = np.vstack([np.zeros((1, edge_feat.shape[1]), dtype=np.float32), edge_feat])

    full_data = Data(src_node_ids=src, dst_node_ids=dst, node_interact_times=ts, edge_ids=edge_ids, labels=labels)
    train_data = Data(
        src_node_ids=src[train_idx],
        dst_node_ids=dst[train_idx],
        node_interact_times=ts[train_idx],
        edge_ids=edge_ids[train_idx],
        labels=labels[train_idx],
    )
    val_data = Data(
        src_node_ids=src[val_idx],
        dst_node_ids=dst[val_idx],
        node_interact_times=ts[val_idx],
        edge_ids=edge_ids[val_idx],
        labels=labels[val_idx],
    )
    test_data = Data(
        src_node_ids=src[test_idx],
        dst_node_ids=dst[test_idx],
        node_interact_times=ts[test_idx],
        edge_ids=edge_ids[test_idx],
        labels=labels[test_idx],
    )
    return node_raw, edge_raw, full_data, train_data, val_data, test_data


def iterate_batches(data_obj, batch_size):
    loader = get_idx_data_loader(indices_list=list(range(len(data_obj.src_node_ids))), batch_size=batch_size, shuffle=False)
    for indices in loader:
        idx = indices.numpy()
        yield (
            data_obj.src_node_ids[idx],
            data_obj.dst_node_ids[idx],
            data_obj.node_interact_times[idx],
            data_obj.edge_ids[idx],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=200, type=int)
    parser.add_argument("--max-edges", default=10000, type=int)
    parser.add_argument("--train-ratio", default=0.70, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--eval-num-negs", default=20, type=int)
    parser.add_argument(
        "--out",
        default="/home/xuxj/2026/dyn_hetero_community/results/baseline_status/dyglib_custom_metrics.csv",
        type=str,
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        if 0 <= args.gpu_id < torch.cuda.device_count():
            device = f"cuda:{args.gpu_id}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
    obj = torch.load(args.data, map_location="cpu")
    node_raw, edge_raw, full_data, train_data, val_data, test_data = build_data_objects(
        obj=obj,
        max_edges=args.max_edges,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy="uniform", seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy="uniform", seed=1)

    train_neg_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids, seed=0)
    eval_neg_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=1)

    src_mean, src_std, dst_mean, dst_std = compute_src_dst_node_time_shifts(
        train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times
    )
    output_dim = int(max(172, node_raw.shape[1]))
    model_backbone = MemoryModel(
        node_raw_features=node_raw,
        edge_raw_features=edge_raw,
        neighbor_sampler=train_neighbor_sampler,
        time_feat_dim=100,
        output_dim=output_dim,
        model_name="TGN",
        num_layers=1,
        num_heads=2,
        dropout=0.1,
        src_node_mean_time_shift=src_mean,
        src_node_std_time_shift=src_std,
        dst_node_mean_time_shift_dst=dst_mean,
        dst_node_std_time_shift=dst_std,
        device=device,
    )
    link_predictor = MergeLayer(input_dim1=output_dim, input_dim2=output_dim, hidden_dim=output_dim, output_dim=1)
    model = nn.Sequential(model_backbone, link_predictor).to(device)
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    def train_one_epoch():
        model.train()
        model[0].set_neighbor_sampler(train_neighbor_sampler)
        model[0].memory_bank.__init_memory_bank__()
        total_loss = 0.0
        n_batches = math.ceil(len(train_data.src_node_ids) / args.batch_size)
        for batch_src, batch_dst, batch_ts, batch_edge_ids in iterate_batches(train_data, args.batch_size):
            _, neg_dst = train_neg_sampler.sample(size=len(batch_src))
            neg_src = batch_src

            neg_src_emb, neg_dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=neg_src,
                dst_node_ids=neg_dst,
                node_interact_times=batch_ts,
                edge_ids=None,
                edges_are_positive=False,
                num_neighbors=10,
            )
            src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=batch_src,
                dst_node_ids=batch_dst,
                node_interact_times=batch_ts,
                edge_ids=batch_edge_ids,
                edges_are_positive=True,
                num_neighbors=10,
            )

            pos_prob = model[1](input_1=src_emb, input_2=dst_emb).squeeze(dim=-1).sigmoid()
            neg_prob = model[1](input_1=neg_src_emb, input_2=neg_dst_emb).squeeze(dim=-1).sigmoid()
            pred = torch.cat([pos_prob, neg_prob], dim=0)
            label = torch.cat([torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model[0].memory_bank.detach_memory_bank()
            total_loss += float(loss.item())
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(eval_data, history_data):
        model.eval()
        model[0].set_neighbor_sampler(full_neighbor_sampler)
        model[0].memory_bank.__init_memory_bank__()

        # Replay history interactions in temporal order to build memory state.
        for batch_src, batch_dst, batch_ts, batch_edge_ids in iterate_batches(history_data, args.batch_size):
            model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=batch_src,
                dst_node_ids=batch_dst,
                node_interact_times=batch_ts,
                edge_ids=batch_edge_ids,
                edges_are_positive=True,
                num_neighbors=10,
            )

        y_true, y_score = [], []
        pos_all, neg_all = [], []
        for batch_src, batch_dst, batch_ts, batch_edge_ids in iterate_batches(eval_data, args.batch_size):
            src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=batch_src,
                dst_node_ids=batch_dst,
                node_interact_times=batch_ts,
                edge_ids=batch_edge_ids,
                edges_are_positive=True,
                num_neighbors=10,
            )

            pos_prob = model[1](input_1=src_emb, input_2=dst_emb).squeeze(dim=-1).sigmoid().cpu().numpy()
            y_true.extend([1] * len(pos_prob))
            y_score.extend(pos_prob.tolist())
            neg_scores = []
            for _ in range(args.eval_num_negs):
                _, neg_dst = eval_neg_sampler.sample(size=len(batch_src))
                neg_src = batch_src
                neg_src_emb, neg_dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=neg_src,
                    dst_node_ids=neg_dst,
                    node_interact_times=batch_ts,
                    edge_ids=None,
                    edges_are_positive=False,
                    num_neighbors=10,
                )
                neg_prob = model[1](input_1=neg_src_emb, input_2=neg_dst_emb).squeeze(dim=-1).sigmoid().cpu().numpy()
                neg_scores.append(neg_prob)
            neg_scores = np.stack(neg_scores, axis=1)
            y_true.extend([0] * int(neg_scores.size))
            y_score.extend(neg_scores.reshape(-1).tolist())
            pos_all.append(pos_prob)
            neg_all.append(neg_scores)
        auc, ap = auc_ap(y_true=y_true, y_score=y_score)
        mrr = mrr_from_pos_neg(np.concatenate(pos_all, axis=0), np.concatenate(neg_all, axis=0))
        return mrr, auc, ap

    val_mrr = val_auc = val_ap = 0.0
    epoch_secs = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch()
        val_mrr, val_auc, val_ap = evaluate(val_data, train_data)
        ep_sec = time.time() - t0
        epoch_secs.append(ep_sec)
        print(
            f"epoch={epoch} loss={train_loss:.6f} val_mrr={val_mrr:.6f} "
            f"val_auc={val_auc:.6f} val_ap={val_ap:.6f} sec={ep_sec:.2f}"
        )

    merged_src = np.concatenate([train_data.src_node_ids, val_data.src_node_ids], axis=0)
    merged_dst = np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids], axis=0)
    merged_ts = np.concatenate([train_data.node_interact_times, val_data.node_interact_times], axis=0)
    merged_eid = np.concatenate([train_data.edge_ids, val_data.edge_ids], axis=0)
    merged_labels = np.concatenate([train_data.labels, val_data.labels], axis=0)
    history_data = Data(
        src_node_ids=merged_src, dst_node_ids=merged_dst, node_interact_times=merged_ts, edge_ids=merged_eid, labels=merged_labels
    )
    test_mrr, test_auc, test_ap = evaluate(test_data, history_data)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    avg_ep_sec = float(np.mean(epoch_secs)) if len(epoch_secs) > 0 else 0.0
    out.write_text(
        "split,mrr,auc,ap,avg_epoch_sec,total_params,trainable_params,device,eval_num_negs\n"
        + f"val,{val_mrr},{val_auc},{val_ap},{avg_ep_sec},{total_params},{trainable_params},{device},{args.eval_num_negs}\n"
        + f"test,{test_mrr},{test_auc},{test_ap},{avg_ep_sec},{total_params},{trainable_params},{device},{args.eval_num_negs}\n",
        encoding="utf-8",
    )
    print(
        f"test_mrr={test_mrr:.6f} test_auc={test_auc:.6f} test_ap={test_ap:.6f} "
        f"params={total_params} trainable={trainable_params} avg_epoch_sec={avg_ep_sec:.2f}"
    )


if __name__ == "__main__":
    main()
