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
ROOT_REPEATMIXER = WORKSPACE_ROOT / "baselines" / "src" / "RepeatMixer-main"
root_str = str(ROOT_REPEATMIXER)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from models.RepeatMixer import RepeatMixer
from models.modules import MergeSingleLayer
from utils.DataLoader import Data, get_idx_data_loader
from utils.utils import NegativeEdgeSampler, get_historical_neighbor_sampler, get_neighbor_sampler


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


def sanitize_logits(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)


def logits_to_prob(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(sanitize_logits(x))


def pad_feature_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    if arr.shape[1] >= target_dim:
        return arr
    pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def build_data_objects(obj):
    src_all = obj["edges"]["src"].numpy().astype(np.int64)
    dst_all = obj["edges"]["dst"].numpy().astype(np.int64)
    ts_all = obj["edges"]["ts"].numpy().astype(np.float64)
    edge_feat_all = obj["edges"]["edge_feat"].numpy().astype(np.float32)
    split = obj["split"]

    def sanitize(ids):
        arr = np.asarray(ids, dtype=np.int64).reshape(-1)
        arr = arr[(arr >= 0) & (arr < src_all.shape[0])]
        if arr.size == 0:
            return arr
        return arr[np.argsort(ts_all[arr], kind="mergesort")]

    train_old = sanitize(split["train"].numpy())
    val_old = sanitize(split["val"].numpy())
    test_old = sanitize(split["test"].numpy())
    order = np.sort(np.concatenate([x for x in (train_old, val_old, test_old) if x.size > 0], axis=0))
    old_to_new_edge = np.full(src_all.shape[0], -1, dtype=np.int64)
    old_to_new_edge[order] = np.arange(order.shape[0], dtype=np.int64)
    train_idx = old_to_new_edge[train_old]
    val_idx = old_to_new_edge[val_old]
    test_idx = old_to_new_edge[test_old]

    src = src_all[order]
    dst = dst_all[order]
    ts = ts_all[order]
    edge_feat = edge_feat_all[order]

    active_nodes = np.unique(np.concatenate([src, dst], axis=0))
    src = np.searchsorted(active_nodes, src) + 1
    dst = np.searchsorted(active_nodes, dst) + 1

    node_feat_all = obj["node_feat"].numpy().astype(np.float32)
    node_feat = node_feat_all[active_nodes]
    target_dim = int(max(172, node_feat.shape[1], edge_feat.shape[1]))
    node_feat = pad_feature_dim(node_feat, target_dim)
    edge_feat = pad_feature_dim(edge_feat, target_dim)

    edge_ids = np.arange(src.shape[0], dtype=np.int64) + 1
    labels = np.ones(src.shape[0], dtype=np.float32)
    edge_pairs = np.stack([src, dst], axis=1)
    _, raw_edge_ids = np.unique(edge_pairs, axis=0, return_inverse=True)
    raw_edge_ids = raw_edge_ids.astype(np.int64) + 1

    node_raw = np.vstack([np.zeros((1, node_feat.shape[1]), dtype=np.float32), node_feat])
    edge_raw = np.vstack([np.zeros((1, edge_feat.shape[1]), dtype=np.float32), edge_feat])

    full_data = Data(
        src_node_ids=src,
        dst_node_ids=dst,
        node_interact_times=ts,
        edge_ids=edge_ids,
        labels=labels,
        raw_edge_ids=raw_edge_ids,
    )
    train_data = Data(
        src_node_ids=src[train_idx],
        dst_node_ids=dst[train_idx],
        node_interact_times=ts[train_idx],
        edge_ids=edge_ids[train_idx],
        labels=labels[train_idx],
        raw_edge_ids=raw_edge_ids[train_idx],
    )
    val_data = Data(
        src_node_ids=src[val_idx],
        dst_node_ids=dst[val_idx],
        node_interact_times=ts[val_idx],
        edge_ids=edge_ids[val_idx],
        labels=labels[val_idx],
        raw_edge_ids=raw_edge_ids[val_idx],
    )
    test_data = Data(
        src_node_ids=src[test_idx],
        dst_node_ids=dst[test_idx],
        node_interact_times=ts[test_idx],
        edge_ids=edge_ids[test_idx],
        labels=labels[test_idx],
        raw_edge_ids=raw_edge_ids[test_idx],
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
            data_obj.raw_edge_ids[idx],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=200, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--eval-num-negs", default=20, type=int)
    parser.add_argument("--num-neighbors", default=20, type=int)
    parser.add_argument("--time-gap", default=2000, type=int)
    parser.add_argument("--time-feat-dim", default=100, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--high-order", action="store_true")
    parser.add_argument(
        "--out",
        default="/home/xuxj/2026/dyn_relation_memory/results/baseline_formal_aminer_custom/metrics/repeatmixer_metrics.csv",
        type=str,
    )
    args = parser.parse_args()

    if args.gpu_id < 0:
        device = "cpu"
    elif torch.cuda.is_available():
        if 0 <= args.gpu_id < torch.cuda.device_count():
            device = f"cuda:{args.gpu_id}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"

    obj = torch.load(args.data, map_location="cpu")
    node_raw, edge_raw, full_data, train_data, val_data, test_data = build_data_objects(obj=obj)

    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy="recent", seed=0)
    historical_train_sampler = get_historical_neighbor_sampler(data=train_data, sample_neighbor_strategy="recent", seed=1)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy="recent", seed=2)
    full_historical_sampler = get_historical_neighbor_sampler(data=full_data, sample_neighbor_strategy="recent", seed=3)

    last_train_time = float(np.max(train_data.node_interact_times)) if len(train_data.node_interact_times) > 0 else 0.0
    train_neg_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids,
        dst_node_ids=train_data.dst_node_ids,
        interact_times=train_data.node_interact_times,
        last_observed_time=last_train_time,
        negative_sample_strategy="random",
        seed=0,
    )
    eval_neg_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids,
        dst_node_ids=full_data.dst_node_ids,
        interact_times=full_data.node_interact_times,
        last_observed_time=float(np.max(full_data.node_interact_times)),
        negative_sample_strategy="random",
        seed=1,
    )

    backbone = RepeatMixer(
        node_raw_features=node_raw,
        edge_raw_features=edge_raw,
        neighbor_sampler=train_neighbor_sampler,
        edge_neighbor_sampler=historical_train_sampler,
        time_feat_dim=args.time_feat_dim,
        num_tokens=args.num_neighbors,
        high_order=args.high_order,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device,
    )
    link_predictor = MergeSingleLayer(
        input_dim1=node_raw.shape[1],
        hidden_dim=max(32, node_raw.shape[1] // 2),
        output_dim=1,
    )
    model = nn.Sequential(backbone, link_predictor).to(device)
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    def train_one_epoch():
        model.train()
        model[0].set_neighbor_sampler(train_neighbor_sampler)
        model[0].set_edge_neighbor_sampler(historical_train_sampler)
        total_loss = 0.0
        n_batches = math.ceil(len(train_data.src_node_ids) / args.batch_size)
        for batch_src, batch_dst, batch_ts, _, _ in iterate_batches(train_data, args.batch_size):
            _, neg_dst = train_neg_sampler.sample(size=len(batch_src))
            neg_src = batch_src

            pos_emb = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=batch_src,
                dst_node_ids=batch_dst,
                node_interact_times=batch_ts,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )
            neg_emb = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=neg_src,
                dst_node_ids=neg_dst,
                node_interact_times=batch_ts,
                mask=True,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )

            pos_logit = sanitize_logits(model[1](input_1=pos_emb, input_2=None, input_3=None).squeeze(dim=-1))
            neg_logit = sanitize_logits(model[1](input_1=neg_emb, input_2=None, input_3=None).squeeze(dim=-1))
            pred = torch.cat([pos_logit, neg_logit], dim=0)
            label = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=0)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(eval_data):
        model.eval()
        model[0].set_neighbor_sampler(full_neighbor_sampler)
        model[0].set_edge_neighbor_sampler(full_historical_sampler)

        y_true, y_score = [], []
        pos_all, neg_all = [], []
        for batch_src, batch_dst, batch_ts, _, _ in iterate_batches(eval_data, args.batch_size):
            pos_emb = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=batch_src,
                dst_node_ids=batch_dst,
                node_interact_times=batch_ts,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )
            pos_prob = logits_to_prob(model[1](input_1=pos_emb, input_2=None, input_3=None).squeeze(dim=-1)).cpu().numpy()
            y_true.extend([1] * len(pos_prob))
            y_score.extend(pos_prob.tolist())

            neg_scores = []
            for _ in range(args.eval_num_negs):
                _, neg_dst = eval_neg_sampler.sample(size=len(batch_src))
                neg_emb = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=batch_src,
                    dst_node_ids=neg_dst,
                    node_interact_times=batch_ts,
                    mask=True,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                )
                neg_prob = logits_to_prob(model[1](input_1=neg_emb, input_2=None, input_3=None).squeeze(dim=-1)).cpu().numpy()
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
        val_mrr, val_auc, val_ap = evaluate(val_data)
        ep_sec = time.time() - t0
        epoch_secs.append(ep_sec)
        print(
            f"epoch={epoch} loss={train_loss:.6f} val_mrr={val_mrr:.6f} "
            f"val_auc={val_auc:.6f} val_ap={val_ap:.6f} sec={ep_sec:.2f}"
        )

    test_mrr, test_auc, test_ap = evaluate(test_data)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    avg_ep_sec = float(np.mean(epoch_secs)) if len(epoch_secs) > 0 else 0.0
    out.write_text(
        "split,model,mrr,auc,ap,avg_epoch_sec,total_params,trainable_params,device,eval_num_negs\n"
        + f"val,RepeatMixer,{val_mrr},{val_auc},{val_ap},{avg_ep_sec},{total_params},{trainable_params},{device},{args.eval_num_negs}\n"
        + f"test,RepeatMixer,{test_mrr},{test_auc},{test_ap},{avg_ep_sec},{total_params},{trainable_params},{device},{args.eval_num_negs}\n",
        encoding="utf-8",
    )
    print(
        f"test_mrr={test_mrr:.6f} test_auc={test_auc:.6f} test_ap={test_ap:.6f} "
        f"params={total_params} trainable={trainable_params} avg_epoch_sec={avg_ep_sec:.2f}"
    )


if __name__ == "__main__":
    main()
