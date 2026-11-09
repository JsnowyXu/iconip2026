#!/usr/bin/env python3
import argparse
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ROOT_TGB2 = WORKSPACE_ROOT / "baselines" / "src" / "TGB2-main" / "TGB2-main"
ROOT_DYGLIB = WORKSPACE_ROOT / "baselines" / "src" / "RepeatMixer-main"
for root in (ROOT_TGB2, ROOT_DYGLIB):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

# Avoid picking up an incompatible system cuDNN when running the DyGLib zoo.
os.environ.pop("LD_LIBRARY_PATH", None)

from models.CAWN import CAWN
from models.DyGFormer import DyGFormer
from models.GraphMixer import GraphMixer
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.TCL import TCL
from models.TGAT import TGAT
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


def sanitize_logits(x: torch.Tensor) -> torch.Tensor:
    # Some DyGLib backbones can produce NaN/Inf spikes on heterogeneous task
    # adapters. Keep the probe numerically safe instead of tripping CUDA BCE.
    return torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)


def logits_to_prob(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(sanitize_logits(x))


def f1_binary(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    if tp == 0:
        return 0.0
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    if p + r == 0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def _topk_indices(probs, k):
    k = int(max(1, min(k, probs.shape[1])))
    part = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(probs, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def _precision_recall_at_k(y_true, probs, k):
    topk = _topk_indices(probs, k)
    hits = np.take_along_axis(y_true, topk, axis=1).astype(np.float64)
    precision = float((hits.sum(axis=1) / float(k)).mean())
    positives = y_true.sum(axis=1).astype(np.float64)
    valid = positives > 0
    recall = float((hits.sum(axis=1)[valid] / positives[valid]).mean()) if valid.any() else 0.0
    return precision, recall


def _ndcg_at_k(y_true, probs, k):
    topk = _topk_indices(probs, k)
    gains = np.take_along_axis(y_true, topk, axis=1).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = (gains * discounts.reshape(1, -1)).sum(axis=1)
    positives = y_true.sum(axis=1).astype(np.int64)
    ideal = np.zeros_like(dcg)
    for i, cnt in enumerate(positives.tolist()):
        if cnt <= 0:
            continue
        cur_k = min(int(k), int(cnt))
        ideal[i] = discounts[:cur_k].sum()
    valid = ideal > 0
    return float((dcg[valid] / ideal[valid]).mean()) if valid.any() else 0.0


def node_multilabel_metrics(y_true, logits):
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(np.int64)
    micro_f1 = f1_binary(y_true.reshape(-1), pred.reshape(-1))
    macro = []
    for j in range(y_true.shape[1]):
        macro.append(f1_binary(y_true[:, j], pred[:, j]))
    out = {
        "node_ml_micro_f1": float(micro_f1),
        "node_ml_macro_f1": float(np.mean(macro) if macro else 0.0),
    }
    for k in [1, 3, 5]:
        kk = min(int(k), int(probs.shape[1]))
        precision, recall = _precision_recall_at_k(y_true, probs, kk)
        out[f"node_ml_precision_at_{k}"] = precision
        out[f"node_ml_recall_at_{k}"] = recall
        out[f"node_ml_ndcg_at_{k}"] = _ndcg_at_k(y_true, probs, kk)
    return out


def temporal_node_split(node_ids_orig: np.ndarray, old_to_new: np.ndarray, obj: dict):
    valid = (node_ids_orig >= 0) & (node_ids_orig < old_to_new.shape[0]) & (old_to_new[node_ids_orig] > 0)
    if not valid.any():
        z = np.array([], dtype=np.int64)
        return z, z, z, z
    keep_orig = node_ids_orig[valid]
    keep_new = old_to_new[keep_orig].astype(np.int64)

    ts_map = {}
    extra = obj.get("meta", {}).get("extra", {})
    pat = extra.get("patent_attributes", {}) if isinstance(extra, dict) else {}
    pat_ids = pat.get("node_ids", []) if isinstance(pat, dict) else []
    pat_ts = pat.get("publication_ts", []) if isinstance(pat, dict) else []
    for nid, t in zip(pat_ids, pat_ts):
        ts_map[int(nid)] = float(t)

    ts = np.array([ts_map.get(int(n), float(i)) for i, n in enumerate(keep_orig)], dtype=np.float64)
    order = np.argsort(ts)
    n = len(order)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.15)
    tr = order[:n_tr]
    va = order[n_tr : n_tr + n_va]
    te = order[n_tr + n_va :]
    return keep_new, keep_orig, tr.astype(np.int64), te.astype(np.int64)


def build_data_objects(obj, max_edges: int, supervised_only: bool = True):
    src_all = obj["edges"]["src"].numpy().astype(np.int64)
    dst_all = obj["edges"]["dst"].numpy().astype(np.int64)
    ts_all = obj["edges"]["ts"].numpy().astype(np.float64)
    edge_feat_all = obj["edges"]["edge_feat"].numpy().astype(np.float32)
    label_all = obj["edges"].get("edge_label", torch.ones_like(obj["edges"]["src"]).float()).numpy().astype(np.float32)

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
        if supervised_only:
            raw_splits = {k: v[label_all[v] > 0.5] for k, v in raw_splits.items()}
        keep_splits = proportional_take(raw_splits, int(max_edges))
        order = np.concatenate([v for v in keep_splits.values() if v.size > 0], axis=0)
        order = order[np.argsort(ts_all[order], kind="mergesort")]
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
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_idx = np.arange(0, n_train, dtype=np.int64)
        val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
        test_idx = np.arange(n_train + n_val, n, dtype=np.int64)

    src = src_all[order]
    dst = dst_all[order]
    ts = ts_all[order]
    edge_feat = edge_feat_all[order]

    active_nodes = np.unique(np.concatenate([src, dst], axis=0))
    src = np.searchsorted(active_nodes, src) + 1
    dst = np.searchsorted(active_nodes, dst) + 1

    node_feat_all = obj["node_feat"].numpy().astype(np.float32)
    node_feat = node_feat_all[active_nodes]
    edge_ids = np.arange(src.shape[0], dtype=np.int64) + 1
    edge_pairs = np.stack([src, dst], axis=1)
    _, raw_edge_ids = np.unique(edge_pairs, axis=0, return_inverse=True)
    raw_edge_ids = raw_edge_ids.astype(np.int64) + 1
    labels = label_all[order].astype(np.float32)

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
    old_to_new = np.full((int(obj["num_nodes"]),), -1, dtype=np.int64)
    old_to_new[active_nodes] = np.arange(active_nodes.shape[0], dtype=np.int64) + 1

    return node_raw, edge_raw, full_data, train_data, val_data, test_data, active_nodes, old_to_new


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


def build_model(
    model_name: str,
    node_raw: np.ndarray,
    edge_raw: np.ndarray,
    train_neighbor_sampler,
    train_data: Data,
    device: str,
    output_dim: int,
):
    emb_dim = int(node_raw.shape[1])
    if model_name == "TGAT":
        backbone = TGAT(
            node_raw_features=node_raw,
            edge_raw_features=edge_raw,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=100,
            num_layers=1,
            num_heads=2,
            dropout=0.1,
            device=device,
        )
    elif model_name in ["JODIE", "DyRep", "TGN"]:
        src_mean, src_std, dst_mean, dst_std = compute_src_dst_node_time_shifts(
            train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times
        )
        backbone = MemoryModel(
            node_raw_features=node_raw,
            edge_raw_features=edge_raw,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=100,
            model_name=model_name,
            num_layers=1,
            num_heads=2,
            dropout=0.1,
            src_node_mean_time_shift=src_mean,
            src_node_std_time_shift=src_std,
            dst_node_mean_time_shift_dst=dst_mean,
            dst_node_std_time_shift=dst_std,
            device=device,
        )
    elif model_name == "CAWN":
        backbone = CAWN(
            node_raw_features=node_raw,
            edge_raw_features=edge_raw,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=100,
            position_feat_dim=32,
            walk_length=2,
            num_walk_heads=4,
            dropout=0.1,
            device=device,
        )
    elif model_name == "TCL":
        backbone = TCL(
            node_raw_features=node_raw,
            edge_raw_features=edge_raw,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=100,
            num_layers=1,
            num_heads=2,
            num_depths=11,
            dropout=0.1,
            device=device,
        )
    elif model_name == "GraphMixer":
        backbone = GraphMixer(
            node_raw_features=node_raw,
            edge_raw_features=edge_raw,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=100,
            num_tokens=10,
            num_layers=2,
            dropout=0.1,
            device=device,
        )
    elif model_name == "DyGFormer":
        backbone = DyGFormer(
            node_raw_features=node_raw,
            edge_raw_features=edge_raw,
            neighbor_sampler=train_neighbor_sampler,
            time_feat_dim=100,
            channel_embedding_dim=32,
            patch_size=1,
            num_layers=1,
            num_heads=2,
            dropout=0.1,
            max_input_sequence_length=32,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    link_predictor = MergeLayer(input_dim1=emb_dim, input_dim2=emb_dim, input_dim3=0, hidden_dim=emb_dim, output_dim=1)
    model = nn.Sequential(backbone, link_predictor).to(device)
    return model


def compute_embeddings(model_name: str, model, bsrc, bdst, bts, beid, neg_dst, num_neighbors=10, time_gap=2000):
    neg_src = bsrc
    if model_name in ["TGAT", "CAWN", "TCL"]:
        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=bsrc, dst_node_ids=bdst, node_interact_times=bts, num_neighbors=num_neighbors
        )
        neg_src_emb, neg_dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=neg_src, dst_node_ids=neg_dst, node_interact_times=bts, num_neighbors=num_neighbors
        )
    elif model_name in ["JODIE", "DyRep", "TGN"]:
        neg_src_emb, neg_dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=neg_src,
            dst_node_ids=neg_dst,
            node_interact_times=bts,
            edge_ids=None,
            edges_are_positive=False,
            num_neighbors=num_neighbors,
        )
        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=bsrc,
            dst_node_ids=bdst,
            node_interact_times=bts,
            edge_ids=beid,
            edges_are_positive=True,
            num_neighbors=num_neighbors,
        )
    elif model_name == "GraphMixer":
        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=bsrc,
            dst_node_ids=bdst,
            node_interact_times=bts,
            num_neighbors=num_neighbors,
            time_gap=time_gap,
        )
        neg_src_emb, neg_dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=neg_src,
            dst_node_ids=neg_dst,
            node_interact_times=bts,
            num_neighbors=num_neighbors,
            time_gap=time_gap,
        )
    elif model_name == "DyGFormer":
        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=bsrc, dst_node_ids=bdst, node_interact_times=bts
        )
        neg_src_emb, neg_dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=neg_src, dst_node_ids=neg_dst, node_interact_times=bts
        )
    else:
        raise ValueError(model_name)
    return src_emb, dst_emb, neg_src_emb, neg_dst_emb


def compute_pos_embeddings(model_name: str, model, bsrc, bdst, bts, beid, num_neighbors=10, time_gap=2000):
    if model_name in ["TGAT", "CAWN", "TCL"]:
        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=bsrc, dst_node_ids=bdst, node_interact_times=bts, num_neighbors=num_neighbors
        )
    elif model_name in ["JODIE", "DyRep", "TGN"]:
        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=bsrc,
            dst_node_ids=bdst,
            node_interact_times=bts,
            edge_ids=beid,
            edges_are_positive=True,
            num_neighbors=num_neighbors,
        )
    elif model_name == "GraphMixer":
        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=bsrc,
            dst_node_ids=bdst,
            node_interact_times=bts,
            num_neighbors=num_neighbors,
            time_gap=time_gap,
        )
    elif model_name == "DyGFormer":
        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=bsrc, dst_node_ids=bdst, node_interact_times=bts
        )
    else:
        raise ValueError(model_name)
    return src_emb, dst_emb


@torch.no_grad()
def extract_node_embeddings(
    model_name: str,
    model,
    history_data,
    batch_size: int,
    max_node_id: int,
):
    if model_name in ["DyRep", "TGAT", "TGN", "CAWN", "TCL", "GraphMixer", "DyGFormer"]:
        # neighbor sampler is set by caller
        pass
    if model_name in ["JODIE", "DyRep", "TGN"]:
        model[0].memory_bank.__init_memory_bank__()
    model.eval()

    device = next(model.parameters()).device
    sums = None
    cnt = torch.zeros((max_node_id + 1, 1), dtype=torch.float32, device=device)

    for bsrc, bdst, bts, beid in iterate_batches(history_data, batch_size):
        src_emb, dst_emb = compute_pos_embeddings(model_name, model, bsrc, bdst, bts, beid, num_neighbors=10)
        if sums is None:
            sums = torch.zeros((max_node_id + 1, int(src_emb.shape[1])), dtype=torch.float32, device=device)
        bsrc_t = torch.from_numpy(bsrc).long().to(sums.device)
        bdst_t = torch.from_numpy(bdst).long().to(sums.device)
        sums.index_add_(0, bsrc_t, src_emb.detach())
        sums.index_add_(0, bdst_t, dst_emb.detach())
        cnt.index_add_(0, bsrc_t, torch.ones((len(bsrc), 1), device=sums.device))
        cnt.index_add_(0, bdst_t, torch.ones((len(bdst), 1), device=sums.device))

        if model_name in ["JODIE", "DyRep", "TGN"]:
            model[0].memory_bank.detach_memory_bank()

    if sums is None:
        return np.zeros((max_node_id + 1, 0), dtype=np.float32)
    emb = sums / cnt.clamp_min(1.0)
    return emb.detach().cpu().numpy()


def train_node_probe_multilabel(emb: np.ndarray, node_ids: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
    x = torch.from_numpy(emb[node_ids]).float()
    y = torch.from_numpy(labels.astype(np.float32)).float()
    tr = torch.from_numpy(train_idx).long()
    te = torch.from_numpy(test_idx).long()

    model = nn.Linear(x.shape[1], y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    for _ in range(30):
        model.train()
        logits = model(x[tr])
        loss = F.binary_cross_entropy_with_logits(logits, y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x[te]).cpu().numpy()
    met = node_multilabel_metrics(y[te].cpu().numpy().astype(np.int64), logits)
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=str)
    ap.add_argument("--models", default="TGN,TGAT,CAWN,TCL,GraphMixer,DyGFormer", type=str)
    ap.add_argument("--epochs", default=1, type=int)
    ap.add_argument("--batch-size", default=200, type=int)
    ap.add_argument("--max-edges", default=0, type=int)
    ap.add_argument("--include-context-edges", action="store_true")
    ap.add_argument("--gpu-id", default=0, type=int)
    ap.add_argument("--eval-num-negs", default=20, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument(
        "--out",
        default="/home/xuxj/2026/dyn_hetero_community/results/baseline_status/dyglib_model_zoo_metrics.csv",
        type=str,
    )
    args = ap.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    task_name = Path(args.data).stem
    is_node_task = ("node_multilabel" in task_name) or ("node_multiclass" in task_name)
    node_raw, edge_raw, full_data, train_data, val_data, test_data, active_nodes, old_to_new = build_data_objects(
        obj=obj, max_edges=args.max_edges, supervised_only=not args.include_context_edges
    )

    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy="uniform", seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy="uniform", seed=1)
    train_neg_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids, seed=0)
    eval_neg_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=1)

    output_dim = int(max(172, node_raw.shape[1]))
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []

    ml_node_ids_new = np.array([], dtype=np.int64)
    ml_labels = np.zeros((0, 1), dtype=np.float32)
    tr_nodes = np.array([], dtype=np.int64)
    te_nodes = np.array([], dtype=np.int64)
    if is_node_task and obj.get("node_multilabel", None) is not None:
        ml_ids = obj["node_multilabel"]["node_ids"].numpy().astype(np.int64)
        ml = obj["node_multilabel"]["labels"].numpy().astype(np.float32)
        ml_node_ids_new, keep_orig, tr_nodes, te_nodes = temporal_node_split(ml_ids, old_to_new, obj)
        if ml_node_ids_new.size > 0:
            # map labels using kept original order
            id_to_pos = {int(n): i for i, n in enumerate(ml_ids.tolist())}
            keep_pos = np.array([id_to_pos[int(n)] for n in keep_orig], dtype=np.int64)
            ml_labels = ml[keep_pos]

    for model_name in model_names:
        print(f"=== running DyGLib model: {model_name} ===")
        try:
            model = build_model(model_name, node_raw, edge_raw, train_neighbor_sampler, train_data, device, output_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            total_params = int(sum(p.numel() for p in model.parameters()))
            trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            epoch_secs = []
            model_t0 = time.time()

            def train_one_epoch():
                model.train()
                if model_name in ["DyRep", "TGAT", "TGN", "CAWN", "TCL", "GraphMixer", "DyGFormer"]:
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                if model_name in ["JODIE", "DyRep", "TGN"]:
                    model[0].memory_bank.__init_memory_bank__()

                total_loss = 0.0
                n_batches = math.ceil(len(train_data.src_node_ids) / args.batch_size)
                for bsrc, bdst, bts, beid in iterate_batches(train_data, args.batch_size):
                    _, neg_dst = train_neg_sampler.sample(size=len(bsrc))
                    src_emb, dst_emb, neg_src_emb, neg_dst_emb = compute_embeddings(
                        model_name, model, bsrc, bdst, bts, beid, neg_dst, num_neighbors=10
                    )
                    pos_logit = sanitize_logits(model[1](input_1=src_emb, input_2=dst_emb, input_3=None).squeeze(dim=-1))
                    neg_logit = sanitize_logits(model[1](input_1=neg_src_emb, input_2=neg_dst_emb, input_3=None).squeeze(dim=-1))
                    pred = torch.cat([pos_logit, neg_logit], dim=0)
                    label = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=0)
                    loss = criterion(pred, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if model_name in ["JODIE", "DyRep", "TGN"]:
                        model[0].memory_bank.detach_memory_bank()
                    total_loss += float(loss.item())
                return total_loss / max(n_batches, 1)

            @torch.no_grad()
            def evaluate(eval_data, history_data):
                model.eval()
                if model_name in ["DyRep", "TGAT", "TGN", "CAWN", "TCL", "GraphMixer", "DyGFormer"]:
                    model[0].set_neighbor_sampler(full_neighbor_sampler)
                if model_name in ["JODIE", "DyRep", "TGN"]:
                    model[0].memory_bank.__init_memory_bank__()

                for bsrc, bdst, bts, beid in iterate_batches(history_data, args.batch_size):
                    if model_name in ["JODIE", "DyRep", "TGN"]:
                        model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=bsrc,
                            dst_node_ids=bdst,
                            node_interact_times=bts,
                            edge_ids=beid,
                            edges_are_positive=True,
                            num_neighbors=10,
                        )

                y_true, y_score = [], []
                pos_all, neg_all = [], []
                for bsrc, bdst, bts, beid in iterate_batches(eval_data, args.batch_size):
                    neg_dst_list = []
                    for _ in range(args.eval_num_negs):
                        _, neg_dst = eval_neg_sampler.sample(size=len(bsrc))
                        neg_dst_list.append(neg_dst)
                    src_emb, dst_emb = compute_pos_embeddings(
                        model_name, model, bsrc, bdst, bts, beid, num_neighbors=10
                    )
                    pos_prob = logits_to_prob(model[1](input_1=src_emb, input_2=dst_emb, input_3=None).squeeze(dim=-1)).cpu().numpy()
                    neg_probs = []
                    for neg_dst in neg_dst_list:
                        _, _, neg_src_emb, neg_dst_emb = compute_embeddings(
                            model_name, model, bsrc, bdst, bts, beid, neg_dst, num_neighbors=10
                        )
                        neg_prob = logits_to_prob(model[1](input_1=neg_src_emb, input_2=neg_dst_emb, input_3=None).squeeze(dim=-1)).cpu().numpy()
                        neg_probs.append(neg_prob)
                    neg_probs = np.stack(neg_probs, axis=1)
                    y_true.extend([1] * len(pos_prob))
                    y_score.extend(pos_prob.tolist())
                    y_true.extend([0] * int(neg_probs.shape[0] * neg_probs.shape[1]))
                    y_score.extend(neg_probs.reshape(-1).tolist())
                    pos_all.append(pos_prob)
                    neg_all.append(neg_probs)
                auc, ap = auc_ap(y_true=y_true, y_score=y_score)
                mrr = mrr_from_pos_neg(np.concatenate(pos_all, axis=0), np.concatenate(neg_all, axis=0))
                return mrr, auc, ap

            val_mrr = val_auc = val_ap = 0.0
            for ep in range(1, args.epochs + 1):
                t0 = time.time()
                train_loss = train_one_epoch()
                val_mrr, val_auc, val_ap = evaluate(val_data, train_data)
                ep_sec = time.time() - t0
                epoch_secs.append(ep_sec)
                print(
                    f"[{model_name}] epoch={ep} loss={train_loss:.6f} "
                    f"val_mrr={val_mrr:.6f} val_auc={val_auc:.6f} val_ap={val_ap:.6f} sec={ep_sec:.2f}"
                )

            hist = Data(
                src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids], axis=0),
                dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids], axis=0),
                node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times], axis=0),
                edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids], axis=0),
                labels=np.concatenate([train_data.labels, val_data.labels], axis=0),
                raw_edge_ids=np.concatenate([train_data.raw_edge_ids, val_data.raw_edge_ids], axis=0),
            )
            test_mrr, test_auc, test_ap = evaluate(test_data, hist)
            node_micro_f1 = np.nan
            node_macro_f1 = np.nan
            node_precision_at_1 = np.nan
            node_precision_at_3 = np.nan
            node_precision_at_5 = np.nan
            node_recall_at_1 = np.nan
            node_recall_at_3 = np.nan
            node_recall_at_5 = np.nan
            node_ndcg_at_1 = np.nan
            node_ndcg_at_3 = np.nan
            node_ndcg_at_5 = np.nan
            if ml_node_ids_new.size > 10 and tr_nodes.size > 5 and te_nodes.size > 5:
                if model_name in ["DyRep", "TGAT", "TGN", "CAWN", "TCL", "GraphMixer", "DyGFormer"]:
                    model[0].set_neighbor_sampler(full_neighbor_sampler)
                node_emb = extract_node_embeddings(
                    model_name=model_name,
                    model=model,
                    history_data=hist,
                    batch_size=args.batch_size,
                    max_node_id=int(max(full_data.src_node_ids.max(), full_data.dst_node_ids.max())),
                )
                node_met = train_node_probe_multilabel(
                    emb=node_emb,
                    node_ids=ml_node_ids_new,
                    labels=ml_labels,
                    train_idx=tr_nodes,
                    test_idx=te_nodes,
                )
                node_micro_f1 = float(node_met.get("node_ml_micro_f1", np.nan))
                node_macro_f1 = float(node_met.get("node_ml_macro_f1", np.nan))
                node_precision_at_1 = float(node_met.get("node_ml_precision_at_1", np.nan))
                node_precision_at_3 = float(node_met.get("node_ml_precision_at_3", np.nan))
                node_precision_at_5 = float(node_met.get("node_ml_precision_at_5", np.nan))
                node_recall_at_1 = float(node_met.get("node_ml_recall_at_1", np.nan))
                node_recall_at_3 = float(node_met.get("node_ml_recall_at_3", np.nan))
                node_recall_at_5 = float(node_met.get("node_ml_recall_at_5", np.nan))
                node_ndcg_at_1 = float(node_met.get("node_ml_ndcg_at_1", np.nan))
                node_ndcg_at_3 = float(node_met.get("node_ml_ndcg_at_3", np.nan))
                node_ndcg_at_5 = float(node_met.get("node_ml_ndcg_at_5", np.nan))
            total_sec = float(time.time() - model_t0)
            avg_epoch_sec = float(np.mean(epoch_secs)) if len(epoch_secs) > 0 else 0.0
            rows.append(
                {
                    "model": model_name,
                    "status": "ok",
                    "val_mrr": val_mrr,
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "test_mrr": test_mrr,
                    "test_auc": test_auc,
                    "test_ap": test_ap,
                    "node_ml_micro_f1": node_micro_f1,
                    "node_ml_macro_f1": node_macro_f1,
                    "node_ml_precision_at_1": node_precision_at_1,
                    "node_ml_precision_at_3": node_precision_at_3,
                    "node_ml_precision_at_5": node_precision_at_5,
                    "node_ml_recall_at_1": node_recall_at_1,
                    "node_ml_recall_at_3": node_recall_at_3,
                    "node_ml_recall_at_5": node_recall_at_5,
                    "node_ml_ndcg_at_1": node_ndcg_at_1,
                    "node_ml_ndcg_at_3": node_ndcg_at_3,
                    "node_ml_ndcg_at_5": node_ndcg_at_5,
                    "avg_epoch_sec": avg_epoch_sec,
                    "total_sec": total_sec,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "device": device,
                    "seed": args.seed,
                    "error_message": "",
                }
            )
            print(
                f"[{model_name}] test_mrr={test_mrr:.6f} test_auc={test_auc:.6f} test_ap={test_ap:.6f} "
                f"node_ml_micro_f1={node_micro_f1} params={total_params} avg_epoch_sec={avg_epoch_sec:.2f}"
            )
        except Exception as e:
            rows.append(
                {
                    "model": model_name,
                    "status": f"failed:{type(e).__name__}",
                    "val_mrr": np.nan,
                    "val_auc": np.nan,
                    "val_ap": np.nan,
                    "test_mrr": np.nan,
                    "test_auc": np.nan,
                    "test_ap": np.nan,
                    "node_ml_micro_f1": np.nan,
                    "node_ml_macro_f1": np.nan,
                    "node_ml_precision_at_1": np.nan,
                    "node_ml_precision_at_3": np.nan,
                    "node_ml_precision_at_5": np.nan,
                    "node_ml_recall_at_1": np.nan,
                    "node_ml_recall_at_3": np.nan,
                    "node_ml_recall_at_5": np.nan,
                    "node_ml_ndcg_at_1": np.nan,
                    "node_ml_ndcg_at_3": np.nan,
                    "node_ml_ndcg_at_5": np.nan,
                    "avg_epoch_sec": np.nan,
                    "total_sec": np.nan,
                    "total_params": np.nan,
                    "trainable_params": np.nan,
                    "device": device,
                    "seed": args.seed,
                    "error_message": str(e),
                }
            )
            print(f"[{model_name}] failed: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
