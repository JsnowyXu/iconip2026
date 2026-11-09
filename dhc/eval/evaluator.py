import time
from typing import Dict

import numpy as np
import torch

from dhc.data.batcher import TemporalBatcher, make_edge_batch
from dhc.utils.metrics import binary_metrics, multiclass_metrics, multilabel_metrics, topk_mrr


@torch.no_grad()
def replay_history(
    model,
    data,
    history_idx: torch.Tensor,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
    tag: str = "replay_history",
) -> None:
    # Reset before switching to eval mode. The underlying PyG TGN memory flushes
    # pending messages inside train(False), which can cause a large one-shot
    # allocation on long streams if we keep stale training-state buffers alive.
    model.reset_state()
    model.eval()
    total = int((history_idx.numel() + batch_size - 1) // batch_size) if history_idx.numel() > 0 else 0
    if verbose:
        print(f"[{tag}] start edges={int(history_idx.numel())} batches={total} batch_size={batch_size}")
    last_log = time.time()
    for step, idx in enumerate(TemporalBatcher(history_idx, batch_size=batch_size, shuffle=False), start=1):
        b = make_edge_batch(data, idx)
        neg = torch.randint(0, data.num_nodes, (b["src"].numel(),), device=device)
        out = model.forward_batch(b["src"], b["dst"], neg, b["rel"], b["ts"], b["edge_feat"])
        model.update_state(out["cache"])
        if verbose and (step == total or step == 1 or time.time() - last_log >= 30.0):
            print(f"[{tag}] progress batch={step}/{total}")
            last_log = time.time()
    if verbose:
        print(f"[{tag}] done")


@torch.no_grad()
def eval_link_prediction(
    model,
    data,
    history_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    batch_size: int = 512,
    num_neg: int = 20,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
    tag: str = "link_eval",
) -> Dict[str, float]:
    if eval_idx.numel() == 0:
        return {"auc": 0.5, "ap": 0.0, "acc": 0.0, "mrr": 0.0}

    # Replay history to build temporal memory.
    replay_history(model, data, history_idx=history_idx, batch_size=batch_size, device=device, verbose=verbose, tag=f"{tag}:replay")

    pos_all = []
    neg_all = []
    y_score = []
    y_true = []

    total = int((eval_idx.numel() + batch_size - 1) // batch_size) if eval_idx.numel() > 0 else 0
    if verbose:
        print(f"[{tag}] start edges={int(eval_idx.numel())} batches={total} num_neg={num_neg}")
    last_log = time.time()
    for step, idx in enumerate(TemporalBatcher(eval_idx, batch_size=batch_size, shuffle=False), start=1):
        b = make_edge_batch(data, idx)
        src = b["src"]
        dst = b["dst"]
        rel = b["rel"]
        ts = b["ts"]
        edge_label = b["label"].float()
        supervised = edge_label > 0.5

        if supervised.any():
            src_sup = src[supervised]
            dst_sup = dst[supervised]
            rel_sup = rel[supervised]
            ts_sup = ts[supervised]

            pos_logit = model.predict_links(src_sup, dst_sup, rel_sup, ts_sup)
            pos_score = torch.sigmoid(pos_logit).detach().cpu().numpy()

            neg_scores = []
            for _ in range(num_neg):
                neg_dst = torch.randint(0, data.num_nodes, (src_sup.numel(),), device=device)
                neg_logit = model.predict_links(src_sup, neg_dst, rel_sup, ts_sup)
                neg_scores.append(torch.sigmoid(neg_logit).detach().cpu().numpy())

            neg_scores = np.stack(neg_scores, axis=1)
            pos_all.append(pos_score)
            neg_all.append(neg_scores)

            y_score.append(pos_score)
            y_true.append(np.ones_like(pos_score))
            y_score.append(neg_scores.reshape(-1))
            y_true.append(np.zeros(neg_scores.size))

        # Update memory with ground-truth positives.
        filler_neg = torch.randint(0, data.num_nodes, (src.numel(),), device=device)
        out = model.forward_batch(src, dst, filler_neg, rel, ts, b["edge_feat"])
        model.update_state(out["cache"])
        if verbose and (step == total or step == 1 or time.time() - last_log >= 30.0):
            print(f"[{tag}] progress batch={step}/{total}")
            last_log = time.time()

    if not pos_all:
        return {"auc": 0.5, "ap": 0.0, "acc": 0.0, "mrr": 0.0}

    pos_all = np.concatenate(pos_all, axis=0)
    neg_all = np.concatenate(neg_all, axis=0)
    y_score = np.concatenate(y_score, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    basic = binary_metrics(y_true, y_score)
    mrr = topk_mrr(pos_all, neg_all)
    basic["mrr"] = mrr
    if verbose:
        print(f"[{tag}] done auc={basic['auc']:.6f} ap={basic['ap']:.6f} mrr={basic['mrr']:.6f}")
    return basic


@torch.no_grad()
def eval_node_tasks(
    model,
    data,
    history_idx: torch.Tensor,
    batch_size: int = 4096,
    device: torch.device = torch.device("cpu"),
    target_node_ids: torch.Tensor = None,
    multilabel_threshold: float = 0.5,
    verbose: bool = False,
    tag: str = "node_eval",
) -> Dict[str, float]:
    replay_history(
        model,
        data,
        history_idx=history_idx,
        batch_size=min(1024, batch_size),
        device=device,
        verbose=verbose,
        tag=f"{tag}:replay",
    )

    out_metrics: Dict[str, float] = {}
    if target_node_ids is None:
        target_node_ids = torch.arange(data.num_nodes, device=device)
    target_node_ids = target_node_ids.long().to(device)
    if target_node_ids.numel() == 0:
        return out_metrics

    pred_mc_logits = []
    pred_ml_logits = []
    pred_nodes = []
    total = int((target_node_ids.numel() + batch_size - 1) // batch_size) if target_node_ids.numel() > 0 else 0
    if verbose:
        print(f"[{tag}] start nodes={int(target_node_ids.numel())} batches={total} batch_size={batch_size}")
    last_log = time.time()
    for step, idx in enumerate(TemporalBatcher(target_node_ids, batch_size=batch_size, shuffle=False), start=1):
        heads = model.predict_node_heads(idx)
        if heads["multiclass"] is not None:
            pred_mc_logits.append(heads["multiclass"].detach().cpu().numpy())
        if heads["multilabel"] is not None:
            pred_ml_logits.append(heads["multilabel"].detach().cpu().numpy())
        pred_nodes.append(idx.detach().cpu().numpy())
        if verbose and (step == total or step == 1 or time.time() - last_log >= 30.0):
            print(f"[{tag}] progress batch={step}/{total}")
            last_log = time.time()

    if len(pred_nodes) == 0:
        return out_metrics
    pred_nodes = np.concatenate(pred_nodes, axis=0).astype(np.int64)

    pred_map = {nid: i for i, nid in enumerate(pred_nodes.tolist())}

    if data.node_multiclass is not None and len(pred_mc_logits) > 0:
        mc_ids = data.node_multiclass["node_ids"].detach().cpu().numpy().astype(np.int64)
        mc_y = data.node_multiclass["labels"].detach().cpu().numpy().astype(np.int64)
        mc_logits_full = np.concatenate(pred_mc_logits, axis=0)
        sel = []
        keep = []
        for i, n in enumerate(mc_ids.tolist()):
            if n in pred_map:
                sel.append(pred_map[n])
                keep.append(i)
        if len(sel) > 0:
            mc_logits = mc_logits_full[sel]
            mc_true = mc_y[np.asarray(keep, dtype=np.int64)]
            mm = multiclass_metrics(mc_true, mc_logits)
            out_metrics.update({f"node_mc_{k}": float(v) for k, v in mm.items()})

    if data.node_multilabel is not None and len(pred_ml_logits) > 0:
        ml_ids = data.node_multilabel["node_ids"].detach().cpu().numpy().astype(np.int64)
        ml_y = data.node_multilabel["labels"].detach().cpu().numpy().astype(np.float32)
        ml_logits_full = np.concatenate(pred_ml_logits, axis=0)
        sel = []
        keep = []
        for i, n in enumerate(ml_ids.tolist()):
            if n in pred_map:
                sel.append(pred_map[n])
                keep.append(i)
        if len(sel) > 0:
            ml_logits = ml_logits_full[sel]
            ml_true = ml_y[np.asarray(keep, dtype=np.int64)]
            mm = multilabel_metrics(ml_true, ml_logits, threshold=float(multilabel_threshold))
            out_metrics.update({f"node_ml_{k}": float(v) for k, v in mm.items()})
            out_metrics["node_ml_threshold"] = float(multilabel_threshold)

    if verbose:
        print(f"[{tag}] done keys={sorted(out_metrics.keys())}")
    return out_metrics
