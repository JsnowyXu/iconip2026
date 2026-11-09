from typing import Dict, Iterable, Tuple

import numpy as np


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score)) + 1
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    denom = tp + fp
    precision = np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    positives = (y == 1).sum()
    if positives == 0:
        return 0.0
    ap = (precision * (y == 1)).sum() / positives
    return float(ap)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)
    y_pred = (y_score >= 0.5).astype(int)
    acc = float((y_pred == y_true).mean())
    return {
        "auc": _safe_auc(y_true, y_score),
        "ap": _safe_ap(y_true, y_score),
        "acc": acc,
    }


def topk_mrr(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    # pos_scores: [N], neg_scores: [N, K]
    pos_scores = pos_scores.reshape(-1, 1)
    ranks = 1 + (neg_scores >= pos_scores).sum(axis=1)
    return float((1.0 / ranks).mean())


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = float(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = float(np.logical_and(y_true == 1, y_pred == 0).sum())
    if tp <= 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp <= 0 and fp <= 0 and fn <= 0:
        return 0.0
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def multiclass_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0 or logits.size == 0:
        return {"acc": 0.0, "macro_f1": 0.0}
    y_true = y_true.astype(np.int64)
    y_pred = np.argmax(logits, axis=1).astype(np.int64)
    acc = float((y_pred == y_true).mean())

    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        yt = (y_true == c).astype(np.int64)
        yp = (y_pred == c).astype(np.int64)
        f1s.append(_f1_binary(yt, yp))
    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    return {"acc": acc, "macro_f1": macro_f1}


def _topk_indices(probs: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(k, probs.shape[1])))
    part = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(probs, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def _precision_recall_at_k(y_true: np.ndarray, probs: np.ndarray, k: int) -> Tuple[float, float]:
    topk = _topk_indices(probs, k)
    hits = np.take_along_axis(y_true, topk, axis=1).astype(np.float64)
    precision = float((hits.sum(axis=1) / float(k)).mean())

    positives = y_true.sum(axis=1).astype(np.float64)
    valid = positives > 0
    if valid.any():
        recall = float((hits.sum(axis=1)[valid] / positives[valid]).mean())
    else:
        recall = 0.0
    return precision, recall


def _ndcg_at_k(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
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
    if valid.any():
        return float((dcg[valid] / ideal[valid]).mean())
    return 0.0


def multilabel_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    if y_true.size == 0 or logits.size == 0:
        return {
            "auc": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "map": 0.0,
            "precision_at_1": 0.0,
            "precision_at_3": 0.0,
            "precision_at_5": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "ndcg_at_1": 0.0,
            "ndcg_at_3": 0.0,
            "ndcg_at_5": 0.0,
        }
    y_true = y_true.astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    y_pred = (probs >= threshold).astype(np.int64)

    # micro F1
    micro_f1 = _f1_binary(y_true.reshape(-1), y_pred.reshape(-1))

    # macro F1 and mAP over labels
    c = y_true.shape[1]
    f1s = []
    aucs = []
    aps = []
    for j in range(c):
        f1s.append(_f1_binary(y_true[:, j], y_pred[:, j]))
        aucs.append(_safe_auc(y_true[:, j], probs[:, j]))
        aps.append(_safe_ap(y_true[:, j], probs[:, j]))
    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    m_auc = float(np.mean(aucs)) if len(aucs) > 0 else 0.0
    m_ap = float(np.mean(aps)) if len(aps) > 0 else 0.0

    k_list = [1, 3, 5]
    out = {"auc": m_auc, "micro_f1": micro_f1, "macro_f1": macro_f1, "map": m_ap}
    for k in k_list:
        kk = min(int(k), int(probs.shape[1]))
        p_at_k, r_at_k = _precision_recall_at_k(y_true, probs, kk)
        out[f"precision_at_{k}"] = p_at_k
        out[f"recall_at_{k}"] = r_at_k
        out[f"ndcg_at_{k}"] = _ndcg_at_k(y_true, probs, kk)
    return out


def find_best_multilabel_threshold(
    y_true: np.ndarray,
    logits: np.ndarray,
    candidates: Iterable[float] = None,
) -> Tuple[float, float]:
    if y_true.size == 0 or logits.size == 0:
        return 0.5, 0.0
    if candidates is None:
        candidates = np.linspace(0.01, 0.99, 99)

    best_thr = 0.5
    best_f1 = -1.0
    for thr in candidates:
        cur = multilabel_metrics(y_true, logits, threshold=float(thr)).get("micro_f1", 0.0)
        if cur > best_f1:
            best_f1 = float(cur)
            best_thr = float(thr)
    return best_thr, max(0.0, best_f1)
