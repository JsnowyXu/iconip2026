#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency guard
    XGBClassifier = None


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


def split_supervised(obj: Dict, split_name: str) -> np.ndarray:
    idx = obj["split"][split_name].detach().cpu().numpy().astype(np.int64)
    labels = obj["edges"]["edge_label"].detach().cpu().numpy().astype(np.float32)
    return idx[labels[idx] > 0.5]


def positive_history_idx(obj: Dict, *split_names: str) -> np.ndarray:
    parts = [split_supervised(obj, split_name) for split_name in split_names]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts, axis=0).astype(np.int64)


def safe_norm_rows(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1).clip(min=1e-12)


class HistoryFeatures:
    def __init__(self, obj: Dict, history_idx: np.ndarray, use_common_neighbors: bool = False):
        edges = obj["edges"]
        self.src = edges["src"].detach().cpu().numpy().astype(np.int64)
        self.dst = edges["dst"].detach().cpu().numpy().astype(np.int64)
        self.ts = edges["ts"].detach().cpu().numpy().astype(np.float64)
        self.rel = edges["rel"].detach().cpu().numpy().astype(np.int64)
        self.node_feat = obj["node_feat"].detach().cpu().numpy().astype(np.float32)
        self.node_type = obj.get("node_type", torch.zeros(int(obj["num_nodes"]), dtype=torch.long)).detach().cpu().numpy().astype(np.int64)
        self.num_nodes = int(obj["num_nodes"])
        self.num_rels = int(obj.get("num_rels", max(1, int(self.rel.max()) + 1 if self.rel.size else 1)))
        self.hist_idx = np.asarray(history_idx, dtype=np.int64)
        self.max_ts = float(self.ts[self.hist_idx].max()) if self.hist_idx.size else 0.0
        self.use_common_neighbors = bool(use_common_neighbors)

        self.src_deg = np.zeros(self.num_nodes, dtype=np.float32)
        self.dst_deg = np.zeros(self.num_nodes, dtype=np.float32)
        self.node_deg = np.zeros(self.num_nodes, dtype=np.float32)
        self.rel_count = np.zeros(max(1, self.num_rels), dtype=np.float32)
        self.pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
        self.pair_rel_count: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.last_pair_ts: Dict[Tuple[int, int], float] = {}
        self.last_src_ts: Dict[int, float] = {}
        self.last_dst_ts: Dict[int, float] = {}
        self.neigh: Dict[int, set] = defaultdict(set)

        for i in self.hist_idx.tolist():
            s = int(self.src[i])
            d = int(self.dst[i])
            r = int(self.rel[i])
            t = float(self.ts[i])
            self.src_deg[s] += 1.0
            self.dst_deg[d] += 1.0
            self.node_deg[s] += 1.0
            self.node_deg[d] += 1.0
            if 0 <= r < self.rel_count.shape[0]:
                self.rel_count[r] += 1.0
            self.pair_count[(s, d)] += 1
            self.pair_rel_count[(s, d, r)] += 1
            self.last_pair_ts[(s, d)] = max(t, self.last_pair_ts.get((s, d), -np.inf))
            self.last_src_ts[s] = max(t, self.last_src_ts.get(s, -np.inf))
            self.last_dst_ts[d] = max(t, self.last_dst_ts.get(d, -np.inf))
            if self.use_common_neighbors:
                self.neigh[s].add(d)
                self.neigh[d].add(s)

        self.node_feat_norm = safe_norm_rows(self.node_feat)
        self.rel_total = float(max(1.0, float(self.rel_count.sum())))

    def _common_neighbors(self, s: int, d: int) -> int:
        a = self.neigh.get(s)
        b = self.neigh.get(d)
        if not a or not b:
            return 0
        if len(a) > len(b):
            a, b = b, a
        return sum(1 for x in a if x in b)

    def transform(self, src: np.ndarray, dst: np.ndarray, rel: np.ndarray, ts: np.ndarray) -> np.ndarray:
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        rel = np.asarray(rel, dtype=np.int64)
        ts = np.asarray(ts, dtype=np.float64)
        n = src.shape[0]
        out = np.zeros((n, 18), dtype=np.float32)
        sf = self.node_feat[src]
        df = self.node_feat[dst]
        dot = (sf * df).sum(axis=1)
        cos = dot / (self.node_feat_norm[src] * self.node_feat_norm[dst])
        l2 = np.linalg.norm(sf - df, axis=1)
        for i, (s, d, r, t) in enumerate(zip(src.tolist(), dst.tolist(), rel.tolist(), ts.tolist())):
            pair = (int(s), int(d))
            lp = self.last_pair_ts.get(pair, -np.inf)
            ls = self.last_src_ts.get(int(s), -np.inf)
            ld = self.last_dst_ts.get(int(d), -np.inf)
            out[i, 0] = float(r) / max(1.0, float(self.num_rels - 1))
            out[i, 1] = np.log1p(max(0.0, t - self.max_ts))
            out[i, 2] = np.log1p(self.pair_count.get(pair, 0))
            out[i, 3] = np.log1p(self.pair_rel_count.get((int(s), int(d), int(r)), 0))
            out[i, 4] = np.log1p(self.src_deg[int(s)])
            out[i, 5] = np.log1p(self.dst_deg[int(d)])
            out[i, 6] = np.log1p(self.node_deg[int(s)])
            out[i, 7] = np.log1p(self.node_deg[int(d)])
            out[i, 8] = np.log1p(self.rel_count[int(r)] if 0 <= int(r) < self.rel_count.shape[0] else 0.0)
            out[i, 9] = np.log1p(self._common_neighbors(int(s), int(d))) if self.use_common_neighbors else 0.0
            out[i, 10] = 1.0 if self.node_type[int(s)] == self.node_type[int(d)] else 0.0
            out[i, 11] = cos[i]
            out[i, 12] = dot[i]
            out[i, 13] = l2[i]
            out[i, 14] = np.log1p(max(0.0, t - lp)) if np.isfinite(lp) else 0.0
            out[i, 15] = np.log1p(max(0.0, t - ls)) if np.isfinite(ls) else 0.0
            out[i, 16] = np.log1p(max(0.0, t - ld)) if np.isfinite(ld) else 0.0
            out[i, 17] = float(self.rel_count[int(r)] / self.rel_total) if 0 <= int(r) < self.rel_count.shape[0] else 0.0
        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def count_score(self, src: np.ndarray, dst: np.ndarray, rel: np.ndarray, ts: np.ndarray) -> np.ndarray:
        score = np.zeros(len(src), dtype=np.float32)
        for i, (s, d, r, t) in enumerate(zip(src.tolist(), dst.tolist(), rel.tolist(), ts.tolist())):
            pair = (int(s), int(d))
            c_pair = float(self.pair_count.get(pair, 0))
            c_rel = float(self.pair_rel_count.get((int(s), int(d), int(r)), 0))
            c_src = float(self.src_deg[int(s)])
            c_dst = float(self.dst_deg[int(d)])
            lp = self.last_pair_ts.get(pair, -np.inf)
            rec = 0.0 if not np.isfinite(lp) else 1.0 / (1.0 + max(0.0, float(t) - float(lp)))
            score[i] = np.log1p(c_pair) + 0.5 * np.log1p(c_rel) + 0.1 * np.log1p(c_src * c_dst) + rec
        return score


def make_training_matrix(obj: Dict, train_pos: np.ndarray, rng: np.random.RandomState, max_train_pos: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = obj["edges"]
    src = edges["src"].detach().cpu().numpy().astype(np.int64)
    dst = edges["dst"].detach().cpu().numpy().astype(np.int64)
    rel = edges["rel"].detach().cpu().numpy().astype(np.int64)
    ts = edges["ts"].detach().cpu().numpy().astype(np.float64)
    if max_train_pos > 0 and train_pos.size > max_train_pos:
        keep = rng.choice(train_pos, size=max_train_pos, replace=False)
        train_pos = np.sort(keep)
    pos_src = src[train_pos]
    pos_dst = dst[train_pos]
    pos_rel = rel[train_pos]
    pos_ts = ts[train_pos]
    neg_dst = rng.randint(0, int(obj["num_nodes"]), size=pos_src.shape[0])
    same = neg_dst == pos_dst
    neg_dst[same] = (neg_dst[same] + 1) % int(obj["num_nodes"])
    x_src = np.concatenate([pos_src, pos_src])
    x_dst = np.concatenate([pos_dst, neg_dst])
    x_rel = np.concatenate([pos_rel, pos_rel])
    x_ts = np.concatenate([pos_ts, pos_ts])
    y = np.concatenate([np.ones(pos_src.shape[0]), np.zeros(pos_src.shape[0])]).astype(np.int64)
    return x_src, x_dst, x_rel, x_ts, y


def build_models(seed: int, use_gpu: bool) -> List[Tuple[str, object, str]]:
    models: List[Tuple[str, object, str]] = []
    if XGBClassifier is not None:
        xgb_kwargs = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": 4,
            "tree_method": "hist",
        }
        if use_gpu:
            xgb_kwargs["device"] = "cuda"
        models.append(("XGBoost", XGBClassifier(**xgb_kwargs), "cuda" if use_gpu else "cpu"))
    models.append(
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=240, max_depth=18, min_samples_leaf=2, random_state=seed, n_jobs=4),
            "cpu",
        )
    )
    models.append(
        (
            "LogisticRegression",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            "cpu",
        )
    )
    return models


def predict_proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(np.float64)
    return model.decision_function(x).astype(np.float64)


def evaluate_model(
    model_name: str,
    model,
    feat: HistoryFeatures,
    obj: Dict,
    eval_pos: np.ndarray,
    eval_num_negs: int,
    rng: np.random.RandomState,
    max_eval_pos: int = 0,
):
    edges = obj["edges"]
    src_all = edges["src"].detach().cpu().numpy().astype(np.int64)
    dst_all = edges["dst"].detach().cpu().numpy().astype(np.int64)
    rel_all = edges["rel"].detach().cpu().numpy().astype(np.int64)
    ts_all = edges["ts"].detach().cpu().numpy().astype(np.float64)
    eval_pos = np.asarray(eval_pos, dtype=np.int64)
    if max_eval_pos > 0 and eval_pos.size > max_eval_pos:
        eval_pos = np.sort(rng.choice(eval_pos, size=int(max_eval_pos), replace=False))
    pos_src = src_all[eval_pos]
    pos_dst = dst_all[eval_pos]
    pos_rel = rel_all[eval_pos]
    pos_ts = ts_all[eval_pos]
    n = pos_src.shape[0]
    if n == 0:
        return 0.0, 0.5, 0.0
    neg_dst = rng.randint(0, int(obj["num_nodes"]), size=(n, eval_num_negs))
    same = neg_dst == pos_dst.reshape(-1, 1)
    neg_dst[same] = (neg_dst[same] + 1) % int(obj["num_nodes"])

    if model_name == "TemporalCount":
        pos_score = feat.count_score(pos_src, pos_dst, pos_rel, pos_ts)
        neg_score = feat.count_score(
            np.repeat(pos_src, eval_num_negs),
            neg_dst.reshape(-1),
            np.repeat(pos_rel, eval_num_negs),
            np.repeat(pos_ts, eval_num_negs),
        )
    else:
        pos_x = feat.transform(pos_src, pos_dst, pos_rel, pos_ts)
        neg_x = feat.transform(
            np.repeat(pos_src, eval_num_negs),
            neg_dst.reshape(-1),
            np.repeat(pos_rel, eval_num_negs),
            np.repeat(pos_ts, eval_num_negs),
        )
        pos_score = predict_proba(model, pos_x)
        neg_score = predict_proba(model, neg_x)
    neg_score = neg_score.reshape(n, eval_num_negs)
    y_true = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n * eval_num_negs, dtype=np.int64)])
    y_score = np.concatenate([pos_score.reshape(-1), neg_score.reshape(-1)])
    auc, ap = auc_ap(y_true, y_score)
    mrr = mrr_from_pos_neg(pos_score, neg_score)
    return mrr, auc, ap


def append_rows(rows: List[Dict], model_name: str, split: str, metrics, total_sec: float, device: str, args):
    mrr, auc, ap = metrics
    rows.append(
        {
            "split": split,
            "model": model_name,
            "mrr": float(mrr),
            "auc": float(auc),
            "ap": float(ap),
            "avg_epoch_sec": np.nan,
            "total_sec": float(total_sec),
            "total_params": 0,
            "trainable_params": 0,
            "device": device,
            "eval_num_negs": int(args.eval_num_negs),
            "max_train_pos": int(args.max_train_pos),
            "max_eval_pos": int(args.max_eval_pos),
            "use_common_neighbors": bool(args.use_common_neighbors),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-num-negs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-pos", type=int, default=200000)
    ap.add_argument("--max-eval-pos", type=int, default=0)
    ap.add_argument("--use-common-neighbors", action="store_true")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    obj = torch.load(args.data, map_location="cpu")
    train_pos = split_supervised(obj, "train")
    val_pos = split_supervised(obj, "val")
    test_pos = split_supervised(obj, "test")
    train_hist = positive_history_idx(obj, "train")
    hist_test = positive_history_idx(obj, "train", "val")
    use_gpu = bool(args.gpu_id >= 0 and torch.cuda.is_available())
    xgb_device = f"cuda:{args.gpu_id}" if use_gpu else "cpu"

    train_feat = HistoryFeatures(obj, train_hist, use_common_neighbors=args.use_common_neighbors)
    x_src, x_dst, x_rel, x_ts, y = make_training_matrix(obj, train_pos, rng, int(args.max_train_pos))
    x_train = train_feat.transform(x_src, x_dst, x_rel, x_ts)
    rows: List[Dict] = []

    for model_name, model, model_device in build_models(seed=args.seed, use_gpu=use_gpu):
        t0 = time.time()
        try:
            model.fit(x_train, y)
            val_metrics = evaluate_model(
                model_name,
                model,
                train_feat,
                obj,
                val_pos,
                args.eval_num_negs,
                np.random.RandomState(args.seed + 1001),
                args.max_eval_pos,
            )
            test_feat = HistoryFeatures(obj, hist_test, use_common_neighbors=args.use_common_neighbors)
            test_metrics = evaluate_model(
                model_name,
                model,
                test_feat,
                obj,
                test_pos,
                args.eval_num_negs,
                np.random.RandomState(args.seed + 2001),
                args.max_eval_pos,
            )
            total_sec = time.time() - t0
            display_device = xgb_device if model_device == "cuda" else "cpu"
            append_rows(rows, model_name, "val", val_metrics, total_sec, display_device, args)
            append_rows(rows, model_name, "test", test_metrics, total_sec, display_device, args)
            print(f"[{model_name}] val_mrr={val_metrics[0]:.6f} test_mrr={test_metrics[0]:.6f} sec={total_sec:.1f}")
        except Exception as e:
            print(f"[{model_name}] failed: {type(e).__name__}: {e}")

    t0 = time.time()
    val_metrics = evaluate_model(
        "TemporalCount",
        None,
        train_feat,
        obj,
        val_pos,
        args.eval_num_negs,
        np.random.RandomState(args.seed + 1001),
        args.max_eval_pos,
    )
    test_feat = HistoryFeatures(obj, hist_test, use_common_neighbors=args.use_common_neighbors)
    test_metrics = evaluate_model(
        "TemporalCount",
        None,
        test_feat,
        obj,
        test_pos,
        args.eval_num_negs,
        np.random.RandomState(args.seed + 2001),
        args.max_eval_pos,
    )
    total_sec = time.time() - t0
    append_rows(rows, "TemporalCount", "val", val_metrics, total_sec, "cpu", args)
    append_rows(rows, "TemporalCount", "test", test_metrics, total_sec, "cpu", args)
    print(f"[TemporalCount] val_mrr={val_metrics[0]:.6f} test_mrr={test_metrics[0]:.6f} sec={total_sec:.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
