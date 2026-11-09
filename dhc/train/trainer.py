from pathlib import Path
from typing import Dict, Optional
import time

import numpy as np
import torch
import torch.nn.functional as F

from dhc.data.batcher import TemporalBatcher, make_edge_batch
from dhc.eval.evaluator import eval_link_prediction, eval_node_tasks, replay_history
from dhc.utils.metrics import find_best_multilabel_threshold
from dhc.utils.io import save_json


class Trainer:
    def __init__(
        self,
        model,
        data,
        device: torch.device,
        task_type: str = "edge_transductive",
        lr: float = 1e-3,
        batch_size: int = 512,
        num_epochs: int = 10,
        lambda_cl: float = 0.1,
        lambda_community_entropy: float = 0.0,
        lambda_mc: float = 1.0,
        lambda_ml: float = 1.0,
        relation_loss_weights: str = "",
        ml_use_pos_weight: bool = False,
        ml_pos_weight_max: float = 50.0,
        ml_pos_weight_power: float = 0.5,
        ml_threshold_search: bool = False,
        ml_threshold_num_candidates: int = 99,
        ml_calib_max_nodes: int = 0,
        node_eval_max_nodes: int = 0,
        typed_negative_ratio: float = 0.6,
        popular_negative_ratio: float = 0.2,
        save_memory_artifacts: bool = False,
        lightweight_checkpoint: bool = True,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 0.0,
        monitor_metric: str = "val_mrr",
        verbose: bool = True,
        out_dir: str = "results/default",
    ):
        self.model = model
        self.data = data
        self.device = device
        self.task_type = str(task_type or "edge_transductive").strip()
        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.lambda_cl = float(lambda_cl)
        self.lambda_community_entropy = float(max(0.0, lambda_community_entropy))
        self.lambda_mc = float(lambda_mc)
        self.lambda_ml = float(lambda_ml)
        self.relation_loss_weights = str(relation_loss_weights or "").strip()
        self.ml_use_pos_weight = bool(ml_use_pos_weight)
        self.ml_pos_weight_max = float(max(1.0, ml_pos_weight_max))
        self.ml_pos_weight_power = float(min(max(ml_pos_weight_power, 0.0), 1.0))
        self.ml_threshold_search = bool(ml_threshold_search)
        self.ml_threshold_num_candidates = int(max(3, ml_threshold_num_candidates))
        self.ml_calib_max_nodes = int(max(0, ml_calib_max_nodes))
        self.node_eval_max_nodes = int(max(0, node_eval_max_nodes))
        self.typed_negative_ratio = float(min(max(typed_negative_ratio, 0.0), 1.0))
        self.popular_negative_ratio = float(min(max(popular_negative_ratio, 0.0), 1.0))
        self.save_memory_artifacts = bool(save_memory_artifacts)
        self.lightweight_checkpoint = bool(lightweight_checkpoint)
        self.early_stop_patience = int(max(0, early_stop_patience))
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.monitor_metric = str(monitor_metric).strip() or "auto"
        self.verbose = bool(verbose)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.memory_snapshots = []

        self.mc_table = None
        self.ml_table = None
        self.mc_train_table = None
        self.ml_train_table = None
        self.ml_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.rel_name_map = self._build_relation_name_map()
        self.rel_loss_weight = self._build_relation_loss_weight()
        self.monitor_node_ids = self._build_monitor_nodes(max_nodes=96)
        self.node_type_candidates = self._build_node_type_candidates()
        self.popular_negative_nodes, self.popular_negative_weights = self._build_popular_negative_pool()
        self.node_eval_splits = self._build_node_eval_splits()

        if data.node_multiclass is not None:
            num_nodes = data.num_nodes
            table = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            node_ids = data.node_multiclass["node_ids"].long().to(device)
            labels = data.node_multiclass["labels"].long().to(device)
            table[node_ids] = labels
            self.mc_table = table
            self.mc_train_table = self._mask_mc_table(table, self.node_eval_splits["train"])

        if data.node_multilabel is not None:
            num_nodes = data.num_nodes
            c = data.node_multilabel["labels"].shape[1]
            table = torch.full((num_nodes, c), -1.0, dtype=torch.float, device=device)
            node_ids = data.node_multilabel["node_ids"].long().to(device)
            labels = data.node_multilabel["labels"].float().to(device)
            table[node_ids] = labels
            self.ml_table = table
            if self.ml_use_pos_weight:
                with torch.no_grad():
                    pos = labels.sum(dim=0)
                    neg = labels.shape[0] - pos
                    pos_weight = neg / torch.clamp(pos, min=1.0)
                    pos_weight = torch.clamp(pos_weight, min=1.0, max=self.ml_pos_weight_max)
                    pos_weight = torch.pow(pos_weight, self.ml_pos_weight_power)
                self.ml_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.ml_train_table = self._mask_ml_table(table, self.node_eval_splits["train"])

        self.monitor_metric = self._resolve_monitor_metric(self.monitor_metric)

    def _skip_aux_eval(self) -> bool:
        name = self.out_dir.name.lower()
        return name in {"dhc_trans_no_rel", "dhc_trans_no_comm"}

    def _build_relation_name_map(self) -> Dict[int, str]:
        out = {}
        extra = self.data.meta.get("extra", {}) if isinstance(self.data.meta, dict) else {}
        rel_map = extra.get("relation_map", {})
        if isinstance(rel_map, dict):
            for k, v in rel_map.items():
                try:
                    out[int(v)] = str(k)
                except Exception:
                    continue
        return out

    def _is_node_task(self) -> bool:
        return self.task_type.startswith("node_")

    def _is_edge_task(self) -> bool:
        return self.task_type.startswith("edge_")

    def _resolve_monitor_metric(self, metric: str) -> str:
        metric = str(metric or "auto").strip()
        if metric != "auto":
            return metric
        if self._is_node_task():
            if self.data.node_multilabel is not None:
                return "val_node_ml_micro_f1"
            if self.data.node_multiclass is not None:
                return "val_node_mc_acc"
        return "val_ap"

    def _build_node_eval_splits(self) -> Dict[str, torch.Tensor]:
        candidate = []
        if self.data.node_multiclass is not None:
            candidate.append(self.data.node_multiclass["node_ids"].long())
        if self.data.node_multilabel is not None:
            candidate.append(self.data.node_multilabel["node_ids"].long())
        if len(candidate) == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return {"train": empty, "val": empty, "test": empty}

        cand = torch.cat(candidate, dim=0).unique().long().to(self.device)
        train_nodes = torch.cat([self.data.src[self.data.train_idx], self.data.dst[self.data.train_idx]], dim=0).unique()
        val_nodes = torch.cat([self.data.src[self.data.val_idx], self.data.dst[self.data.val_idx]], dim=0).unique()
        test_nodes = torch.cat([self.data.src[self.data.test_idx], self.data.dst[self.data.test_idx]], dim=0).unique()

        train_mask = torch.isin(cand, train_nodes)
        val_mask = (~train_mask) & torch.isin(cand, val_nodes)
        test_mask = (~train_mask) & (~val_mask) & torch.isin(cand, test_nodes)

        if int(val_mask.sum().item()) == 0 or int(test_mask.sum().item()) == 0:
            src = self.data.src.detach().cpu().numpy()
            dst = self.data.dst.detach().cpu().numpy()
            ts = self.data.ts.detach().cpu().numpy()
            first_seen = np.full(self.data.num_nodes, np.inf, dtype=np.float64)
            for s, d, t in zip(src, dst, ts):
                if t < first_seen[int(s)]:
                    first_seen[int(s)] = float(t)
                if t < first_seen[int(d)]:
                    first_seen[int(d)] = float(t)
            cand_np = cand.detach().cpu().numpy().astype(np.int64)
            seen_ts = first_seen[cand_np]
            valid = np.isfinite(seen_ts)
            if valid.any():
                t1 = np.quantile(seen_ts[valid], 0.70)
                t2 = np.quantile(seen_ts[valid], 0.85)
                train_mask = torch.from_numpy(valid & (seen_ts <= t1)).to(cand.device)
                val_mask = torch.from_numpy(valid & (seen_ts > t1) & (seen_ts <= t2)).to(cand.device)
                test_mask = torch.from_numpy(valid & (seen_ts > t2)).to(cand.device)

        train_ids = cand[train_mask].long().to(self.device)
        val_ids = cand[val_mask].long().to(self.device)
        test_ids = cand[test_mask].long().to(self.device)
        if val_ids.numel() == 0:
            val_ids = train_ids[: min(train_ids.numel(), max(1, train_ids.numel() // 5))]
        if test_ids.numel() == 0:
            start = min(train_ids.numel(), max(0, train_ids.numel() - max(1, train_ids.numel() // 5)))
            test_ids = train_ids[start:]
        return {"train": train_ids, "val": val_ids, "test": test_ids}

    def _mask_mc_table(self, table: torch.Tensor, allowed_node_ids: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(table, -1)
        if allowed_node_ids is None or allowed_node_ids.numel() == 0:
            return out
        out[allowed_node_ids.long()] = table[allowed_node_ids.long()]
        return out

    def _mask_ml_table(self, table: torch.Tensor, allowed_node_ids: torch.Tensor) -> torch.Tensor:
        out = torch.full_like(table, -1.0)
        if allowed_node_ids is None or allowed_node_ids.numel() == 0:
            return out
        out[allowed_node_ids.long()] = table[allowed_node_ids.long()]
        return out

    def _build_relation_loss_weight(self) -> torch.Tensor:
        weight = torch.ones(int(self.data.num_rels), dtype=torch.float, device=self.device)
        spec = self.relation_loss_weights
        if not spec:
            return weight
        rev_name_map = {str(v): int(k) for k, v in self.rel_name_map.items()}
        for item in spec.split(","):
            tok = item.strip()
            if not tok or "=" not in tok:
                continue
            key, val = tok.split("=", 1)
            key = key.strip()
            try:
                w = float(val.strip())
            except Exception:
                continue
            if w <= 0:
                continue
            rid = None
            try:
                rid = int(key)
            except Exception:
                rid = rev_name_map.get(key)
            if rid is None:
                continue
            if 0 <= int(rid) < int(self.data.num_rels):
                weight[int(rid)] = float(w)
        return weight

    def _build_monitor_nodes(self, max_nodes: int = 96) -> torch.Tensor:
        extra = self.data.meta.get("extra", {}) if isinstance(self.data.meta, dict) else {}
        picks = []
        if isinstance(extra, dict):
            patent = extra.get("patent_attributes", {}).get("node_ids", [])
            assignee = extra.get("assignee_node_ids", [])
            ipc = extra.get("ipc_attributes", {}).get("node_ids", [])
            for arr in [patent, assignee, ipc]:
                if isinstance(arr, list) and len(arr) > 0:
                    picks.extend(arr[: max(8, max_nodes // 3)])
        if len(picks) == 0:
            picks = list(range(min(self.data.num_nodes, max_nodes)))
        picks = np.array(sorted(set(int(x) for x in picks if 0 <= int(x) < self.data.num_nodes)), dtype=np.int64)
        if picks.size > max_nodes:
            picks = picks[:max_nodes]
        return torch.from_numpy(picks).long().to(self.device)

    def _checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        state = self.model.state_dict()
        if not self.lightweight_checkpoint:
            return state
        drop_prefixes = [
            "node_feat",
            "node_type",
            "memory.node_memory",
            "memory.node_last_update",
            "memory.rel_memory",
            "memory.rel_last_update",
            "recent_neighbors.",
            "community.community_memory",
            "community.community_last_update",
            "tgn_memory.memory",
            "tgn_memory.last_update",
            "tgn_memory._assoc",
        ]
        out = {}
        for k, v in state.items():
            if any(k == p or k.startswith(p + ".") for p in drop_prefixes):
                continue
            out[k] = v
        return out

    def _build_node_type_candidates(self) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        node_type = self.data.node_type.long()
        for typ in torch.unique(node_type).detach().cpu().tolist():
            ids = torch.where(node_type == int(typ))[0].long()
            if ids.numel() > 0:
                out[int(typ)] = ids.to(self.device)
        return out

    def _build_popular_negative_pool(self):
        train_idx = self.data.train_idx.long()
        if train_idx.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            empty_w = torch.empty(0, dtype=torch.float, device=self.device)
            return empty, empty_w
        train_label = self.data.edge_label[train_idx] > 0.5
        use_idx = train_idx[train_label]
        if use_idx.numel() == 0:
            use_idx = train_idx
        dst = self.data.dst[use_idx].long()
        cnt = torch.bincount(dst, minlength=self.data.num_nodes).float()
        keep = torch.where(cnt > 0)[0].long()
        if keep.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            empty_w = torch.empty(0, dtype=torch.float, device=self.device)
            return empty, empty_w
        topk = min(int(keep.numel()), 4096)
        weights, perm = cnt[keep].topk(topk)
        nodes = keep[perm].to(self.device)
        weights = weights.to(self.device)
        weights = weights / weights.sum().clamp_min(1e-12)
        return nodes, weights

    def _sample_negative_dst(self, dst: torch.Tensor) -> torch.Tensor:
        neg = torch.randint(0, self.data.num_nodes, (dst.numel(),), device=self.device)
        if dst.numel() == 0:
            return neg

        chooser = torch.rand(dst.numel(), device=self.device)
        typed_mask = chooser < self.typed_negative_ratio
        popular_mask = (chooser >= self.typed_negative_ratio) & (
            chooser < self.typed_negative_ratio + self.popular_negative_ratio
        )

        if typed_mask.any():
            pos_type = self.data.node_type[dst[typed_mask]].long()
            typed_index = torch.where(typed_mask)[0]
            for typ in torch.unique(pos_type).detach().cpu().tolist():
                sub_local = torch.where(pos_type == int(typ))[0]
                if sub_local.numel() == 0:
                    continue
                cand = self.node_type_candidates.get(int(typ))
                if cand is None or cand.numel() == 0:
                    continue
                pick = cand[torch.randint(0, cand.numel(), (sub_local.numel(),), device=self.device)]
                neg[typed_index[sub_local]] = pick

        if popular_mask.any() and self.popular_negative_nodes.numel() > 0:
            pick_idx = torch.multinomial(
                self.popular_negative_weights,
                int(popular_mask.sum().item()),
                replacement=True,
            )
            neg[popular_mask] = self.popular_negative_nodes[pick_idx]

        same = neg == dst
        if same.any():
            neg[same] = torch.randint(0, self.data.num_nodes, (int(same.sum().item()),), device=self.device)
            same = neg == dst
            if same.any():
                neg[same] = (dst[same] + 1) % self.data.num_nodes
        return neg

    @torch.no_grad()
    def _snapshot_memory(self, epoch: int) -> None:
        snap = {"epoch": int(epoch)}
        if hasattr(self.model, "memory"):
            node_mem = self.model.memory.node_memory[self.monitor_node_ids].detach().cpu().numpy().astype(np.float32)
            rel_mem = self.model.memory.rel_memory.detach().cpu().numpy().astype(np.float32)
            snap["monitor_node_ids"] = self.monitor_node_ids.detach().cpu().numpy().astype(np.int64)
            snap["node_memory"] = node_mem
            snap["rel_memory"] = rel_mem
        if hasattr(self.model, "community") and hasattr(self.model.community, "community_memory"):
            snap["community_memory"] = self.model.community.community_memory.detach().cpu().numpy().astype(np.float32)
        self.memory_snapshots.append(snap)

    def _eval_relation_breakdown(
        self,
        history_idx: torch.Tensor,
        eval_idx: torch.Tensor,
        num_neg: int = 20,
    ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        if eval_idx.numel() == 0:
            return out
        edge_label = self.data.edge_label[eval_idx] > 0.5
        eval_idx = eval_idx[edge_label]
        if eval_idx.numel() == 0:
            return out
        rel_eval = self.data.rel[eval_idx]
        unique_rel = torch.unique(rel_eval).detach().cpu().numpy().astype(np.int64).tolist()
        if self.verbose:
            print(f"[relation_breakdown] start relations={len(unique_rel)}")
        for step, rid in enumerate(unique_rel, start=1):
            sub_idx = eval_idx[rel_eval == int(rid)]
            if sub_idx.numel() == 0:
                continue
            m = eval_link_prediction(
                self.model,
                self.data,
                history_idx=history_idx,
                eval_idx=sub_idx,
                batch_size=self.batch_size,
                num_neg=num_neg,
                device=self.device,
                verbose=self.verbose,
                tag=f"relation_breakdown:{int(rid)}",
            )
            rname = self.rel_name_map.get(int(rid), f"rel_{int(rid)}")
            out[rname] = {k: float(v) for k, v in m.items()}
            if self.verbose:
                print(f"[relation_breakdown] progress relation={step}/{len(unique_rel)} name={rname}")
        if self.verbose:
            print("[relation_breakdown] done")
        return out

    def _eval_named_task_splits(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        extra = self.data.meta.get("extra", {}) if isinstance(self.data.meta, dict) else {}
        task_splits = extra.get("task_splits", {}) if isinstance(extra, dict) else {}
        if not isinstance(task_splits, dict):
            return out
        for tname, split in task_splits.items():
            if not isinstance(split, dict):
                continue
            num_e = int(self.data.src.numel())

            def _sanitize(ids):
                arr = np.asarray(ids, dtype=np.int64).reshape(-1)
                arr = arr[(arr >= 0) & (arr < num_e)]
                arr = np.unique(arr)
                return torch.as_tensor(arr, dtype=torch.long, device=self.device)

            tr = _sanitize(split.get("edge_indices_train", []))
            va = _sanitize(split.get("edge_indices_val", []))
            te = _sanitize(split.get("edge_indices_test", []))
            if te.numel() == 0:
                continue
            hist = tr if va.numel() == 0 else torch.cat([tr, va], dim=0)
            out[tname] = {
                "val": eval_link_prediction(
                    self.model,
                    self.data,
                    history_idx=tr,
                    eval_idx=va,
                    batch_size=self.batch_size,
                    num_neg=20,
                    device=self.device,
                )
                if va.numel() > 0
                else {"auc": 0.5, "ap": 0.0, "acc": 0.0, "mrr": 0.0},
                "test": eval_link_prediction(
                    self.model,
                    self.data,
                    history_idx=hist,
                    eval_idx=te,
                    batch_size=self.batch_size,
                    num_neg=20,
                    device=self.device,
                ),
            }
            out[tname]["val"] = {k: float(v) for k, v in out[tname]["val"].items()}
            out[tname]["test"] = {k: float(v) for k, v in out[tname]["test"].items()}
        return out

    def _node_loss(self, node_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model.predict_node_heads(node_ids)
        loss_mc = node_ids.new_tensor(0.0, dtype=torch.float)
        loss_ml = node_ids.new_tensor(0.0, dtype=torch.float)

        if self.mc_train_table is not None and out["multiclass"] is not None:
            y = self.mc_train_table[node_ids]
            mask = y >= 0
            if mask.any():
                loss_mc = F.cross_entropy(out["multiclass"][mask], y[mask])

        if self.ml_train_table is not None and out["multilabel"] is not None:
            y = self.ml_train_table[node_ids]
            mask = (y >= 0).all(dim=1)
            if mask.any():
                loss_ml = self.ml_loss_fn(out["multilabel"][mask], y[mask])

        return {"mc": loss_mc, "ml": loss_ml}

    @torch.no_grad()
    def _predict_multilabel_logits(self, node_ids: torch.Tensor, batch_size: int) -> np.ndarray:
        logits = []
        for idx in TemporalBatcher(node_ids, batch_size=batch_size, shuffle=False):
            out = self.model.predict_node_heads(idx)
            if out["multilabel"] is not None:
                logits.append(out["multilabel"].detach().cpu().numpy())
        if len(logits) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(logits, axis=0)

    @torch.no_grad()
    def _fit_multilabel_threshold(self) -> Dict[str, float]:
        out = {"threshold": 0.5, "calib_micro_f1": 0.0, "calib_nodes": 0}
        if not self.ml_threshold_search or self.data.node_multilabel is None:
            return out

        ml_ids = self.data.node_multilabel["node_ids"].long().to(self.device)
        ml_y = self.data.node_multilabel["labels"].detach().cpu().numpy().astype(np.float32)
        if ml_ids.numel() == 0:
            return out

        train_nodes = torch.cat([self.data.src[self.data.train_idx], self.data.dst[self.data.train_idx]], dim=0).unique()
        mask = torch.isin(ml_ids, train_nodes)
        if int(mask.sum().item()) < 64:
            mask = torch.ones_like(mask, dtype=torch.bool)

        calib_nodes = ml_ids[mask]
        calib_true = ml_y[mask.detach().cpu().numpy().astype(bool)]
        if calib_nodes.numel() == 0 or calib_true.size == 0:
            return out
        if self.ml_calib_max_nodes > 0 and calib_nodes.numel() > self.ml_calib_max_nodes:
            calib_nodes = calib_nodes[: self.ml_calib_max_nodes]
            calib_true = calib_true[: self.ml_calib_max_nodes]

        replay_history(
            self.model,
            self.data,
            history_idx=self.data.train_idx,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
            tag="ml_threshold_calib:replay",
        )
        calib_logits = self._predict_multilabel_logits(calib_nodes, batch_size=max(512, self.batch_size))
        if calib_logits.size == 0:
            return out

        candidates = np.linspace(0.01, 0.99, num=self.ml_threshold_num_candidates)
        thr, micro_f1 = find_best_multilabel_threshold(calib_true, calib_logits, candidates=candidates)
        out["threshold"] = float(thr)
        out["calib_micro_f1"] = float(micro_f1)
        out["calib_nodes"] = int(calib_nodes.numel())
        out["num_threshold_candidates"] = int(self.ml_threshold_num_candidates)
        return out

    def train(self) -> Dict[str, float]:
        best_val = -1.0
        best_result: Optional[Dict[str, float]] = None
        best_epoch = 0
        no_improve_epochs = 0
        history = []
        epoch_times = []

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            self.model.train()
            self.model.reset_state()
            epoch_loss = 0.0
            epoch_batches = 0

            for idx in TemporalBatcher(self.data.train_idx, batch_size=self.batch_size, shuffle=False):
                batch = make_edge_batch(self.data, idx)
                src = batch["src"]
                dst = batch["dst"]
                rel = batch["rel"]
                ts = batch["ts"]
                edge_feat = batch["edge_feat"]
                edge_label = batch["label"].float()
                neg_dst = self._sample_negative_dst(dst)

                out = self.model.forward_batch(src, dst, neg_dst, rel, ts, edge_feat)
                pos_logit = out["pos_logit"]
                neg_logit = out["neg_logit"]
                supervised_mask = edge_label > 0.5
                if self._is_edge_task() and supervised_mask.any():
                    pos_loss = F.binary_cross_entropy_with_logits(
                        pos_logit[supervised_mask], torch.ones_like(pos_logit[supervised_mask]), reduction="none"
                    )
                    neg_loss = F.binary_cross_entropy_with_logits(
                        neg_logit[supervised_mask], torch.zeros_like(neg_logit[supervised_mask]), reduction="none"
                    )
                    rel_w = self.rel_loss_weight[rel[supervised_mask]]
                    lp_loss = ((pos_loss + neg_loss) * rel_w).mean()
                else:
                    lp_loss = pos_logit.new_tensor(0.0)

                node_ids = torch.cat([src, dst]).unique()
                node_loss = self._node_loss(node_ids) if self._is_node_task() else {"mc": pos_logit.new_tensor(0.0), "ml": pos_logit.new_tensor(0.0)}

                loss = (
                    lp_loss
                    + self.lambda_cl * out["contrastive_loss"]
                    + self.lambda_community_entropy * out.get("community_entropy", pos_logit.new_tensor(0.0))
                    + (self.lambda_mc * node_loss["mc"] if self._is_node_task() else 0.0)
                    + (self.lambda_ml * node_loss["ml"] if self._is_node_task() else 0.0)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.model.update_state(out["cache"])

                epoch_loss += float(loss.detach().item())
                epoch_batches += 1

            if self._is_edge_task():
                val_metrics = eval_link_prediction(
                    self.model,
                    self.data,
                    history_idx=self.data.train_idx,
                    eval_idx=self.data.val_idx,
                    batch_size=self.batch_size,
                    num_neg=20,
                    device=self.device,
                )
                record = {
                    "epoch": epoch,
                    "loss": epoch_loss / max(1, epoch_batches),
                    "epoch_sec": float(epoch_sec := time.time() - t0),
                    **{f"val_{k}": float(v) for k, v in val_metrics.items()},
                }
            else:
                val_node_ids = self.node_eval_splits.get("val", torch.empty(0, dtype=torch.long, device=self.device))
                val_metrics = eval_node_tasks(
                    self.model,
                    self.data,
                    history_idx=self.data.train_idx,
                    batch_size=max(512, self.batch_size),
                    device=self.device,
                    target_node_ids=val_node_ids,
                    multilabel_threshold=0.5,
                    verbose=False,
                    tag="val_node_eval",
                )
                record = {
                    "epoch": epoch,
                    "loss": epoch_loss / max(1, epoch_batches),
                    "epoch_sec": float(epoch_sec := time.time() - t0),
                    **{f"val_{k}": float(v) for k, v in val_metrics.items()},
                }
            if self.save_memory_artifacts:
                self._snapshot_memory(epoch=epoch)
            epoch_times.append(float(epoch_sec))
            history.append(record)

            monitor_val = record.get(self.monitor_metric, np.nan)
            curr_val = float(monitor_val) if np.isfinite(monitor_val) else -1e18
            if curr_val > best_val + self.early_stop_min_delta:
                best_val = curr_val
                torch.save(self._checkpoint_state_dict(), self.out_dir / "best_model.pt")
                best_result = record
                best_epoch = int(epoch)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if self.verbose:
                msg = (
                    f"[train] epoch={epoch:03d} loss={record['loss']:.6f} "
                    f"monitor={self.monitor_metric}:{curr_val:.6f} "
                    f"time={record['epoch_sec']:.2f}s "
                    f"best_epoch={best_epoch} stale={no_improve_epochs}/{self.early_stop_patience}"
                )
                if self._is_edge_task():
                    msg += (
                        f" val_mrr={record.get('val_mrr', 0.0):.6f}"
                        f" val_auc={record.get('val_auc', 0.0):.6f}"
                        f" val_ap={record.get('val_ap', 0.0):.6f}"
                    )
                else:
                    if "val_node_ml_micro_f1" in record:
                        msg += (
                            f" val_node_ml_micro_f1={record.get('val_node_ml_micro_f1', 0.0):.6f}"
                            f" val_node_ml_map={record.get('val_node_ml_map', 0.0):.6f}"
                        )
                    if "val_node_mc_acc" in record:
                        msg += f" val_node_mc_acc={record.get('val_node_mc_acc', 0.0):.6f}"
                print(msg)

            if self.early_stop_patience > 0 and no_improve_epochs >= self.early_stop_patience:
                if self.verbose:
                    print(
                        f"[train] early-stop triggered at epoch={epoch} "
                        f"(best_epoch={best_epoch}, patience={self.early_stop_patience})"
                    )
                break

        self.model.load_state_dict(torch.load(self.out_dir / "best_model.pt", map_location=self.device), strict=False)
        hist_train_val = torch.cat([self.data.train_idx, self.data.val_idx], dim=0)
        if self._is_edge_task():
            test_metrics = eval_link_prediction(
                self.model,
                self.data,
                history_idx=hist_train_val,
                eval_idx=self.data.test_idx,
                batch_size=self.batch_size,
                num_neg=20,
                device=self.device,
                verbose=self.verbose,
                tag="test_link_eval",
            )
        else:
            test_metrics = {}

        if self._skip_aux_eval():
            if self.verbose:
                print(f"[train] skip auxiliary eval for ablation run: {self.out_dir.name}")
            ml_calib = {"threshold": 0.5, "calib_micro_f1": 0.0, "calib_nodes": 0}
            node_metrics = {}
            rel_metrics = {}
            task_metrics = {}
        elif self._is_edge_task():
            ml_calib = {"threshold": 0.5, "calib_micro_f1": 0.0, "calib_nodes": 0}
            node_metrics = {}
            rel_metrics = self._eval_relation_breakdown(history_idx=hist_train_val, eval_idx=self.data.test_idx, num_neg=20)
            task_metrics = self._eval_named_task_splits()
        else:
            eval_nodes = self.node_eval_splits.get("test", torch.empty(0, dtype=torch.long, device=self.device))
            if eval_nodes is not None and self.node_eval_max_nodes > 0 and eval_nodes.numel() > self.node_eval_max_nodes:
                eval_nodes = eval_nodes[: self.node_eval_max_nodes]
            ml_calib = self._fit_multilabel_threshold()
            node_metrics = eval_node_tasks(
                self.model,
                self.data,
                history_idx=hist_train_val,
                batch_size=max(512, self.batch_size),
                device=self.device,
                target_node_ids=eval_nodes,
                multilabel_threshold=float(ml_calib.get("threshold", 0.5)),
                verbose=self.verbose,
                tag="test_node_eval",
            )
            rel_metrics = {}
            task_metrics = {}

        mem_trace_path = None
        final_state_path = None
        if self.save_memory_artifacts:
            mem_trace_path = self.out_dir / "memory_trace.pt"
            torch.save({"snapshots": self.memory_snapshots}, mem_trace_path)

            replay_history(self.model, self.data, history_idx=hist_train_val, batch_size=self.batch_size, device=self.device)
            final_state_path = self.out_dir / "memory_final_state.pt"
            torch.save(
                {
                    "monitor_node_ids": self.monitor_node_ids.detach().cpu(),
                    "node_memory": self.model.memory.node_memory[self.monitor_node_ids].detach().cpu(),
                    "rel_memory": self.model.memory.rel_memory.detach().cpu(),
                    "community_memory": self.model.community.community_memory.detach().cpu()
                    if hasattr(self.model, "community") and hasattr(self.model.community, "community_memory")
                    else None,
                },
                final_state_path,
            )

        summary = {
            "best_epoch": int(best_epoch),
            "stopped_epoch": int(history[-1]["epoch"]) if len(history) > 0 else 0,
            "monitor_metric": self.monitor_metric,
            "best_val": best_result,
            "test": {k: float(v) for k, v in test_metrics.items()},
            "test_relation_breakdown": rel_metrics,
            "test_task_splits": task_metrics,
            "node_eval": node_metrics,
            "multilabel_threshold_calibration": ml_calib,
            "history": history,
            "epoch_time_sec": epoch_times,
            "avg_epoch_time_sec": float(np.mean(epoch_times)) if len(epoch_times) > 0 else 0.0,
            "memory_trace_file": str(mem_trace_path) if mem_trace_path is not None else "",
            "memory_final_state_file": str(final_state_path) if final_state_path is not None else "",
        }
        save_json(summary, str(self.out_dir / "metrics.json"))

        # Save csv logs for plotting.
        if history:
            keys = sorted(history[0].keys())
            with (self.out_dir / "train_history.csv").open("w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for row in history:
                    f.write(",".join(str(row[k]) for k in keys) + "\n")

        return {
            "test_mrr": float(summary.get("test", {}).get("mrr", 0.0)),
            "test_auc": float(summary.get("test", {}).get("auc", 0.0)),
            "test_ap": float(summary.get("test", {}).get("ap", 0.0)),
            "test_node_ml_micro_f1": float(summary.get("node_eval", {}).get("node_ml_micro_f1", 0.0)),
        }
