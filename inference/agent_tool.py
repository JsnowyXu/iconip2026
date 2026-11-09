"""Stateful, causal link ranking over the dataset's internal node IDs."""
import inspect
import json
from pathlib import Path
from threading import RLock

import torch

from dhc.data.unified_dataset import load_unified_pt
from dhc.models.dhc_model import DHCModel
from dhc.eval.evaluator import replay_history


class DHCTool:
    def __init__(self, run_dir, root=None, device="cpu", history="train", replay_limit=None):
        self.root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        run_dir = Path(run_dir)
        if not run_dir.is_absolute():
            run_dir = self.root / run_dir
        metrics = json.loads((run_dir / "metrics.json").read_text())
        args = metrics.get("run_args") or json.loads((run_dir / "model_stats.json").read_text())["args"]
        data_path = args["data"]
        if "/data/" in data_path:
            data_path = "data/" + data_path.split("/data/", 1)[1]
        self.device = torch.device(device)
        self.data = load_unified_pt(str(self.root / data_path), self.device)
        data = self.data
        kwargs = {k: args[k] for k in inspect.signature(DHCModel).parameters if k in args}
        kwargs.update(num_nodes=data.num_nodes, num_rels=data.num_rels,
                      num_node_types=data.num_node_types, node_feat=data.node_feat,
                      node_type=data.node_type, edge_feat_dim=data.edge_feat.shape[1],
                      num_multiclass=int(data.node_multiclass["labels"].max()) + 1 if data.node_multiclass else 0,
                      num_multilabel=data.node_multilabel["labels"].shape[1] if data.node_multilabel else 0)
        for name in ["relation_memory", "community", "recent_neighbors", "edge_type_features"]:
            kwargs["use_" + name] = not args.get("no_" + name, False)
        self.model = DHCModel(**kwargs).to(self.device)
        state = torch.load(run_dir / "best_model.pt", map_location=self.device)
        # Match the saved lightweight checkpoint exactly; never silently ignore a missing weight.
        expected = self.model.state_dict()
        unexpected = set(state) - set(expected)
        missing_parameters = set(dict(self.model.named_parameters())) - set(state)
        if unexpected or missing_parameters:
            raise ValueError(f"Incomplete checkpoint: unexpected={unexpected}, missing_parameters={missing_parameters}. "
                             "Legacy saver also dropped trained node_type_emb; retrain with the fixed saver.")
        self.model.load_state_dict(state, strict=False)
        self.lock = RLock()
        if history not in {"none", "train", "train_val"}:
            raise ValueError("history must be none, train, or train_val")
        idx = data.train_idx[:0] if history == "none" else data.train_idx
        if history == "train_val":
            idx = torch.cat([idx, data.val_idx])
        if replay_limit is not None:
            idx = idx[:replay_limit]
        replay_history(self.model, data, idx, batch_size=args.get("batch_size", 256), device=self.device)
        self.watermark = float(data.ts[idx].max()) if idx.numel() else float("-inf")

    def _validate(self, src, candidates, rel, timestamp):
        if any(not isinstance(x, int) or isinstance(x, bool) for x in [src, rel, *candidates]):
            raise ValueError("Node and relation IDs must be integers")
        if not 0 <= src < self.data.num_nodes or any(not 0 <= x < self.data.num_nodes for x in candidates):
            raise ValueError("Node ID outside the fixed training vocabulary")
        if not 0 <= rel < self.data.num_rels:
            raise ValueError("Invalid relation ID")
        if not torch.isfinite(torch.tensor(timestamp)) or timestamp < self.watermark:
            raise ValueError("Timestamp must be finite and >= observed history watermark")

    @torch.no_grad()
    def rank_links(self, source_id, candidate_ids, relation_id, timestamp, top_k=10):
        """Read-only ranking. sigmoid_score is not a calibrated probability."""
        with self.lock:
            self._validate(source_id, candidate_ids, relation_id, timestamp)
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1 or not candidate_ids:
                raise ValueError("Nonempty candidates and positive top_k required")
            if len(candidate_ids) > 4096:
                raise ValueError("At most 4096 candidates per call")
            dst = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
            logits = self.model.predict_links(torch.full_like(dst, source_id), dst,
                        torch.full_like(dst, relation_id), torch.full_like(dst, timestamp, dtype=torch.float))
            values, indices = logits.topk(min(top_k, len(candidate_ids)))
            return {"source_id": source_id, "relation_id": relation_id, "timestamp": timestamp,
                    "history_watermark": self.watermark if self.watermark != float("-inf") else None,
                    "ranked": [{"node_id": candidate_ids[i], "logit": float(v),
                                "sigmoid_score": float(v.sigmoid())} for v, i in zip(values, indices.tolist())]}

    @torch.no_grad()
    def observe(self, source_id, destination_id, relation_id, timestamp, edge_features):
        """Commit one confirmed real event, including context edges; never commit predictions."""
        with self.lock:
            self._validate(source_id, [destination_id], relation_id, timestamp)
            if len(edge_features) != self.data.edge_feat.shape[1]:
                raise ValueError("edge_features dimension differs from training data")
            def ids(x):
                return torch.tensor([x], device=self.device, dtype=torch.long)
            feat = torch.tensor([edge_features], device=self.device, dtype=torch.float)
            if not torch.isfinite(feat).all():
                raise ValueError("edge_features must be finite")
            out = self.model.forward_batch(ids(source_id), ids(destination_id), ids(destination_id),
                        ids(relation_id), torch.tensor([timestamp], device=self.device), feat)
            self.model.update_state(out["cache"])
            self.watermark = float(timestamp)
            return {"observed": True, "history_watermark": self.watermark}
