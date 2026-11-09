from typing import Dict, Optional

import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

from dhc.models.community import CommunityMemory
from dhc.models.memory import NodeRelationMemory
from dhc.models.recent_neighbors import RecentNeighborStore
from dhc.models.relation_message import RelationTimeMessage
from dhc.models.safe_tgn_memory import SafeTGNMemory
from dhc.models.time_encoding import TimeEncoding


class DHCModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_rels: int,
        num_node_types: int,
        node_feat: torch.Tensor,
        node_type: torch.Tensor,
        edge_feat_dim: int,
        mem_dim: int = 128,
        rel_dim: int = 64,
        time_dim: int = 32,
        msg_dim: int = 128,
        community_dim: int = 128,
        num_communities: int = 64,
        num_multiclass: int = 0,
        num_multilabel: int = 0,
        use_relation_memory: bool = True,
        use_community: bool = True,
        community_assign_temperature: float = 1.0,
        community_assign_topk: int = 0,
        use_recent_neighbors: bool = True,
        recent_neighbor_size: int = 10,
        use_edge_type_features: bool = True,
        neighbor_aggregation: str = "attn",
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_rels = int(num_rels)
        self.mem_dim = int(mem_dim)
        self.rel_dim = int(rel_dim)
        self.time_dim = int(time_dim)
        self.msg_dim = int(msg_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.use_relation_memory = bool(use_relation_memory)
        self.use_community = bool(use_community)
        self.use_recent_neighbors = bool(use_recent_neighbors)
        self.use_edge_type_features = bool(use_edge_type_features)
        self.recent_neighbor_size = int(max(1, recent_neighbor_size))
        self.neighbor_aggregation = str(neighbor_aggregation or "attn").strip().lower()
        if self.neighbor_aggregation not in {"attn", "mean"}:
            raise ValueError(f"unsupported neighbor_aggregation: {self.neighbor_aggregation}")
        self.raw_edge_feat_dim = int(edge_feat_dim) + (int(num_rels) if self.use_edge_type_features else 0)

        self.register_buffer("node_feat", node_feat.float())
        self.register_buffer("node_type", node_type.long())

        self.node_type_emb = nn.Embedding(max(1, num_node_types), 16)
        self.rel_emb = nn.Embedding(num_rels, rel_dim)

        self.memory = NodeRelationMemory(num_nodes, num_rels, mem_dim=mem_dim, rel_mem_dim=mem_dim)
        self.time_enc = TimeEncoding(time_dim)
        self.recent_neighbors = RecentNeighborStore(
            num_nodes,
            self.recent_neighbor_size,
            edge_feat_dim=self.raw_edge_feat_dim,
        )
        self.tgn_memory = SafeTGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=self.raw_edge_feat_dim,
            memory_dim=mem_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(self.raw_edge_feat_dim, mem_dim, time_dim),
            aggregator_module=LastAggregator(),
        )

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat.size(1) + 16 + mem_dim, mem_dim),
            nn.ReLU(),
            nn.Linear(mem_dim, mem_dim),
        )
        self.local_graph_encoder = TransformerConv(
            mem_dim,
            max(1, mem_dim // 2),
            heads=2,
            dropout=0.1,
            edge_dim=rel_dim + self.raw_edge_feat_dim + time_dim,
        )
        self.local_graph_fusion = nn.Sequential(
            nn.Linear(2 * mem_dim, mem_dim),
            nn.ReLU(),
            nn.Linear(mem_dim, mem_dim),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * mem_dim, mem_dim),
            nn.Sigmoid(),
        )
        self.neighbor_value = nn.Linear(mem_dim + rel_dim + time_dim, mem_dim)
        self.neighbor_attn = nn.Sequential(
            nn.Linear(2 * mem_dim + rel_dim + time_dim, mem_dim),
            nn.ReLU(),
            nn.Linear(mem_dim, 1),
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(2 * mem_dim, mem_dim),
            nn.ReLU(),
            nn.Linear(mem_dim, mem_dim),
        )
        self.rel_mem_to_vec = nn.Linear(mem_dim, rel_dim)

        msg_in_dim = 2 * mem_dim + self.raw_edge_feat_dim + time_dim
        self.msg_fn = RelationTimeMessage(msg_in_dim, rel_dim, msg_dim)
        self.msg_to_mem = nn.Linear(msg_dim, mem_dim)
        self.msg_to_rel = nn.Linear(msg_dim, mem_dim)

        self.community = CommunityMemory(
            num_communities,
            mem_dim,
            community_dim,
            assign_temperature=community_assign_temperature,
            assign_topk=community_assign_topk,
        )

        self.link_predictor = nn.Sequential(
            nn.Linear(2 * mem_dim + rel_dim + time_dim, mem_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mem_dim, 1),
        )

        self.multiclass_head = nn.Linear(mem_dim, num_multiclass) if num_multiclass > 0 else None
        self.multilabel_head = nn.Linear(mem_dim, num_multilabel) if num_multilabel > 0 else None

    def reset_state(self):
        self.memory.reset_state()
        self.recent_neighbors.reset_state()
        self.tgn_memory.reset_state()
        self.community.reset_state()

    def _base_node_repr(self, node_ids: torch.Tensor) -> torch.Tensor:
        feat = self.node_feat[node_ids]
        typ = self.node_type_emb(self.node_type[node_ids].clamp_min(0))
        mem = self.memory.get_node_memory(node_ids)
        return self.node_encoder(torch.cat([feat, typ, mem], dim=-1))

    def _recent_neighbor_context(self, node_ids: torch.Tensor, query_h: torch.Tensor, query_ts: torch.Tensor) -> torch.Tensor:
        neigh_ids, rel_ids, neigh_ts, event_ids, _ = self.recent_neighbors.get(node_ids)
        mask = event_ids >= 0
        if not bool(mask.any().item()):
            return query_h.new_zeros(query_h.shape)

        neigh_safe = neigh_ids.clamp_min(0)
        rel_safe = rel_ids.clamp_min(0)

        neigh_mem = self.memory.node_memory[neigh_safe]
        rel_vec = self.rel_emb(rel_safe)
        if self.use_relation_memory:
            rel_mem = self.memory.rel_memory[rel_safe]
            rel_vec = rel_vec + self.rel_mem_to_vec(rel_mem)

        dt = (query_ts.view(-1, 1) - neigh_ts).clamp_min(0.0)
        dt_vec = self.time_enc(dt.reshape(-1)).view(node_ids.size(0), self.recent_neighbor_size, -1)

        value = self.neighbor_value(torch.cat([neigh_mem, rel_vec, dt_vec], dim=-1))
        if self.neighbor_aggregation == "mean":
            denom = mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            return torch.sum(mask.unsqueeze(-1).float() * value, dim=1) / denom

        attn_in = torch.cat([query_h.unsqueeze(1).expand_as(value), value, rel_vec, dt_vec], dim=-1)
        score = self.neighbor_attn(attn_in).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=-1) * mask.float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.sum(alpha.unsqueeze(-1) * value, dim=1)

    def _node_repr(self, node_ids: torch.Tensor, query_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_h = self._base_node_repr(node_ids)
        h = base_h
        if self.use_recent_neighbors:
            if query_ts is None:
                query_ts = self.memory.get_node_last_update(node_ids)
            neigh_ctx = self._recent_neighbor_context(node_ids, base_h, query_ts)
            h = self.node_fusion(torch.cat([base_h, neigh_ctx], dim=-1))
        if self.use_community:
            h, _ = self.community.context(h)
        return h

    def _rel_vec(self, rel_ids: torch.Tensor) -> torch.Tensor:
        rel_vec = self.rel_emb(rel_ids)
        if self.use_relation_memory:
            rel_mem = self.memory.get_rel_memory(rel_ids)
            rel_vec = rel_vec + self.rel_mem_to_vec(rel_mem)
        return rel_vec

    def _augment_edge_feat(self, edge_feat: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_edge_type_features:
            return edge_feat
        rel_onehot = torch.nn.functional.one_hot(rel_ids.clamp_min(0), num_classes=self.num_rels).float()
        if rel_onehot.device != edge_feat.device:
            rel_onehot = rel_onehot.to(edge_feat.device)
        return torch.cat([edge_feat, rel_onehot], dim=-1)

    def _delta_t(self, node_ids: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        last = self.memory.get_node_last_update(node_ids)
        return (ts - last).clamp_min(0.0)

    def _encode_link_query_nodes(self, query_nodes: torch.Tensor) -> torch.Tensor:
        if not self.use_recent_neighbors:
            h = self._base_node_repr(query_nodes)
            if self.use_community:
                h, _ = self.community.context(h)
            return h

        local_nodes, edge_index, rel_ids, edge_ts, edge_feat = self.recent_neighbors.subgraph(query_nodes)
        base = self._base_node_repr(local_nodes)
        if edge_index.numel() == 0:
            local_h = base
        else:
            src_local = local_nodes[edge_index[0]]
            src_last = self.memory.get_node_last_update(src_local)
            rel_dt = (src_last - edge_ts).abs()
            edge_attr = torch.cat([self._rel_vec(rel_ids), edge_feat, self.time_enc(rel_dt)], dim=-1)
            msg_h = self.local_graph_encoder(base, edge_index, edge_attr)
            local_h = self.local_graph_fusion(torch.cat([base, msg_h], dim=-1))
        if self.use_community:
            local_h, _ = self.community.context(local_h)
        assoc = torch.empty(self.num_nodes, dtype=torch.long, device=query_nodes.device)
        assoc[local_nodes] = torch.arange(local_nodes.size(0), device=query_nodes.device)
        return local_h[assoc[query_nodes]]

    def predict_links(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        rel: torch.Tensor,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        query_nodes = torch.cat([src, dst], dim=0).unique()
        dhc_query_h = self._encode_link_query_nodes(query_nodes)
        tgn_query_h, _ = self.tgn_memory(query_nodes)
        gate = self.edge_gate(torch.cat([dhc_query_h, tgn_query_h], dim=-1))
        query_h = gate * tgn_query_h + (1.0 - gate) * dhc_query_h
        assoc = torch.empty(self.num_nodes, dtype=torch.long, device=src.device)
        assoc[query_nodes] = torch.arange(query_nodes.size(0), device=src.device)
        src_h = query_h[assoc[src]]
        dst_h = query_h[assoc[dst]]
        rel_v = self._rel_vec(rel)
        dt = self.time_enc(self._delta_t(src, ts))
        x = torch.cat([src_h, dst_h, rel_v, dt], dim=-1)
        return self.link_predictor(x).squeeze(-1)

    def forward_batch(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        neg_dst: torch.Tensor,
        rel: torch.Tensor,
        ts: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        aug_edge_feat = self._augment_edge_feat(edge_feat, rel)
        pos_logit = self.predict_links(src, dst, rel, ts)
        neg_logit = self.predict_links(src, neg_dst, rel, ts)

        # Relation-time dual-aware messages.
        src_mem = self.memory.get_node_memory(src)
        dst_mem = self.memory.get_node_memory(dst)
        rel_vec = self._rel_vec(rel)
        dt_src = self.time_enc(self._delta_t(src, ts))
        dt_dst = self.time_enc(self._delta_t(dst, ts))

        src_msg_raw = torch.cat([dst_mem, src_mem, aug_edge_feat, dt_src], dim=-1)
        dst_msg_raw = torch.cat([src_mem, dst_mem, aug_edge_feat, dt_dst], dim=-1)
        src_msg = self.msg_fn(src_msg_raw, rel_vec)
        dst_msg = self.msg_fn(dst_msg_raw, rel_vec)

        src_msg_mem = self.msg_to_mem(src_msg)
        dst_msg_mem = self.msg_to_mem(dst_msg)
        rel_msg_mem = self.msg_to_rel((src_msg + dst_msg) / 2.0)

        # For community updates and contrastive objective.
        active_nodes = torch.cat([src, dst]).unique()
        active_emb = self._node_repr(active_nodes)
        contrastive = self.community.contrastive_loss(active_emb) if self.use_community else pos_logit.new_tensor(0.0)
        community_entropy = self.community.normalized_entropy(active_emb) if self.use_community else pos_logit.new_tensor(0.0)

        cache = {
            "src": src.detach(),
            "dst": dst.detach(),
            "rel": rel.detach(),
            "ts": ts.detach(),
            "src_msg_mem": src_msg_mem.detach(),
            "dst_msg_mem": dst_msg_mem.detach(),
            "rel_msg_mem": rel_msg_mem.detach(),
            "edge_feat": aug_edge_feat.detach(),
            "active_nodes": active_nodes.detach(),
            "active_emb": active_emb.detach(),
        }

        return {
            "pos_logit": pos_logit,
            "neg_logit": neg_logit,
            "contrastive_loss": contrastive,
            "community_entropy": community_entropy,
            "cache": cache,
        }

    @torch.no_grad()
    def update_state(self, cache: Dict[str, torch.Tensor]) -> None:
        src = cache["src"]
        dst = cache["dst"]
        rel = cache["rel"]
        ts = cache["ts"]

        all_nodes = torch.cat([src, dst], dim=0)
        all_msgs = torch.cat([cache["src_msg_mem"], cache["dst_msg_mem"]], dim=0)
        all_ts = torch.cat([ts, ts], dim=0)
        self.memory.update_nodes(all_nodes, all_msgs, all_ts)
        self.recent_neighbors.insert(src, dst, rel, ts, edge_feat=cache["edge_feat"])
        self.tgn_memory.update_state(src, dst, ts.long(), cache["edge_feat"])
        self.tgn_memory.detach()

        if self.use_relation_memory:
            self.memory.update_rels(rel, cache["rel_msg_mem"], ts)

        if self.use_community:
            self.community.update(cache["active_emb"], ts)

    def predict_node_heads(self, node_ids: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        h = self._node_repr(node_ids)
        out = {
            "multiclass": self.multiclass_head(h) if self.multiclass_head is not None else None,
            "multilabel": self.multilabel_head(h) if self.multilabel_head is not None else None,
        }
        return out
