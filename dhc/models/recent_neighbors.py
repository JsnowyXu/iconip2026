from typing import Tuple

import torch
from torch import nn


class RecentNeighborStore(nn.Module):
    """Stores the most recent undirected interactions per node."""

    def __init__(self, num_nodes: int, size: int, edge_feat_dim: int = 0):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.size = int(max(1, size))
        self.edge_feat_dim = int(max(0, edge_feat_dim))

        self.register_buffer("neighbors", torch.full((num_nodes, self.size), -1, dtype=torch.long))
        self.register_buffer("relations", torch.full((num_nodes, self.size), -1, dtype=torch.long))
        self.register_buffer("timestamps", torch.zeros((num_nodes, self.size), dtype=torch.float))
        self.register_buffer("event_ids", torch.full((num_nodes, self.size), -1, dtype=torch.long))
        if self.edge_feat_dim > 0:
            self.register_buffer("edge_feat", torch.zeros((num_nodes, self.size, self.edge_feat_dim), dtype=torch.float))
        else:
            self.register_buffer("edge_feat", torch.zeros((num_nodes, self.size, 0), dtype=torch.float))
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))
        self.cur_e_id = 0

    def reset_state(self) -> None:
        self.neighbors.fill_(-1)
        self.relations.fill_(-1)
        self.timestamps.zero_()
        self.event_ids.fill_(-1)
        if self.edge_feat_dim > 0:
            self.edge_feat.zero_()
        self.cur_e_id = 0

    @torch.no_grad()
    def insert(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        rel: torch.Tensor,
        ts: torch.Tensor,
        edge_feat: torch.Tensor = None,
    ) -> None:
        if src.numel() == 0:
            return

        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        rels = torch.cat([rel, rel], dim=0)
        tss = torch.cat([ts, ts], dim=0)
        if self.edge_feat_dim > 0:
            if edge_feat is None:
                feat = self.edge_feat.new_zeros((src.numel(), self.edge_feat_dim))
            else:
                feat = edge_feat
            feats = torch.cat([feat, feat], dim=0)
        else:
            feats = self.edge_feat.new_zeros((src.numel() * 2, 0))
        e_ids = torch.arange(self.cur_e_id, self.cur_e_id + src.numel(), device=src.device, dtype=torch.long).repeat(2)
        self.cur_e_id += int(src.numel())

        nodes, perm = nodes.sort()
        neighbors = neighbors[perm]
        rels = rels[perm]
        tss = tss[perm]
        feats = feats[perm]
        e_ids = e_ids[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes] * self.size

        dense_e = e_ids.new_full((n_id.numel() * self.size,), -1)
        dense_n = neighbors.new_full((n_id.numel() * self.size,), -1)
        dense_r = rels.new_full((n_id.numel() * self.size,), -1)
        dense_t = tss.new_zeros((n_id.numel() * self.size,))
        dense_f = self.edge_feat.new_zeros((n_id.numel() * self.size, self.edge_feat_dim))

        dense_e[dense_id] = e_ids
        dense_n[dense_id] = neighbors
        dense_r[dense_id] = rels
        dense_t[dense_id] = tss
        if self.edge_feat_dim > 0:
            dense_f[dense_id] = feats

        dense_e = dense_e.view(-1, self.size)
        dense_n = dense_n.view(-1, self.size)
        dense_r = dense_r.view(-1, self.size)
        dense_t = dense_t.view(-1, self.size)
        dense_f = dense_f.view(-1, self.size, self.edge_feat_dim)

        old_e = self.event_ids[n_id, : self.size]
        old_n = self.neighbors[n_id, : self.size]
        old_r = self.relations[n_id, : self.size]
        old_t = self.timestamps[n_id, : self.size]
        old_f = self.edge_feat[n_id, : self.size]

        all_e = torch.cat([old_e, dense_e], dim=-1)
        all_n = torch.cat([old_n, dense_n], dim=-1)
        all_r = torch.cat([old_r, dense_r], dim=-1)
        all_t = torch.cat([old_t, dense_t], dim=-1)
        all_f = torch.cat([old_f, dense_f], dim=1)

        top_e, perm = all_e.topk(self.size, dim=-1)
        self.event_ids[n_id] = top_e
        self.neighbors[n_id] = torch.gather(all_n, 1, perm)
        self.relations[n_id] = torch.gather(all_r, 1, perm)
        self.timestamps[n_id] = torch.gather(all_t, 1, perm)
        if self.edge_feat_dim > 0:
            self.edge_feat[n_id] = torch.gather(all_f, 1, perm.unsqueeze(-1).expand(-1, -1, self.edge_feat_dim))

    def get(self, node_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.neighbors[node_ids],
            self.relations[node_ids],
            self.timestamps[node_ids],
            self.event_ids[node_ids],
            self.edge_feat[node_ids],
        )

    def subgraph(
        self,
        node_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        neigh, rel, ts, event_ids, edge_feat = self.get(node_ids)
        nodes = node_ids.view(-1, 1).repeat(1, self.size)
        mask = event_ids >= 0
        if not bool(mask.any().item()):
            empty_idx = node_ids.new_empty((2, 0))
            empty_rel = rel.new_empty((0,))
            empty_ts = ts.new_empty((0,))
            empty_feat = edge_feat.new_empty((0, self.edge_feat_dim))
            return node_ids.unique(), empty_idx, empty_rel, empty_ts, empty_feat

        neigh = neigh[mask]
        nodes = nodes[mask]
        rel = rel[mask]
        ts = ts[mask]
        edge_feat = edge_feat[mask]

        n_id = torch.cat([node_ids, neigh]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        edge_index = torch.stack([self._assoc[neigh], self._assoc[nodes]], dim=0)
        return n_id, edge_index, rel, ts, edge_feat
