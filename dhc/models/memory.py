from typing import Tuple

import torch
from torch import nn


class NodeRelationMemory(nn.Module):
    def __init__(self, num_nodes: int, num_rels: int, mem_dim: int, rel_mem_dim: int):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_rels = int(num_rels)
        self.mem_dim = int(mem_dim)
        self.rel_mem_dim = int(rel_mem_dim)

        self.node_updater = nn.GRUCell(mem_dim, mem_dim)
        self.rel_updater = nn.GRUCell(rel_mem_dim, rel_mem_dim)

        self.register_buffer("node_memory", torch.zeros(num_nodes, mem_dim))
        self.register_buffer("node_last_update", torch.zeros(num_nodes))
        self.register_buffer("rel_memory", torch.zeros(num_rels, rel_mem_dim))
        self.register_buffer("rel_last_update", torch.zeros(num_rels))

    def reset_state(self):
        self.node_memory.zero_()
        self.node_last_update.zero_()
        self.rel_memory.zero_()
        self.rel_last_update.zero_()

    def get_node_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.node_memory[node_ids]

    def get_rel_memory(self, rel_ids: torch.Tensor) -> torch.Tensor:
        return self.rel_memory[rel_ids]

    def get_node_last_update(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.node_last_update[node_ids]

    @staticmethod
    def _last_message(ids: torch.Tensor, msg: torch.Tensor, ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ids.numel() == 0:
            return ids, msg, ts
        # Keep last timestamp message per id.
        order = torch.argsort(ts)
        ids_o = ids[order]
        msg_o = msg[order]
        ts_o = ts[order]
        uniq, inv = torch.unique(ids_o, return_inverse=True)
        last_pos = torch.zeros_like(uniq)
        last_pos.scatter_(0, inv, torch.arange(inv.numel(), device=inv.device))
        return uniq, msg_o[last_pos], ts_o[last_pos]

    @torch.no_grad()
    def update_nodes(self, node_ids: torch.Tensor, messages: torch.Tensor, timestamps: torch.Tensor) -> None:
        uniq, msg, ts = self._last_message(node_ids, messages, timestamps)
        if uniq.numel() == 0:
            return
        prev = self.node_memory[uniq]
        self.node_memory[uniq] = self.node_updater(msg, prev)
        self.node_last_update[uniq] = ts

    @torch.no_grad()
    def update_rels(self, rel_ids: torch.Tensor, rel_messages: torch.Tensor, timestamps: torch.Tensor) -> None:
        uniq, msg, ts = self._last_message(rel_ids, rel_messages, timestamps)
        if uniq.numel() == 0:
            return
        prev = self.rel_memory[uniq]
        self.rel_memory[uniq] = self.rel_updater(msg, prev)
        self.rel_last_update[uniq] = ts
