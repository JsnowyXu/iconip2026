import math
from typing import Tuple

import torch
from torch import nn


class CommunityMemory(nn.Module):
    """Mesoscopic community memory with soft assignment and contrastive objective."""

    def __init__(
        self,
        num_communities: int,
        node_dim: int,
        community_dim: int,
        tau: float = 0.2,
        assign_temperature: float = 1.0,
        assign_topk: int = 0,
    ):
        super().__init__()
        self.num_communities = int(num_communities)
        self.node_dim = int(node_dim)
        self.community_dim = int(community_dim)
        self.tau = float(tau)
        self.assign_temperature = float(max(1e-4, assign_temperature))
        self.assign_topk = int(max(0, assign_topk))

        self.prototypes = nn.Parameter(torch.randn(num_communities, node_dim) * 0.02)
        self.node_to_comm = nn.Linear(node_dim, community_dim)
        self.proto_to_comm = nn.Linear(node_dim, community_dim, bias=False)
        self.in_proj = nn.Linear(node_dim, community_dim)
        self.out_proj = nn.Linear(node_dim + community_dim, node_dim)
        self.updater = nn.GRUCell(community_dim, community_dim)

        self.register_buffer("community_memory", torch.zeros(num_communities, community_dim))
        self.register_buffer("community_last_update", torch.zeros(num_communities))

    def reset_state(self) -> None:
        self.community_memory.zero_()
        self.community_last_update.zero_()

    def assign(self, node_emb: torch.Tensor) -> torch.Tensor:
        q = self.node_to_comm(node_emb)
        k = self.proto_to_comm(self.prototypes)
        logits = (q @ k.t()) / (math.sqrt(max(1, q.size(-1))) * self.assign_temperature)
        if 0 < self.assign_topk < logits.size(1):
            topk_val, topk_idx = torch.topk(logits, k=self.assign_topk, dim=-1)
            masked = torch.full_like(logits, -1e9)
            masked.scatter_(1, topk_idx, topk_val)
            logits = masked
        return torch.softmax(logits, dim=-1)

    def context(self, node_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.assign(node_emb)
        ctx = alpha @ self.community_memory
        out = self.out_proj(torch.cat([node_emb, ctx], dim=-1))
        return out, alpha

    @torch.no_grad()
    def update(self, node_emb: torch.Tensor, timestamps: torch.Tensor) -> None:
        if node_emb.numel() == 0:
            return
        alpha = self.assign(node_emb)
        msg = self.in_proj(node_emb)
        weighted = alpha.t() @ msg
        denom = alpha.sum(dim=0).unsqueeze(-1).clamp_min(1e-6)
        aggregated = weighted / denom

        t_now = float(timestamps.max().item()) if timestamps.numel() > 0 else 0.0
        t_now_tensor = torch.full_like(self.community_last_update, t_now)
        dt = (t_now_tensor - self.community_last_update).clamp_min(0.0)
        decay = torch.exp(-dt / 1000.0).unsqueeze(-1)
        prev = self.community_memory * decay
        self.community_memory = self.updater(aggregated, prev)
        self.community_last_update = t_now_tensor

    def contrastive_loss(self, node_emb: torch.Tensor) -> torch.Tensor:
        if node_emb.numel() == 0:
            return node_emb.new_tensor(0.0)
        alpha = self.assign(node_emb)
        pos_idx = torch.argmax(alpha, dim=-1)
        node_q = nn.functional.normalize(self.node_to_comm(node_emb), dim=-1)
        comm_k = nn.functional.normalize(self.community_memory, dim=-1)
        logits = node_q @ comm_k.t() / self.tau
        target = pos_idx
        return nn.functional.cross_entropy(logits, target)

    def normalized_entropy(self, node_emb: torch.Tensor) -> torch.Tensor:
        if node_emb.numel() == 0:
            return node_emb.new_tensor(0.0)
        alpha = self.assign(node_emb)
        ent = -(alpha * torch.log(alpha.clamp_min(1e-12))).sum(dim=-1)
        norm = math.log(max(2, self.num_communities))
        return (ent / norm).mean()
