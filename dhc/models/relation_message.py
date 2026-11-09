import torch
from torch import nn


class RelationTimeMessage(nn.Module):
    """Relation-time dual-aware message function."""

    def __init__(self, in_dim: int, rel_dim: int, out_dim: int):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.rel_to_gate = nn.Sequential(nn.Linear(rel_dim, out_dim), nn.Sigmoid())
        self.rel_bias = nn.Linear(rel_dim, out_dim)

    def forward(self, x: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        gate = self.rel_to_gate(rel_emb)
        bias = self.rel_bias(rel_emb)
        return base * gate + bias
