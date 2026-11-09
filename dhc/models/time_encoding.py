import math

import torch
from torch import nn


class TimeEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        freq = torch.exp(torch.linspace(0, math.log(1e4), dim // 2))
        self.register_buffer("freq", freq)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        x = delta_t.float().unsqueeze(-1) / (self.freq + 1e-8)
        sin = torch.sin(x)
        cos = torch.cos(x)
        out = torch.cat([sin, cos], dim=-1)
        if out.size(-1) < self.dim:
            pad = torch.zeros(out.size(0), self.dim - out.size(-1), device=out.device)
            out = torch.cat([out, pad], dim=-1)
        return out
