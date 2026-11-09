from typing import Dict, Iterator

import torch


class TemporalBatcher:
    def __init__(self, idx: torch.Tensor, batch_size: int, shuffle: bool = False):
        self.idx = idx
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.shuffle:
            perm = torch.randperm(self.idx.numel(), device=self.idx.device)
            order = self.idx[perm]
        else:
            order = self.idx
        for i in range(0, order.numel(), self.batch_size):
            yield order[i : i + self.batch_size]


def make_edge_batch(data, batch_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "src": data.src[batch_idx],
        "dst": data.dst[batch_idx],
        "ts": data.ts[batch_idx],
        "rel": data.rel[batch_idx],
        "edge_feat": data.edge_feat[batch_idx],
        "label": data.edge_label[batch_idx],
    }
