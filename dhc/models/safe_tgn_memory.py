import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn.models.tgn import TGNMemory


class SafeTGNMemory(TGNMemory):
    def _pending_node_ids(self) -> Tensor:
        pending = []
        for node_id, payload in self.msg_s_store.items():
            if payload[0].numel() > 0:
                pending.append(node_id)
        for node_id, payload in self.msg_d_store.items():
            if payload[0].numel() > 0:
                pending.append(node_id)
        if not pending:
            return self.memory.new_empty((0,), device=self.memory.device, dtype=torch.long)
        pending = sorted(set(pending))
        return torch.tensor(pending, device=self.memory.device, dtype=torch.long)

    def train(self, mode: bool = True):
        if self.training and not mode:
            pending = self._pending_node_ids()
            if pending.numel() > 0:
                self._update_memory(pending)
            self._reset_message_store()
        nn.Module.train(self, mode)
