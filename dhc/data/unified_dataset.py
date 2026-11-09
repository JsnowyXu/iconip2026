from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class UnifiedGraphData:
    num_nodes: int
    num_rels: int
    num_node_types: int
    node_feat: torch.Tensor
    node_type: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    ts: torch.Tensor
    rel: torch.Tensor
    edge_feat: torch.Tensor
    edge_label: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    node_multiclass: Optional[Dict[str, torch.Tensor]]
    node_multilabel: Optional[Dict[str, torch.Tensor]]
    meta: Dict


def load_unified_pt(path: str, device: Optional[torch.device] = None) -> UnifiedGraphData:
    obj = torch.load(path, map_location="cpu")
    edge = obj["edges"]
    split = obj["split"]

    data = UnifiedGraphData(
        num_nodes=int(obj["num_nodes"]),
        num_rels=int(obj["num_rels"]),
        num_node_types=int(obj.get("num_node_types", 1)),
        node_feat=obj["node_feat"],
        node_type=obj.get("node_type", torch.zeros(obj["num_nodes"], dtype=torch.long)),
        src=edge["src"].long(),
        dst=edge["dst"].long(),
        ts=edge["ts"].float(),
        rel=edge["rel"].long(),
        edge_feat=edge["edge_feat"].float(),
        edge_label=edge.get("edge_label", torch.ones_like(edge["src"]).float()).float(),
        train_idx=split["train"].long(),
        val_idx=split["val"].long(),
        test_idx=split["test"].long(),
        node_multiclass=obj.get("node_multiclass", None),
        node_multilabel=obj.get("node_multilabel", None),
        meta=obj.get("meta", {}),
    )

    if device is not None:
        for k in [
            "node_feat",
            "node_type",
            "src",
            "dst",
            "ts",
            "rel",
            "edge_feat",
            "edge_label",
            "train_idx",
            "val_idx",
            "test_idx",
        ]:
            setattr(data, k, getattr(data, k).to(device))

    return data
