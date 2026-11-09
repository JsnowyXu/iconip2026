from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch


def _safe_label(prefix: str, value: int) -> str:
    return f'{prefix}-{value}'


def sample_graph_payload(pt_path: str, max_edges: int = 180, max_nodes: int = 80) -> Dict[str, Any]:
    path = Path(pt_path)
    if not path.exists():
        return {'nodes': [], 'edges': [], 'meta': {'missing': True}}

    obj = torch.load(path, map_location='cpu')
    edges = obj['edges']
    src = edges['src'].tolist()
    dst = edges['dst'].tolist()
    rel = edges['rel'].tolist()
    ts = edges['ts'].tolist()
    node_type = obj.get('node_type', torch.zeros(obj['num_nodes'], dtype=torch.long)).tolist()
    node_feat = obj.get('node_feat')
    extra = obj.get('meta', {}).get('extra', {}) if isinstance(obj.get('meta', {}), dict) else {}
    relation_map = extra.get('relation_map', {}) if isinstance(extra, dict) else {}
    node_type_map = extra.get('node_type_map', {}) if isinstance(extra, dict) else {}
    relation_name_by_id = {int(v): str(k) for k, v in relation_map.items()} if isinstance(relation_map, dict) else {}
    node_type_name_by_id = {int(v): str(k) for k, v in node_type_map.items()} if isinstance(node_type_map, dict) else {}

    sampled_edges = list(range(min(len(src), max_edges)))
    selected_nodes = []
    seen = set()
    for idx in sampled_edges:
        for nid in (int(src[idx]), int(dst[idx])):
            if nid not in seen:
                seen.add(nid)
                selected_nodes.append(nid)
            if len(selected_nodes) >= max_nodes:
                break
        if len(selected_nodes) >= max_nodes:
            break
    selected = set(selected_nodes)

    degrees = Counter()
    filtered_edges: List[Dict[str, Any]] = []
    for idx in sampled_edges:
        s = int(src[idx])
        d = int(dst[idx])
        if s not in selected or d not in selected:
            continue
        degrees[s] += 1
        degrees[d] += 1
        filtered_edges.append(
            {
                'id': idx,
                'source': s,
                'target': d,
                'relation_id': int(rel[idx]),
                'relation': relation_name_by_id.get(int(rel[idx]), _safe_label('rel', int(rel[idx]))),
                'timestamp': float(ts[idx]),
            }
        )

    nodes = []
    for nid in selected_nodes:
        fdim = int(node_feat.shape[1]) if node_feat is not None else 0
        nodes.append(
            {
                'id': int(nid),
                'label': f'Node {nid}',
                'node_type_id': int(node_type[nid]),
                'node_type': node_type_name_by_id.get(int(node_type[nid]), _safe_label('type', int(node_type[nid]))),
                'degree': int(degrees[nid]),
                'feature_dim': fdim,
            }
        )

    return {
        'nodes': nodes,
        'edges': filtered_edges,
        'meta': {
            'num_nodes_sampled': len(nodes),
            'num_edges_sampled': len(filtered_edges),
            'relation_histogram': Counter(edge['relation'] for edge in filtered_edges),
        },
    }
