import json
from pathlib import Path
from typing import Any, Dict

from core.models import ModelArtifact

KNOWN_DATASET_PREFIXES = [
    "amazon_movies",
    "raw_cn",
    "raw_us",
    "arxiv",
    "dblp",
    "kg4ai",
]


def _infer_dataset_key(dataset_path: str) -> str:
    stem = Path(dataset_path).stem
    for prefix in KNOWN_DATASET_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return stem.split("_")[0] if stem else ""


def _build_model(artifact: ModelArtifact):
    import torch
    from dhc.data.unified_dataset import load_unified_pt
    from dhc.models.dhc_model import DHCModel

    run_args = artifact.config.get('run_args', artifact.config.get('args', {}))
    if not artifact.dataset_path:
        raise ValueError('dataset_path is missing for this artifact')
    device = torch.device('cpu')
    data = load_unified_pt(artifact.dataset_path, device=device)
    num_multiclass = 0
    num_multilabel = 0
    if data.node_multiclass is not None:
        num_multiclass = int(data.node_multiclass['labels'].max().item()) + 1
    if data.node_multilabel is not None:
        num_multilabel = int(data.node_multilabel['labels'].shape[1])
    model = DHCModel(
        num_nodes=data.num_nodes,
        num_rels=data.num_rels,
        num_node_types=data.num_node_types,
        node_feat=data.node_feat,
        node_type=data.node_type,
        edge_feat_dim=data.edge_feat.shape[1],
        mem_dim=int(run_args.get('mem_dim', 128)),
        rel_dim=int(run_args.get('rel_dim', 64)),
        time_dim=int(run_args.get('time_dim', 32)),
        msg_dim=int(run_args.get('msg_dim', 128)),
        community_dim=int(run_args.get('community_dim', 128)),
        num_communities=int(run_args.get('num_communities', 64)),
        num_multiclass=num_multiclass,
        num_multilabel=num_multilabel,
        use_relation_memory=not bool(run_args.get('no_relation_memory', False)),
        use_community=not bool(run_args.get('no_community', False)),
        community_assign_temperature=float(run_args.get('community_assign_temperature', 1.0)),
        community_assign_topk=int(run_args.get('community_assign_topk', 0)),
        use_recent_neighbors=not bool(run_args.get('no_recent_neighbors', False)),
        recent_neighbor_size=int(run_args.get('recent_neighbor_size', 10)),
    ).to(device)
    state = torch.load(artifact.checkpoint_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, data, device


def run_inference(artifact: ModelArtifact, payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from dhc.eval.evaluator import replay_history

    model, data, device = _build_model(artifact)
    history_idx = torch.cat([data.train_idx, data.val_idx], dim=0)
    replay_history(model, data, history_idx=history_idx, batch_size=512, device=device)
    mode = payload.get('mode', 'link')
    if mode == 'link':
        src = torch.tensor([int(payload['src'])], dtype=torch.long, device=device)
        dst = torch.tensor([int(payload['dst'])], dtype=torch.long, device=device)
        rel = torch.tensor([int(payload.get('rel', 0))], dtype=torch.long, device=device)
        ts = torch.tensor([float(payload.get('ts', float(data.ts.max().item()) if data.ts.numel() > 0 else 0.0))], dtype=torch.float32, device=device)
        score = torch.sigmoid(model.predict_links(src, dst, rel, ts)).item()
        return {
            'mode': 'link',
            'score': float(score),
            'confidence': 'high' if score >= 0.8 else 'medium' if score >= 0.5 else 'low',
            'inputs': {'src': int(src.item()), 'dst': int(dst.item()), 'rel': int(rel.item()), 'ts': float(ts.item())},
        }

    node_id = torch.tensor([int(payload['node_id'])], dtype=torch.long, device=device)
    heads = model.predict_node_heads(node_id)
    response: Dict[str, Any] = {'mode': 'node', 'node_id': int(node_id.item())}
    if heads['multiclass'] is not None:
        probs = torch.softmax(heads['multiclass'], dim=-1)[0]
        topv, topi = torch.topk(probs, k=min(5, probs.numel()))
        response['multiclass_topk'] = [
            {'label_id': int(idx.item()), 'score': float(val.item())} for val, idx in zip(topv, topi)
        ]
    if heads['multilabel'] is not None:
        probs = torch.sigmoid(heads['multilabel'])[0]
        topv, topi = torch.topk(probs, k=min(5, probs.numel()))
        response['multilabel_topk'] = [
            {'label_id': int(idx.item()), 'score': float(val.item())} for val, idx in zip(topv, topi)
        ]
    return response


def discover_model_payload(path: Path) -> Dict[str, Any]:
    metrics_path = path.parent / 'metrics.json'
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    config = {'run_args': metrics.get('run_args', {})} if isinstance(metrics, dict) else {}
    dataset_path = config.get('run_args', {}).get('data', '')
    return {
        'name': path.parent.name,
        'dataset_key': _infer_dataset_key(dataset_path) if dataset_path else '',
        'task_key': Path(dataset_path).stem if dataset_path else path.parent.name,
        'artifact_type': 'dhc',
        'checkpoint_path': str(path),
        'metrics_path': str(metrics_path) if metrics_path.exists() else '',
        'dataset_path': dataset_path,
        'config': config,
        'metrics': metrics.get('test', metrics) if isinstance(metrics, dict) else {},
        'supports_inference': bool(dataset_path and metrics_path.exists() and config.get('run_args')),
    }
