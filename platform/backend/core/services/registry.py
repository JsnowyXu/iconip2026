import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .project import data_root, repo_root, results_root


DATASET_CONFIG = {
    'arxiv': {
        'label': 'arXiv Citation Graph',
        'category': 'Academic Network',
        'description': '论文引用时序网络，适合链路预测与节点多标签分类。',
        'report_dir': 'arxiv',
        'unified_path': 'data/processed/arxiv_unified.pt',
        'tasks': ['edge_transductive', 'edge_inductive', 'node_multilabel'],
        'builder': 'paper',
    },
    'dblp': {
        'label': 'DBLP V18',
        'category': 'Academic Heterogeneous Graph',
        'description': '作者-论文异质时序网络，适合关系预测、节点分类与表示学习。',
        'report_dir': 'dblp_v18',
        'unified_path': 'data/processed/dblp_unified.pt',
        'tasks': ['edge_transductive', 'edge_inductive', 'node_multilabel'],
        'builder': 'paper',
    },
    'amazon_movies': {
        'label': 'Amazon Movies',
        'category': 'Recommendation Graph',
        'description': '电影交互与标签网络，适合推荐式链路预测和多标签任务。',
        'report_dir': 'amazon_movies',
        'unified_path': 'data/processed/amazon_movies_unified.pt',
        'tasks': ['edge_transductive', 'edge_inductive', 'node_multilabel'],
        'builder': 'paper',
    },
    'kg4ai': {
        'label': 'KG4AI',
        'category': 'Knowledge Graph',
        'description': 'AI 领域知识图谱，适合异构关系建模与下游推理。',
        'report_dir': 'kg4ai',
        'unified_path': 'data/processed/kg4ai_unified.pt',
        'tasks': ['edge_transductive'],
        'builder': 'paper',
    },
    'raw_cn': {
        'label': 'Chinese Patent',
        'category': 'Patent Graph',
        'description': '中文专利-公司-IPC 异构动态图，突出文本属性与社区演化。',
        'report_dir': 'raw_cn',
        'unified_path': 'data/processed/raw_cn_unified.pt',
        'tasks': ['edge_patent_ipc', 'edge_assignee_ipc_transductive', 'edge_assignee_ipc_inductive', 'node_patent_multilabel'],
        'builder': 'patent',
    },
    'raw_us': {
        'label': 'US Patent',
        'category': 'Patent Graph',
        'description': '美国专利-公司-IPC 异构动态图，适合跨关系增强和社区演化分析。',
        'report_dir': 'raw_us',
        'unified_path': 'data/processed/raw_us_unified.pt',
        'tasks': ['edge_patent_ipc', 'edge_assignee_ipc_transductive', 'edge_assignee_ipc_inductive', 'node_patent_multilabel'],
        'builder': 'patent',
    },
}

TASK_LABELS = {
    'edge_transductive': '边预测 Transductive',
    'edge_inductive': '边预测 Inductive',
    'node_multilabel': '多标签分类',
    'edge_patent_ipc': '专利-IPC 预测',
    'edge_assignee_ipc_transductive': '公司-IPC 预测 Transductive',
    'edge_assignee_ipc_inductive': '公司-IPC 预测 Inductive',
    'node_patent_multilabel': '专利 IPC 多标签分类',
}


def format_bytes(num: int) -> str:
    value = float(num)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if value < 1024 or unit == 'TB':
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{num} B'


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _load_distribution(path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def discover_dataset_catalog() -> List[Dict[str, Any]]:
    catalog = []
    for key, cfg in DATASET_CONFIG.items():
        unified = repo_root() / cfg['unified_path']
        report_dir = data_root() / 'reports' / cfg['report_dir']
        stats = _load_json(report_dir / 'stats.json')
        task_files = list_task_files(key)
        catalog.append(
            {
                'key': key,
                'label': cfg['label'],
                'category': cfg['category'],
                'description': cfg['description'],
                'tasks': [{'key': t, 'label': TASK_LABELS.get(t, t)} for t in cfg['tasks']],
                'builder': cfg['builder'],
                'stats': stats,
                'unified_path': str(unified),
                'has_unified': unified.exists(),
                'unified_size': format_bytes(unified.stat().st_size) if unified.exists() else 'N/A',
                'task_file_count': len(task_files),
                'task_files': task_files,
            }
        )
    return catalog


def list_task_files(dataset_key: str) -> List[Dict[str, Any]]:
    root = repo_root()
    out = []
    for folder in [
        root / 'data' / 'processed' / 'paper_tasks',
        root / 'data' / 'processed' / 'tasks',
        root / 'data' / 'processed' / 'platform_tasks',
    ]:
        if not folder.exists():
            continue
        for path in sorted(folder.glob(f'{dataset_key}_*.pt')):
            out.append(
                {
                    'name': path.stem,
                    'path': str(path),
                    'size': format_bytes(path.stat().st_size),
                }
            )
    return out


def get_dataset_detail(dataset_key: str) -> Optional[Dict[str, Any]]:
    for item in discover_dataset_catalog():
        if item['key'] == dataset_key:
            stats_dir = results_root() / 'dataset_statistics'
            item['distributions'] = {
                'class_distribution': _load_distribution(stats_dir / f'class_distribution_{dataset_key}.csv'),
                'split_distribution': _load_distribution(stats_dir / f'class_distribution_split_{dataset_key}.csv'),
            }
            item['report_path'] = str(data_root() / 'reports' / DATASET_CONFIG[dataset_key]['report_dir'] / 'stats.json')
            return item
    return None


def pipeline_options() -> Dict[str, Any]:
    return {
        'datasets': discover_dataset_catalog(),
        'text_encoders': [
            {'key': 'hybrid_hash', 'label': 'Hybrid Hash', 'description': '适合中英文混合文本的轻量编码。'},
            {'key': 'sentence_transformer', 'label': 'Sentence Transformer', 'description': '适合高质量语义编码，尤其是中文专利。'},
            {'key': 'none', 'label': 'No Text Encoder', 'description': '仅结构特征。'},
        ],
        'pipeline_modes': [
            {'key': 'prepare_unified', 'label': '生成统一图数据'},
            {'key': 'build_tasks', 'label': '生成任务视图'},
            {'key': 'prepare_and_build_tasks', 'label': '统一图 + 任务视图一体化'},
        ],
    }


def training_options() -> Dict[str, Any]:
    task_candidates: List[Dict[str, Any]] = []
    for dataset in discover_dataset_catalog():
        for task_file in dataset['task_files']:
            task_candidates.append(
                {
                    'dataset_key': dataset['key'],
                    'dataset_label': dataset['label'],
                    'task_name': task_file['name'],
                    'data_path': task_file['path'],
                    'size': task_file['size'],
                }
            )
    return {
        'task_candidates': task_candidates,
        'defaults': {
            'epochs': 5,
            'batch_size': 512,
            'lr': 0.001,
            'gpu_id': 0,
            'recent_neighbor_size': 12,
            'lambda_cl': 0.05,
            'lambda_ml': 0.7,
            'community_assign_temperature': 1.0,
            'community_assign_topk': 0,
            'ml_threshold_search': True,
            'typed_negative_ratio': 0.6,
            'popular_negative_ratio': 0.2,
        },
    }
