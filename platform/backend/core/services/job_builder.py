import json
import shlex
from pathlib import Path
from typing import Dict, List, Tuple

from core.models import PlatformJob

from .project import ensure_runtime_dirs, repo_root
from .registry import DATASET_CONFIG


def _quote_cmd(parts: List[str]) -> str:
    return ' '.join(shlex.quote(str(p)) for p in parts)


def _write_script(job: PlatformJob, lines: List[str]) -> Tuple[str, str]:
    dirs = ensure_runtime_dirs()
    script_path = dirs['scripts'] / f'job_{job.id:05d}.sh'
    log_path = dirs['logs'] / f'job_{job.id:05d}.log'
    content = ['#!/usr/bin/env bash', 'set -euo pipefail', f'cd {shlex.quote(str(repo_root()))}'] + lines
    script_path.write_text('\n'.join(content) + '\n', encoding='utf-8')
    script_path.chmod(0o755)
    return str(script_path), str(log_path)


def configure_preprocess_job(job: PlatformJob) -> PlatformJob:
    payload = job.payload
    dataset_key = payload['dataset_key']
    dataset_cfg = DATASET_CONFIG[dataset_key]
    mode = payload.get('mode', 'prepare_and_build_tasks')
    archive_path = payload.get('archive_path', '').strip()
    max_edges = int(payload.get('max_edges', 120000))
    text_encoder = payload.get('text_encoder', 'hybrid_hash')
    text_dim = int(payload.get('text_dim', 64))
    text_model_name = payload.get('text_model_name', '').strip()
    text_batch_size = int(payload.get('text_batch_size', 128))
    dirs = ensure_runtime_dirs()
    processed_dir = repo_root() / 'data' / 'processed' / 'platform'
    task_dir = repo_root() / 'data' / 'processed' / 'platform_tasks'
    processed_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    unified_path = processed_dir / f'{dataset_key}_platform_job{job.id}.pt'
    commands: List[List[str]] = []
    if mode in {'prepare_unified', 'prepare_and_build_tasks'}:
        if not archive_path:
            raise ValueError('archive_path is required for prepare_unified mode')
        cmd = [
            'python', 'scripts/datasets/prepare_dataset.py',
            '--dataset', dataset_key,
            '--archive', archive_path,
            '--max-edges', str(max_edges),
            '--out-pt', str(unified_path),
            '--text-encoder', text_encoder,
            '--text-dim', str(text_dim),
            '--text-batch-size', str(text_batch_size),
        ]
        if text_model_name:
            cmd += ['--text-model-name', text_model_name]
        commands.append(cmd)
    else:
        unified_path = Path(payload['input_unified_path']).resolve()

    if mode in {'build_tasks', 'prepare_and_build_tasks'}:
        builder = dataset_cfg['builder']
        if builder == 'paper':
            cmd = [
                'python', 'scripts/datasets/build_paper_task_views.py',
                '--input', str(unified_path),
                '--out-dir', str(task_dir),
                '--name-prefix', dataset_key,
                '--target-rel', str(int(payload.get('target_rel', -1))),
            ]
        else:
            cmd = [
                'python', 'scripts/datasets/build_task_views.py',
                '--input', str(unified_path),
                '--out-dir', str(task_dir),
                '--name-prefix', dataset_key,
            ]
        commands.append(cmd)

    script_path, log_path = _write_script(job, [
        'export PYTHONPATH="%s:$PYTHONPATH"' % (repo_root() / 'src'),
        *[_quote_cmd(cmd) for cmd in commands],
    ])
    job.command = ['bash', script_path]
    job.command_preview = '\n'.join(_quote_cmd(cmd) for cmd in commands)
    job.script_path = script_path
    job.log_path = log_path
    job.working_directory = str(repo_root())
    job.output_path = str(task_dir if mode != 'prepare_unified' else unified_path)
    return job


def configure_training_job(job: PlatformJob) -> PlatformJob:
    payload = job.payload
    data_path = Path(payload['data_path']).resolve()
    run_name = payload.get('run_name', f"{data_path.stem}_platform_job{job.id}")
    out_dir = repo_root() / 'results' / 'platform_runs' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        'python', '-m', 'dhc.train.run_train',
        '--data', str(data_path),
        '--out-dir', str(out_dir),
        '--seed', str(int(payload.get('seed', 42))),
        '--epochs', str(int(payload.get('epochs', 5))),
        '--batch-size', str(int(payload.get('batch_size', 512))),
        '--lr', str(float(payload.get('lr', 0.001))),
        '--gpu-id', str(int(payload.get('gpu_id', 0))),
        '--early-stop-patience', str(int(payload.get('early_stop_patience', 5))),
        '--recent-neighbor-size', str(int(payload.get('recent_neighbor_size', 12))),
        '--lambda-cl', str(float(payload.get('lambda_cl', 0.05))),
        '--lambda-ml', str(float(payload.get('lambda_ml', 0.7))),
        '--community-assign-temperature', str(float(payload.get('community_assign_temperature', 1.0))),
        '--community-assign-topk', str(int(payload.get('community_assign_topk', 0))),
        '--typed-negative-ratio', str(float(payload.get('typed_negative_ratio', 0.6))),
        '--popular-negative-ratio', str(float(payload.get('popular_negative_ratio', 0.2))),
    ]
    if payload.get('ml_threshold_search', True):
        args.append('--ml-threshold-search')
    if payload.get('ml_pos_weight', True):
        args.append('--ml-pos-weight')

    script_path, log_path = _write_script(job, [
        'export PYTHONPATH="%s:$PYTHONPATH"' % (repo_root() / 'src'),
        _quote_cmd(args),
    ])
    job.command = ['bash', script_path]
    job.command_preview = _quote_cmd(args)
    job.script_path = script_path
    job.log_path = log_path
    job.working_directory = str(repo_root())
    job.output_path = str(out_dir)
    return job
