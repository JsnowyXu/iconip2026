#!/usr/bin/env python3
"""Portable entrypoint. Every subprocess uses the current Python interpreter."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    print(' '.join(map(str, [sys.executable, *args])), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, check=True)


def preprocess(dataset):
    raw = ROOT / 'raw_datasets'
    processed = ROOT / 'data/processed'
    if dataset.startswith('patent_'):
        run(['scripts/experiments/run_patent_preprocessing.py', '--project-root', ROOT,
             '--dataset-root', raw, '--python-bin', sys.executable, '--datasets', dataset])
        return
    output = processed / f'{dataset}_unified.pt'
    if dataset.startswith('thgl_'):
        name = dataset.replace('_', '-')
        run(['scripts/datasets/import_tgb2_thg.py', '--dataset-name', dataset,
             '--edgelist-csv', raw/'tgb2'/name/f'{name}_edgelist.csv',
             '--nodetype-csv', raw/'tgb2'/name/f'{name}_nodetype.csv', '--out-pt', output, '--max-edges', '-1'])
    else:
        archive = {'aminer_paper':raw/'Aminer/AMiner-Paper.rar', 'dblp':raw/'DBLP-Citation-network-V18.zip',
                   'amazon_movies':raw/'Amazon_movies.zip'}[dataset]
        args = ['scripts/datasets/prepare_dataset.py', '--dataset', dataset, '--archive', archive,
                '--out-pt', output, '--max-edges', '200000', '--export-baselines',
                '--baseline-out-dir', ROOT/'data/interim/baseline_adapters'/dataset]
        if dataset == 'aminer_paper':
            args += ['--aminer-min-year', '2002', '--aminer-max-year', '2012']
        run(args)
    args = ['scripts/datasets/build_paper_task_views.py', '--input', output,
            '--out-dir', processed/('tgb2_tasks' if dataset.startswith('thgl_') else 'paper_tasks'),
            '--name-prefix', 'aminer_paper_citation' if dataset == 'aminer_paper' else dataset]
    if dataset in {'aminer_paper', 'dblp', 'thgl_software'}:
        args += ['--target-rel', str({'aminer_paper':0, 'dblp':1, 'thgl_software':6}[dataset])]
    run(args)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['list', 'preprocess', 'train'])
    p.add_argument('--dataset', default='patent_us', choices=['patent_us','patent_cn','patent_in','aminer_paper','dblp','thgl_software','thgl_github','amazon_movies'])
    p.add_argument('--config', help='JSON path under repro/configs (required for train)')
    p.add_argument('--epochs', type=int)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--allow-historical-overlap', action='store_true', help='Explicitly permit an invalid archived split for historical diagnostics only')
    args = p.parse_args()
    if args.stage == 'list':
        for path in sorted((ROOT/'repro/configs').glob('*.json')): print(path.relative_to(ROOT))
    elif args.stage == 'preprocess': preprocess(args.dataset)
    else:
        if not args.config: p.error('--config required for train')
        config = json.loads((ROOT/args.config).read_text())
        values = config['run_args'].copy()
        values['data'] = str(ROOT/values['data'])
        import torch
        obj = torch.load(values['data'], map_location='cpu')
        split = torch.cat([obj['split'][name] for name in ['train','val','test']])
        if split.unique().numel() != split.numel() and not args.allow_historical_overlap:
            raise ValueError('Overlapping archived split indices: rebuild this dataset with preprocess first. '
                             'For historical diagnostics only, explicitly pass --allow-historical-overlap.')
        del obj, split
        values['out_dir'] = str(ROOT/'results/reproduced'/Path(args.config).stem)
        values['gpu_id'] = args.gpu_id
        if args.epochs is not None: values['epochs'] = args.epochs
        command = ['-m', 'dhc.train.run_train']
        for key, value in values.items():
            flag = '--' + key.replace('_','-')
            if isinstance(value, bool):
                if value: command.append(flag)
            else: command.extend([flag, str(value)])
        run(command)

if __name__ == '__main__': main()
