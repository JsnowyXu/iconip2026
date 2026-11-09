#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


def tail_last_line(path: Path) -> str:
    if not path.exists():
        return ''
    try:
        with path.open('rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                return ''
            off = min(size, 4096)
            f.seek(-off, os.SEEK_END)
            txt = f.read().decode('utf-8', errors='ignore')
        lines = [x.strip() for x in txt.splitlines() if x.strip()]
        return lines[-1] if lines else ''
    except Exception:
        return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', default='/home/xuxj/2026/dyn_hetero_community')
    ap.add_argument('--python-bin', default='python')
    ap.add_argument('--manifest', default='/home/xuxj/2026/dyn_hetero_community/results/baseline_formal_all/run_manifest_final.csv')
    ap.add_argument('--max-parallel', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-edges', type=int, default=30000)
    ap.add_argument('--poll-sec', type=int, default=20)
    args = ap.parse_args()

    proj = Path(args.project_root)
    out_root = proj / 'results' / 'baseline_formal_all'
    metric_root = out_root / 'metrics'
    log_root = out_root / 'logs_cawn_fix'
    metric_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest)
    m = m[m['family'] == 'dyglib_model_zoo'].copy()
    jobs = []
    gpus = [0, 1, 2, 3]
    for i, (_, r) in enumerate(m.iterrows()):
        name = str(r['name'])
        task = name.split('__', 1)[-1]
        out_csv = metric_root / f'cawn_rerun__{task}.csv'
        jobs.append(
            {
                'name': f'cawn_rerun__{task}',
                'dataset': str(r['dataset']),
                'task_type': str(r['task_type']),
                'data_path': str(r['data_path']),
                'gpu': gpus[i % len(gpus)],
                'out_path': str(out_csv),
                'log_path': str(log_root / f'cawn_rerun__{task}.log'),
                'cmd': [
                    args.python_bin,
                    str(proj / 'scripts' / 'baselines' / 'run_dyglib_model_zoo_smoke.py'),
                    '--data',
                    str(r['data_path']),
                    '--models',
                    'CAWN',
                    '--epochs',
                    str(args.epochs),
                    '--batch-size',
                    str(args.batch_size),
                    '--max-edges',
                    str(args.max_edges),
                    '--gpu-id',
                    '0',
                    '--out',
                    str(out_csv),
                ],
            }
        )

    queue = list(jobs)
    running = []
    finished = []

    print(f'[cawn-rerun] total_jobs={len(queue)}')
    while queue or running:
        while queue and len(running) < args.max_parallel:
            job = queue.pop(0)
            env = os.environ.copy()
            env.pop('LD_LIBRARY_PATH', None)
            env['CUDA_VISIBLE_DEVICES'] = str(job['gpu'])
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            env['OMP_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            env['OPENBLAS_NUM_THREADS'] = '1'
            env['NUMEXPR_NUM_THREADS'] = '1'
            log_f = Path(job['log_path']).open('w', encoding='utf-8')
            print(f"[cawn-rerun] launch {job['name']} gpu={job['gpu']}")
            p = subprocess.Popen(job['cmd'], cwd=str(proj), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append({'job': job, 'proc': p, 'log_f': log_f, 'start': time.time()})

        time.sleep(max(1, args.poll_sec))
        nr = []
        for it in running:
            rc = it['proc'].poll()
            if rc is None:
                nr.append(it)
                continue
            it['log_f'].close()
            elapsed = float(time.time() - it['start'])
            print(f"[cawn-rerun] done {it['job']['name']} rc={rc} elapsed={elapsed:.1f}s")
            finished.append(
                {
                    'name': it['job']['name'],
                    'dataset': it['job']['dataset'],
                    'task_type': it['job']['task_type'],
                    'data_path': it['job']['data_path'],
                    'gpu': it['job']['gpu'],
                    'return_code': int(rc),
                    'elapsed_sec': elapsed,
                    'log_path': it['job']['log_path'],
                    'out_path': it['job']['out_path'],
                    'cmd': ' '.join(it['job']['cmd']),
                }
            )
        running = nr
        if running:
            print(f"[cawn-rerun] active={len(running)} queued={len(queue)}")
            for it in running:
                tail = tail_last_line(Path(it['job']['log_path']))
                if tail:
                    print(f"  - {it['job']['name']}: {tail}")

    man = out_root / 'run_manifest_cawn_rerun.csv'
    with man.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'dataset', 'task_type', 'data_path', 'gpu', 'return_code', 'elapsed_sec', 'log_path', 'out_path', 'cmd'])
        w.writeheader()
        for r in finished:
            w.writerow(r)
    print(f'[cawn-rerun] saved {man}')


if __name__ == '__main__':
    main()
