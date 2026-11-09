#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


def split_cmd(cmd: str):
    # manifest command is plain space-joined list without shell operators.
    return cmd.strip().split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/home/xuxj/2026/dyn_hetero_community/results/baseline_formal_all/run_manifest.csv')
    ap.add_argument('--project-root', default='/home/xuxj/2026/dyn_hetero_community')
    ap.add_argument('--max-parallel', type=int, default=2)
    ap.add_argument('--poll-sec', type=int, default=10)
    args = ap.parse_args()

    proj = Path(args.project_root)
    out_root = proj / 'results' / 'baseline_formal_all'
    rerun_log_dir = out_root / 'logs_cpu_rerun'
    rerun_log_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    df = df[(df['return_code'] != 0) & (df['family'].isin(['tgb2_custom', 'cdgp']))].copy()
    jobs = []
    for _, r in df.iterrows():
        cmd = split_cmd(str(r['cmd']))
        name = str(r['name'])
        jobs.append(
            {
                'name': name,
                'family': str(r['family']),
                'dataset': str(r['dataset']),
                'task_type': str(r['task_type']),
                'data_path': str(r['data_path']),
                'cmd': cmd,
                'cwd': '/home/xuxj/2026/baselines/src/cdgp/CDGP' if str(r['family']) == 'cdgp' else '/home/xuxj/2026/dyn_hetero_community',
                'out_path': str(r.get('out_path', '')),
                'log_path': str(rerun_log_dir / f"{name}.log"),
            }
        )

    running = []
    queue = list(jobs)
    finished = []

    print(f'[rerun] jobs={len(queue)}')
    while queue or running:
        while queue and len(running) < args.max_parallel:
            job = queue.pop(0)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            env['OMP_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            env['OPENBLAS_NUM_THREADS'] = '1'
            env['NUMEXPR_NUM_THREADS'] = '1'
            log_f = Path(job['log_path']).open('w', encoding='utf-8')
            print(f"[rerun] launch {job['name']} on CPU")
            p = subprocess.Popen(job['cmd'], cwd=job['cwd'], env=env, stdout=log_f, stderr=subprocess.STDOUT)
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
            print(f"[rerun] done {it['job']['name']} rc={rc} elapsed={elapsed:.1f}s")
            finished.append(
                {
                    'name': it['job']['name'],
                    'family': it['job']['family'],
                    'dataset': it['job']['dataset'],
                    'task_type': it['job']['task_type'],
                    'data_path': it['job']['data_path'],
                    'return_code': int(rc),
                    'elapsed_sec': elapsed,
                    'log_path': it['job']['log_path'],
                    'out_path': it['job']['out_path'],
                    'cmd': ' '.join(it['job']['cmd']),
                    'run_mode': 'cpu_rerun',
                }
            )
        running = nr

    out_csv = out_root / 'run_manifest_cpu_rerun.csv'
    with out_csv.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'family', 'dataset', 'task_type', 'data_path', 'return_code', 'elapsed_sec', 'log_path', 'out_path', 'cmd', 'run_mode'])
        w.writeheader()
        for r in finished:
            w.writerow(r)
    print(f'[rerun] saved {out_csv}')


if __name__ == '__main__':
    main()
