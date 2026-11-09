#!/usr/bin/env python3
"""Run the existing raw statistics scripts with relocatable explicit inputs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',required=True,choices=['patent_us','patent_cn','patent_in','thgl_software','thgl_github','dblp','amazon_movies','aminer_paper'])
    a=p.parse_args();raw=ROOT/'raw_datasets';out=ROOT/'data/reports/raw_stats'
    out.mkdir(parents=True,exist_ok=True)
    script=ROOT/'scripts/datasets/raw_stats'/f"{'thgl_software' if a.dataset=='thgl_github' else a.dataset}_raw_stats.py"
    if a.dataset in {'patent_us','patent_cn'}:args=['--dataset-dir',str(raw/a.dataset),'--json']
    elif a.dataset=='patent_in':args=['--dataset-dir',str(raw/a.dataset),'--json-output',str(out/'patent_in_new.json')]
    elif a.dataset=='dblp':args=['--input',str(raw/'DBLP-Citation-network-V18.zip'),'--json']
    elif a.dataset=='amazon_movies':args=['--archive',str(raw/'Amazon_movies.zip'),'--out',str(out/'amazon_movies_new.json')]
    elif a.dataset=='aminer_paper':
        args=[]
        for flag,name in [('paper','Paper'),('author','Author'),('author2paper','Author2Paper'),('coauthor','Coauthor')]:
            args += [f'--{flag}-archive',str(raw/'Aminer'/(f'AMiner-{name}.'+('rar' if flag=='paper' else 'zip')))]
        args+=['--out',str(out/'aminer_paper_new.json')]
    else:
        args=[]
        for flag,name in [('edgelist','edgelist'),('nodetype','nodetype'),('edgemapping','edgemapping'),('nodemapping','nodemapping')]:
            args += [f'--{flag}-csv',str(raw/'tgb2'/a.dataset.replace('_','-')/f"{a.dataset.replace('_','-')}_{name}.csv")]
        args+=['--out-json',str(out/f'{a.dataset}_new.json')]
    with (out/f'{a.dataset}_stdout.txt').open('w') as f:
        subprocess.run([sys.executable,str(script),*args],cwd=ROOT,stdout=f,env=dict(os.environ,PYTHONPATH=str(ROOT)),check=True)
    print(f'Saved raw statistics for {a.dataset} to {out}')

if __name__=='__main__':main()
