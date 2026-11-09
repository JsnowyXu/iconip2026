#!/usr/bin/env python3
"""Validate raw JSONL ZIP -> features -> task views -> baseline adapters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import torch

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'data/smoke/preprocess';base.mkdir(parents=True,exist_ok=True)
archive=base/'dblp.zip'
records=[{'id':str(i),'title':f'Graph paper {i}','venue':'Test', 'year':2000+i//10,
          'authors':[{'id':f'a{i%8}','name':f'Author {i%8}'}],
          'references':[str(max(0,i-1)),str(max(0,i-2))]} for i in range(100)]
with zipfile.ZipFile(archive,'w') as z:
    z.writestr('DBLP-Citation-network-V18.jsonl','\n'.join(json.dumps(x) for x in records))
env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
for args in [
    ['scripts/datasets/prepare_dataset.py','--dataset','dblp','--archive',str(archive),
     '--max-edges','300','--out-pt',str(base/'unified.pt'),'--export-baselines',
     '--baseline-out-dir',str(base/'adapters')],
    ['scripts/datasets/build_paper_task_views.py','--input',str(base/'unified.pt'),
     '--out-dir',str(base/'tasks'),'--name-prefix','dblp','--target-rel','1']]:
    subprocess.run([sys.executable,*args],cwd=ROOT,env=env,check=True)
for name in ['dblp_edge_inductive.pt','dblp_edge_transductive.pt']:
    obj=torch.load(base/'tasks'/name,map_location='cpu')
    joined=torch.cat([obj['split'][x] for x in ['train','val','test']])
    assert joined.unique().numel()==joined.numel()
    assert all(obj['split'][x].numel() for x in ['train','val','test'])
    assert obj['node_feat'].shape[1]==128
print('PASS: raw ZIP parsing, 128-D features, target/context task splits, and baseline export')
