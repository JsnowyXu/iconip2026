#!/usr/bin/env python3
"""Exercise actual training -> complete checkpoint -> stateful agent ranking."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    torch.manual_seed(42)
    data_dir=ROOT/'data/smoke';data_dir.mkdir(parents=True,exist_ok=True)
    n,e=16,48
    obj={'num_nodes':n,'num_rels':2,'num_node_types':2,
         'node_feat':torch.randn(n,8),'node_type':torch.arange(n)%2,
         'edges':{'src':torch.arange(e)%n,'dst':(torch.arange(e)+3)%n,
                  'rel':torch.arange(e)%2,'ts':torch.arange(e).float()+1,
                  'edge_feat':torch.randn(e,2),'edge_label':torch.ones(e)},
         'split':{'train':torch.arange(32),'val':torch.arange(32,40),'test':torch.arange(40,48)},
         'meta':{'dataset_name':'smoke_edge_transductive'}}
    path=data_dir/'graph.pt';torch.save(obj,path)
    run=ROOT/'results/smoke'
    env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
    subprocess.run([sys.executable,'-m','dhc.train.run_train','--data',str(path),
                    '--out-dir',str(run),'--epochs','2','--batch-size','8',
                    '--mem-dim','16','--rel-dim','8','--time-dim','8','--msg-dim','16',
                    '--community-dim','16','--num-communities','4','--recent-neighbor-size','3'],
                   cwd=ROOT,env=env,check=True)
    from inference import DHCTool
    tool=DHCTool(run,history='train')
    result=tool.rank_links(0,[1,2,3],0,40.,top_k=2)
    assert len(result['ranked'])==2
    assert result==tool.rank_links(0,[1,2,3],0,40.,top_k=2)
    before=tool.model.memory.node_memory.clone()
    tool.observe(0,3,0,40.,[0.,1.])
    assert not torch.equal(before,tool.model.memory.node_memory)
    for callback in [lambda:tool.rank_links(n,[1],0,41.),lambda:tool.rank_links(0,[1],0,1.)]:
        try: callback()
        except ValueError: pass
        else: raise AssertionError('Invalid request was accepted')
    weights=torch.load(run/'best_model.pt',map_location='cpu')
    assert 'node_type_emb.weight' in weights
    assert 'memory.node_updater.weight_ih' in weights
    assert 'memory.rel_updater.weight_ih' in weights
    assert 'memory.node_memory' not in weights
    print(json.dumps({'status':'PASS','training_epochs':2,'ranking':result,'checkpoint_contains_memory_grus':True}))

if __name__=='__main__':main()
