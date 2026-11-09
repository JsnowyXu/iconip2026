#!/usr/bin/env python3
"""Audit actual unified/task tensors without mutating them."""
import json
import hashlib
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    def compact(o):
        if isinstance(o, dict): return {k:compact(v) for k,v in o.items()}
        if isinstance(o, list):
            if len(o)>50:
                return {'count':len(o),'sha256':hashlib.sha256(json.dumps(o,separators=(',',':')).encode()).hexdigest(),
                        'full_values':'Original task .pt metadata'}
            return [compact(v) for v in o]
        return o
    rows=[]
    for path in sorted((ROOT/'data/processed').rglob('*.pt')):
        obj=torch.load(path,map_location='cpu')
        if not isinstance(obj,dict) or 'edges' not in obj or 'split' not in obj:continue
        edge=obj['edges'];split=obj['split'];ts=edge['ts'];n=int(obj['num_nodes'])
        indices=[split[k].long() for k in ['train','val','test']]
        joined=torch.cat(indices)
        disjoint=joined.unique().numel()==joined.numel()
        in_bounds=bool(((joined>=0)&(joined<len(ts))).all())
        row={'path':str(path.relative_to(ROOT)),'nodes':n,'edges':len(ts),
             'num_rels':obj['num_rels'],'node_feature_dim':obj['node_feat'].shape[1],
             'disjoint':disjoint,'indices_in_bounds':in_bounds,
             'split':{k:{'events':len(idx),'supervised':int((edge.get('edge_label',torch.ones_like(ts))[idx]>.5).sum()),
                         'time_ordered':bool((ts[idx][1:]>=ts[idx][:-1]).all())} for k,idx in zip(['train','val','test'],indices)},
             'context_protocol':obj.get('meta',{}).get('extra',{}).get('context_protocol',{})}
        rows.append(row)
    out=ROOT/'repro/data_audit.json';out.write_text(json.dumps(compact(rows),indent=2,ensure_ascii=False,default=str)+'\n')
    print(f'Audited {len(rows)} tensors -> {out}')
    if any(not x['disjoint'] or not x['indices_in_bounds'] for x in rows):raise SystemExit('Invalid split indices detected')

if __name__=='__main__':main()
