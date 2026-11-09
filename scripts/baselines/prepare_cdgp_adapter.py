#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--name', required=True)
    args = p.parse_args()

    obj = torch.load(args.input, map_location='cpu')
    src = obj['edges']['src'].numpy().astype(int)
    dst = obj['edges']['dst'].numpy().astype(int)
    ts = obj['edges']['ts'].numpy().astype(float)
    eid = np.arange(len(src)).astype(int)

    tr = set(obj['split']['train'].numpy().tolist())
    va = set(obj['split']['val'].numpy().tolist())
    te = set(obj['split']['test'].numpy().tolist())

    rows = []
    for i in range(len(src)):
        if i in tr:
            typ = 1
        elif i in va:
            typ = 2
        elif i in te:
            typ = 3
        else:
            typ = 1
        rows.append({
            'dst': int(dst[i]),
            'cas': int(src[i]),
            'time': float(ts[i]),
            'eid': int(eid[i]),
            'src': int(src[i]),
            'label': 1.0,
            'type': typ,
        })

    df = pd.DataFrame(rows)
    train_df = df[df['type'] == 1].copy()
    val_df = df[df['type'] == 2].copy()
    test_df = df[df['type'] == 3].copy()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with (out_root / f'{args.name}_withlabel.pkl').open('wb') as f:
        pickle.dump({'train': train_df, 'val': val_df, 'test': test_df}, f, -1)

    n_nodes = int(obj['num_nodes'])
    node_feat = np.zeros((n_nodes, 172), dtype=np.float32)
    edge_feat = np.zeros((len(src),), dtype=np.float32)
    np.save(out_root / f'{args.name}_nodefeature.npy', node_feat)
    np.save(out_root / f'{args.name}_edgefeature.npy', edge_feat)

    print(f'CDGP adapter saved to {out_root}')
    print(f'train={len(train_df)} val={len(val_df)} test={len(test_df)} nodes={n_nodes}')


if __name__ == '__main__':
    main()
