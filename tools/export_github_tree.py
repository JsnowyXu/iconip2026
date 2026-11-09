#!/usr/bin/env python3
"""Export reviewed staged UTF-8 source files in bounded Git Trees API batches."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--batch',type=int)
    p.add_argument('--limit',type=int,default=350000)
    a=p.parse_args()
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0')
    batches=[];batch=[];size=0
    for name in names:
        if not name:continue
        path=ROOT/name
        if path.is_symlink():
            raise ValueError(f'Symlink not suitable for this text export: {name}')
        content=path.read_bytes().decode('utf-8')
        item={'path':name,'mode':'100755' if path.stat().st_mode & 0o111 else '100644','type':'blob','content':content}
        encoded=len(json.dumps(item,ensure_ascii=False).encode())
        if batch and size+encoded>a.limit:batches.append(batch);batch=[];size=0
        batch.append(item);size+=encoded
    if batch:batches.append(batch)
    if a.batch is None:
        print(json.dumps({'batches':len(batches),'files':sum(map(len,batches)),
                          'bytes':[sum(len(x['content'].encode()) for x in b) for b in batches]}))
    else:print(json.dumps(batches[a.batch],ensure_ascii=False))

if __name__=='__main__':main()
