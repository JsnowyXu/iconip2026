#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    errors=[]
    for item in json.loads((ROOT/'repro/artifact_manifest.json').read_text()):
        path=ROOT/item['path']
        if not path.is_file():
            errors.append(str(path)+': missing');continue
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        if path.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:
            errors.append(str(path)+': mismatch')
    if errors: raise SystemExit('\n'.join(errors))
    print('All archived artifacts verified.')

if __name__=='__main__':main()
