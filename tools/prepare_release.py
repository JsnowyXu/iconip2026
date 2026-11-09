#!/usr/bin/env python3
"""Create relocatable configs, evidence tables, and file checksums."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = '/home/xuxj/2026/dyn_relation_memory/'


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--skip-hash',action='store_true')
    parser.add_argument('--folders',nargs='+',default=['data','raw_datasets','results','vendor'])
    parser.add_argument('--append',action='store_true')
    parser.add_argument('--output',default='repro/artifact_manifest.json')
    options=parser.parse_args()
    target = ROOT/'repro/configs'
    target.mkdir(parents=True, exist_ok=True)
    sources = {p.parent: 'main' for p in (ROOT/'results/edge_only_formal/main').glob('*/metrics.json')}
    for folder in ['targeted_edge_tuning_20260423','targeted_edge_tuning_20260423_followup']:
        for metrics in (ROOT/'results'/folder).glob('*/metrics.json'):
            sources[metrics.parent] = 'targeted_search'
    catalog = ROOT/'results/best_config_module_ablation/best_run_catalog.csv'
    if catalog.exists():
        for row in csv.DictReader(catalog.open()):
            sources[ROOT/row['run_dir'].replace(OLD,'')] = 'catalog_best_test_mrr'
    rows=[]
    selected={}
    for directory, group in sorted(sources.items()):
        metrics = directory/'metrics.json'
        if not metrics.exists(): continue
        obj=json.loads(metrics.read_text())
        args=obj.get('run_args')
        if not args:
            args=json.loads((directory/'model_stats.json').read_text())['args']
        args={k:(v.replace(OLD,'') if isinstance(v,str) else v) for k,v in args.items()}
        payload={'source_group':group,'source_run':str(directory.relative_to(ROOT)),
                 'selection_note':'Historical catalog uses test MRR; not an unbiased validation-only selection.' if group=='catalog_best_test_mrr' else 'Original campaign configuration.',
                 'best_val':obj.get('best_val'),
                 'best_epoch':obj.get('best_epoch'),'monitor_metric':obj.get('monitor_metric'),
                 'test':obj.get('test'),'run_args':args}
        (target/f'{directory.name}.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
        task=Path(args['data']).stem
        val_mrr=obj.get('best_val',{}).get('val_mrr')
        if val_mrr is not None and (task not in selected or val_mrr > selected[task]['val_mrr']):
            selected[task]={'config':str((target/f'{directory.name}.json').relative_to(ROOT)),
                            'val_mrr':val_mrr,'source_run':str(directory.relative_to(ROOT)),
                            'note':'Highest recorded best-epoch validation MRR within archived main/tuning runs; historical search may already have inspected test.'}
        rows.append({'run':directory.name,'group':group,'best_epoch':obj.get('best_epoch'),**obj.get('test',{})})
    (ROOT/'repro/best_validation.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n')
    if rows:
        with (ROOT/'repro/results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['run','group','best_epoch','auc','ap','acc','mrr']);w.writeheader();w.writerows(rows)
    if options.skip_hash:
        print(f'{len(rows)} configurations exported')
        return
    manifest_path=ROOT/options.output
    files=json.loads(manifest_path.read_text()) if options.append and manifest_path.exists() else []
    files=[item for item in files if item['path'].split('/',1)[0] not in options.folders]
    print('Hashing archived data/results/vendor files...',flush=True)
    for folder in options.folders:
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or path.is_symlink() or '.git' in path.parts or 'node_modules' in path.parts or '__pycache__' in path.parts: continue
            if path.suffix in {'.pyc'}:continue
            if '.git' in path.parts:continue
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            if len(files)%100==0:print(f'Hashed {len(files)} files',flush=True)
            files.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    manifest_path.write_text(json.dumps(files,indent=2,ensure_ascii=False)+'\n')
    print(f'{len(rows)} configurations; {len(files)} hashed artifacts')

if __name__=='__main__':main()
