#!/usr/bin/env python3
"""Adapt original baseline directory contract inside an ignored staging folder."""
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
stage=root/'.campaign_workspace'
(stage/'baselines').mkdir(parents=True,exist_ok=True)
for link,target in [(stage/'datasets',root/'raw_datasets'),
                    (stage/'baselines/src',root/'vendor/baselines'),
                    (stage/'baselines/ref',root/'vendor/ref')]:
    if not link.exists():link.symlink_to(target,target_is_directory=True)
subprocess.run([sys.executable,str(root/'scripts/experiments/run_edge_only_campaign.py'),
                '--project-root',str(root),'--repo-root',str(stage),'--python-bin',sys.executable,
                *sys.argv[1:]],cwd=root,env=dict(os.environ,PYTHONPATH=str(root)),check=True)
