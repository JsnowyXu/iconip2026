import os
from pathlib import Path
from typing import Dict, Optional

from django.conf import settings


def repo_root() -> Path:
    return Path(settings.DHC_PLATFORM['REPO_ROOT']).resolve()


def runtime_root() -> Path:
    root = Path(settings.DHC_PLATFORM['RUNTIME_ROOT']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def results_root() -> Path:
    return Path(settings.DHC_PLATFORM['RESULTS_ROOT']).resolve()


def data_root() -> Path:
    return Path(settings.DHC_PLATFORM['DATA_ROOT']).resolve()


def ensure_runtime_dirs() -> dict:
    root = runtime_root()
    dirs = {
        'root': root,
        'logs': root / 'logs',
        'scripts': root / 'generated_scripts',
        'outputs': root / 'outputs',
    }
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return dirs


def build_python_env(extra: Optional[Dict[str, str]] = None) -> dict:
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{repo_root() / 'src'}:{env.get('PYTHONPATH', '')}"
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('NUMEXPR_NUM_THREADS', '1')
    env.setdefault('PYTHONUNBUFFERED', '1')
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env
