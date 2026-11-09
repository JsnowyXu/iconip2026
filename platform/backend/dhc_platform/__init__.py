import sys
from pathlib import Path

import pymysql

pymysql.install_as_MySQLdb()

# Ensure the main project package under `src/` is importable for API inference.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / 'src'
if SRC_ROOT.exists():
    src_str = str(SRC_ROOT)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
