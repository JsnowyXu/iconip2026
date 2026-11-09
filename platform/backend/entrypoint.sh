#!/usr/bin/env bash
set -euo pipefail
cd /app/platform/backend

python - <<'PY'
import os
import socket
import time
host = os.environ.get('MYSQL_HOST', '')
port = int(os.environ.get('MYSQL_PORT', '3306'))
if host:
    for _ in range(60):
        try:
            with socket.create_connection((host, port), timeout=2):
                break
        except OSError:
            time.sleep(2)
PY

python manage.py migrate --noinput
exec "$@"
