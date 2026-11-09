import json
import subprocess
import time
from pathlib import Path
from typing import Optional

from django.db import transaction
from django.utils import timezone

from core.models import ModelArtifact, PlatformJob

from .graph import sample_graph_payload
from .inference import discover_model_payload
from .project import build_python_env


def recover_stale_jobs() -> None:
    PlatformJob.objects.filter(status='running').update(status='queued', pid=None, started_at=None)


def _collect_preprocess_result(job: PlatformJob) -> dict:
    path = Path(job.output_path)
    result = {'output_path': job.output_path}
    if path.is_file() and path.suffix == '.pt':
        result['graph_preview'] = sample_graph_payload(str(path), max_edges=80, max_nodes=40)
    elif path.is_dir():
        files = sorted([str(p) for p in path.glob('*.pt')][:20])
        result['generated_files'] = files
        if files:
            result['graph_preview'] = sample_graph_payload(files[0], max_edges=80, max_nodes=40)
    return result


def _register_artifact(job: PlatformJob) -> dict:
    ckpt = Path(job.output_path) / 'best_model.pt'
    if not ckpt.exists():
        return {}
    payload = discover_model_payload(ckpt)
    artifact, _ = ModelArtifact.objects.update_or_create(
        checkpoint_path=payload['checkpoint_path'],
        defaults={**payload, 'source_job': job},
    )
    return {'artifact_id': artifact.id, 'artifact_name': artifact.name}


def finalize_job(job: PlatformJob) -> None:
    result = dict(job.result or {})
    if job.kind == 'preprocess':
        result.update(_collect_preprocess_result(job))
    elif job.kind == 'training':
        result.update(_register_artifact(job))
    job.result = result
    job.save(update_fields=['result', 'updated_at'])


def claim_next_job() -> Optional[PlatformJob]:
    with transaction.atomic():
        job = PlatformJob.objects.select_for_update(skip_locked=True).filter(status='queued').order_by('created_at').first()
        if job is None:
            return None
        job.status = 'running'
        job.started_at = timezone.now()
        job.error_message = ''
        job.save(update_fields=['status', 'started_at', 'error_message', 'updated_at'])
        return job


def run_worker_loop(poll_sec: int = 3) -> None:
    recover_stale_jobs()
    current = None
    current_job_id = None
    log_handle = None
    while True:
        if current is None:
            job = claim_next_job()
            if job is None:
                time.sleep(poll_sec)
                continue
            log_path = Path(job.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open('a', encoding='utf-8')
            current = subprocess.Popen(
                job.command,
                cwd=job.working_directory or None,
                env=build_python_env(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            current_job_id = job.id
            job.pid = current.pid
            job.save(update_fields=['pid', 'updated_at'])
            continue

        rc = current.poll()
        if rc is None:
            time.sleep(poll_sec)
            continue

        if log_handle is not None:
            log_handle.flush()
            log_handle.close()
            log_handle = None

        job = PlatformJob.objects.get(id=current_job_id)
        job.exit_code = int(rc)
        job.finished_at = timezone.now()
        job.status = 'succeeded' if rc == 0 else 'failed'
        if rc != 0 and Path(job.log_path).exists():
            try:
                job.error_message = Path(job.log_path).read_text(encoding='utf-8')[-4000:]
            except Exception:
                job.error_message = 'job failed; unable to read log'
        job.save(update_fields=['exit_code', 'finished_at', 'status', 'error_message', 'updated_at'])
        if rc == 0:
            finalize_job(job)
        current = None
        current_job_id = None
