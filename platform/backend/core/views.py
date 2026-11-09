import json
from pathlib import Path
from typing import Any, Dict

from django.conf import settings
from django.http import HttpRequest, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from .models import ModelArtifact, PlatformJob
from .services.graph import sample_graph_payload
from .services.inference import discover_model_payload, run_inference
from .services.job_builder import configure_preprocess_job, configure_training_job
from .services.registry import discover_dataset_catalog, get_dataset_detail, pipeline_options, training_options


def _json_body(request: HttpRequest) -> Dict[str, Any]:
    if not request.body:
        return {}
    return json.loads(request.body.decode('utf-8'))


def _job_dict(job: PlatformJob) -> Dict[str, Any]:
    return {
        'id': job.id,
        'name': job.name,
        'kind': job.kind,
        'status': job.status,
        'payload': job.payload,
        'command_preview': job.command_preview,
        'script_path': job.script_path,
        'log_path': job.log_path,
        'output_path': job.output_path,
        'pid': job.pid,
        'exit_code': job.exit_code,
        'result': job.result,
        'error_message': job.error_message,
        'created_at': job.created_at.isoformat() if job.created_at else None,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'finished_at': job.finished_at.isoformat() if job.finished_at else None,
    }


def _artifact_dict(artifact: ModelArtifact) -> Dict[str, Any]:
    return {
        'id': artifact.id,
        'name': artifact.name,
        'dataset_key': artifact.dataset_key,
        'task_key': artifact.task_key,
        'artifact_type': artifact.artifact_type,
        'checkpoint_path': artifact.checkpoint_path,
        'metrics_path': artifact.metrics_path,
        'dataset_path': artifact.dataset_path,
        'supports_inference': artifact.supports_inference,
        'metrics': artifact.metrics,
        'config': artifact.config,
        'source_job_id': artifact.source_job_id,
        'created_at': artifact.created_at.isoformat() if artifact.created_at else None,
    }


@require_GET
def health_view(request: HttpRequest) -> JsonResponse:
    return JsonResponse({'ok': True, 'service': 'dhc-platform-backend'})


@require_GET
def home_view(request: HttpRequest) -> JsonResponse:
    readme_path = Path(settings.DHC_PLATFORM['REPO_ROOT']) / 'README.md'
    readme = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''
    return JsonResponse(
        {
            'project': 'Dynamic Heterogeneous Community Platform',
            'github_url': settings.DHC_PLATFORM['GITHUB_URL'],
            'dataset_count': len(discover_dataset_catalog()),
            'job_count': PlatformJob.objects.count(),
            'model_count': ModelArtifact.objects.count(),
            'latest_jobs': [_job_dict(x) for x in PlatformJob.objects.all()[:5]],
            'readme_excerpt': readme[:2000],
            'highlights': [
                '数据集卡片化展示 + 图结构采样可视化',
                '预处理管道自动生成脚本并由 Worker 执行',
                '训练结果自动归档，支持日志追踪与模型推理',
            ],
        }
    )


@require_GET
def dataset_list_view(request: HttpRequest) -> JsonResponse:
    return JsonResponse({'items': discover_dataset_catalog()})


@require_GET
def dataset_detail_view(request: HttpRequest, dataset_key: str) -> JsonResponse:
    item = get_dataset_detail(dataset_key)
    if item is None:
        return JsonResponse({'error': 'dataset not found'}, status=404)
    return JsonResponse(item)


@require_GET
def dataset_graph_view(request: HttpRequest, dataset_key: str) -> JsonResponse:
    item = get_dataset_detail(dataset_key)
    if item is None:
        return JsonResponse({'error': 'dataset not found'}, status=404)
    pt_path = request.GET.get('path') or item['unified_path']
    payload = sample_graph_payload(pt_path, max_edges=int(request.GET.get('max_edges', 180)), max_nodes=int(request.GET.get('max_nodes', 80)))
    return JsonResponse(payload)


@require_GET
def pipeline_options_view(request: HttpRequest) -> JsonResponse:
    return JsonResponse(pipeline_options())


@csrf_exempt
@require_http_methods(['POST'])
def create_preprocess_job_view(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)
    job = PlatformJob.objects.create(
        name=payload.get('name') or f"Preprocess {payload.get('dataset_key', 'dataset')}",
        kind='preprocess',
        status='queued',
        payload=payload,
    )
    try:
        configure_preprocess_job(job)
        job.save()
    except Exception as exc:
        job.status = 'failed'
        job.error_message = str(exc)
        job.save(update_fields=['status', 'error_message', 'updated_at'])
        return JsonResponse({'error': str(exc), 'job': _job_dict(job)}, status=400)
    return JsonResponse({'job': _job_dict(job)}, status=201)


@require_GET
def training_options_view(request: HttpRequest) -> JsonResponse:
    return JsonResponse(training_options())


@csrf_exempt
@require_http_methods(['POST'])
def create_training_job_view(request: HttpRequest) -> JsonResponse:
    payload = _json_body(request)
    job = PlatformJob.objects.create(
        name=payload.get('name') or f"Train {Path(payload.get('data_path', 'task')).stem}",
        kind='training',
        status='queued',
        payload=payload,
    )
    try:
        configure_training_job(job)
        job.save()
    except Exception as exc:
        job.status = 'failed'
        job.error_message = str(exc)
        job.save(update_fields=['status', 'error_message', 'updated_at'])
        return JsonResponse({'error': str(exc), 'job': _job_dict(job)}, status=400)
    return JsonResponse({'job': _job_dict(job)}, status=201)


@require_GET
def job_list_view(request: HttpRequest) -> JsonResponse:
    qs = PlatformJob.objects.all()
    kind = request.GET.get('kind')
    status = request.GET.get('status')
    if kind:
        qs = qs.filter(kind=kind)
    if status:
        qs = qs.filter(status=status)
    return JsonResponse({'items': [_job_dict(x) for x in qs[:100]]})


@require_GET
def job_detail_view(request: HttpRequest, job_id: int) -> JsonResponse:
    return JsonResponse(_job_dict(get_object_or_404(PlatformJob, id=job_id)))


@require_GET
def job_log_view(request: HttpRequest, job_id: int) -> JsonResponse:
    job = get_object_or_404(PlatformJob, id=job_id)
    path = Path(job.log_path)
    content = path.read_text(encoding='utf-8', errors='ignore')[-12000:] if path.exists() else ''
    return JsonResponse({'job_id': job.id, 'log': content, 'exists': path.exists()})


def _sync_models(limit: int = 200) -> int:
    repo_root = Path(settings.DHC_PLATFORM['REPO_ROOT'])
    count = 0
    for ckpt in sorted(repo_root.glob('results/**/best_model.pt'))[:limit]:
        payload = discover_model_payload(ckpt)
        ModelArtifact.objects.update_or_create(checkpoint_path=payload['checkpoint_path'], defaults=payload)
        count += 1
    return count


@csrf_exempt
@require_http_methods(['POST'])
def discover_models_view(request: HttpRequest) -> JsonResponse:
    count = _sync_models()
    return JsonResponse({'discovered': count, 'items': [_artifact_dict(x) for x in ModelArtifact.objects.all()[:100]]})


@require_GET
def model_list_view(request: HttpRequest) -> JsonResponse:
    if request.GET.get('sync') == '1':
        _sync_models()
    return JsonResponse({'items': [_artifact_dict(x) for x in ModelArtifact.objects.all()[:200]]})


@require_GET
def model_detail_view(request: HttpRequest, model_id: int) -> JsonResponse:
    return JsonResponse(_artifact_dict(get_object_or_404(ModelArtifact, id=model_id)))


@csrf_exempt
@require_http_methods(['POST'])
def model_infer_view(request: HttpRequest, model_id: int) -> JsonResponse:
    artifact = get_object_or_404(ModelArtifact, id=model_id)
    if not artifact.supports_inference:
        return JsonResponse({'error': 'this model does not have enough metadata for inference'}, status=400)
    try:
        payload = _json_body(request)
        result = run_inference(artifact, payload)
        return JsonResponse({'artifact': _artifact_dict(artifact), 'result': result})
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)
