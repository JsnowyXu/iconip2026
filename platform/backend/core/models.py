from django.db import models


class PlatformJob(models.Model):
    KIND_CHOICES = [
        ('preprocess', 'Preprocess'),
        ('training', 'Training'),
        ('inference', 'Inference'),
    ]
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('succeeded', 'Succeeded'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]

    name = models.CharField(max_length=255)
    kind = models.CharField(max_length=32, choices=KIND_CHOICES)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default='queued')
    payload = models.JSONField(default=dict, blank=True)
    command = models.JSONField(default=list, blank=True)
    command_preview = models.TextField(blank=True)
    working_directory = models.CharField(max_length=1024, blank=True)
    script_path = models.CharField(max_length=1024, blank=True)
    log_path = models.CharField(max_length=1024, blank=True)
    output_path = models.CharField(max_length=1024, blank=True)
    pid = models.IntegerField(null=True, blank=True)
    exit_code = models.IntegerField(null=True, blank=True)
    retry_count = models.PositiveIntegerField(default=0)
    result = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']


class ModelArtifact(models.Model):
    name = models.CharField(max_length=255)
    dataset_key = models.CharField(max_length=128, blank=True)
    task_key = models.CharField(max_length=128, blank=True)
    artifact_type = models.CharField(max_length=64, default='dhc')
    checkpoint_path = models.CharField(max_length=1024, unique=True)
    metrics_path = models.CharField(max_length=1024, blank=True)
    dataset_path = models.CharField(max_length=1024, blank=True)
    config = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    supports_inference = models.BooleanField(default=False)
    source_job = models.ForeignKey(PlatformJob, null=True, blank=True, on_delete=models.SET_NULL, related_name='artifacts')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
