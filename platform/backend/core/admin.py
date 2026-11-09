from django.contrib import admin

from .models import ModelArtifact, PlatformJob


@admin.register(PlatformJob)
class PlatformJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'kind', 'status', 'created_at', 'started_at', 'finished_at')
    list_filter = ('kind', 'status')
    search_fields = ('name', 'command_preview', 'output_path')


@admin.register(ModelArtifact)
class ModelArtifactAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'dataset_key', 'task_key', 'supports_inference', 'created_at')
    list_filter = ('dataset_key', 'task_key', 'supports_inference')
    search_fields = ('name', 'checkpoint_path', 'dataset_path')
