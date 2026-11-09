from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True
    dependencies = []

    operations = [
        migrations.CreateModel(
            name='PlatformJob',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('kind', models.CharField(choices=[('preprocess', 'Preprocess'), ('training', 'Training'), ('inference', 'Inference')], max_length=32)),
                ('status', models.CharField(choices=[('queued', 'Queued'), ('running', 'Running'), ('succeeded', 'Succeeded'), ('failed', 'Failed'), ('cancelled', 'Cancelled')], default='queued', max_length=32)),
                ('payload', models.JSONField(blank=True, default=dict)),
                ('command', models.JSONField(blank=True, default=list)),
                ('command_preview', models.TextField(blank=True)),
                ('working_directory', models.CharField(blank=True, max_length=1024)),
                ('script_path', models.CharField(blank=True, max_length=1024)),
                ('log_path', models.CharField(blank=True, max_length=1024)),
                ('output_path', models.CharField(blank=True, max_length=1024)),
                ('pid', models.IntegerField(blank=True, null=True)),
                ('exit_code', models.IntegerField(blank=True, null=True)),
                ('retry_count', models.PositiveIntegerField(default=0)),
                ('result', models.JSONField(blank=True, default=dict)),
                ('error_message', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('finished_at', models.DateTimeField(blank=True, null=True)),
            ],
            options={'ordering': ['-created_at']},
        ),
        migrations.CreateModel(
            name='ModelArtifact',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('dataset_key', models.CharField(blank=True, max_length=128)),
                ('task_key', models.CharField(blank=True, max_length=128)),
                ('artifact_type', models.CharField(default='dhc', max_length=64)),
                ('checkpoint_path', models.CharField(max_length=1024, unique=True)),
                ('metrics_path', models.CharField(blank=True, max_length=1024)),
                ('dataset_path', models.CharField(blank=True, max_length=1024)),
                ('config', models.JSONField(blank=True, default=dict)),
                ('metrics', models.JSONField(blank=True, default=dict)),
                ('supports_inference', models.BooleanField(default=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('source_job', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='artifacts', to='core.platformjob')),
            ],
            options={'ordering': ['-created_at']},
        ),
    ]
