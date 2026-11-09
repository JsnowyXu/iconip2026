from django.urls import path

from . import views

urlpatterns = [
    path('health/', views.health_view),
    path('home/', views.home_view),
    path('datasets/', views.dataset_list_view),
    path('datasets/<slug:dataset_key>/', views.dataset_detail_view),
    path('datasets/<slug:dataset_key>/graph/', views.dataset_graph_view),
    path('pipelines/options/', views.pipeline_options_view),
    path('pipelines/preprocess/', views.create_preprocess_job_view),
    path('training/options/', views.training_options_view),
    path('training/jobs/', views.create_training_job_view),
    path('jobs/', views.job_list_view),
    path('jobs/<int:job_id>/', views.job_detail_view),
    path('jobs/<int:job_id>/log/', views.job_log_view),
    path('models/', views.model_list_view),
    path('models/discover/', views.discover_models_view),
    path('models/<int:model_id>/', views.model_detail_view),
    path('models/<int:model_id>/infer/', views.model_infer_view),
]
