from django.urls import path

from . import views

app_name = "upscaler"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/state/", views.state, name="state"),
    path("api/queue/add/", views.queue_add, name="queue-add"),
    path("api/queue/remove/", views.queue_remove, name="queue-remove"),
    path("api/queue/clear/", views.queue_clear, name="queue-clear"),
    path("api/plan/", views.plan, name="plan"),
    path("api/dialog/images/", views.dialog_images, name="dialog-images"),
    path("api/dialog/folder/", views.dialog_folder, name="dialog-folder"),
    path("api/dialog/output/", views.dialog_output, name="dialog-output"),
    path("api/dialog/message/", views.dialog_message, name="dialog-message"),
    path("api/jobs/", views.job_start, name="job-start"),
    path("api/jobs/current/", views.job_status, name="job-status"),
    path("api/jobs/current/cancel/", views.job_cancel, name="job-cancel"),
]
