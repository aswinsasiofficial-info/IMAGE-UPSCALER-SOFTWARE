from django.urls import include, path

urlpatterns = [
    path("", include("app.upscaler.urls")),
]
