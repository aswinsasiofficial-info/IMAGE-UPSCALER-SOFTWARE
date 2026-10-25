from functools import partial

from django.contrib.staticfiles.views import serve
from django.urls import include, path, re_path

urlpatterns = [
    path("", include("app.upscaler.urls")),
    # The page's CSS/JS. The desktop's embedded server serves these before
    # routing; this makes `manage.py runserver` (DEBUG off) serve them too —
    # fine for a local-only app.
    re_path(r"^static/(?P<path>.*)$", partial(serve, insecure=True)),
]
