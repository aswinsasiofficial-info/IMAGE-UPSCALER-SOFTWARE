"""Django settings for the embedded, local-only Upscaler server.

The server only ever listens on 127.0.0.1 inside the desktop process (see
app/web/server.py). There is no database: the queue and batch state live in
memory for the lifetime of the app, like the rest of the desktop UI.
"""

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# A fresh key per launch: nothing signed needs to survive a restart.
SECRET_KEY = os.environ.get("UPSCALER_SECRET_KEY") or secrets.token_urlsafe(50)
DEBUG = os.environ.get("UPSCALER_DEBUG") == "1"
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]   # also blocks DNS-rebinding attempts

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "app.upscaler",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "app.web.middleware.LaunchTokenMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "app.web.urls"
WSGI_APPLICATION = None

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    },
]

DATABASES = {}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

STATIC_URL = "/static/"

CSRF_COOKIE_SAMESITE = "Strict"
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {
        # Errors go to the console the app was launched from; the page polls
        # every few hundred ms, so routine request lines are kept quiet.
        "django": {"handlers": ["console"], "level": "ERROR"},
        "django.server": {"handlers": ["console"], "level": "ERROR", "propagate": False},
    },
}
