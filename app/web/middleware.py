"""Per-launch access token for the embedded server.

The Upscaler API accepts file-system paths and writes files, so only the
app's own window (or the browser it opens) may use it. The desktop shell
opens ``/?launch=<token>``; that request trades the token for a SameSite
cookie and every later request must carry it. Web pages from other sites
can't send the cookie, and other hosts are already refused by
``ALLOWED_HOSTS``. Without a token (``manage.py runserver``) the check is off.
"""

from __future__ import annotations

import secrets
from typing import Optional

from django.http import HttpResponseForbidden, HttpResponseRedirect

COOKIE_NAME = "upscaler_launch"
_launch_token: Optional[str] = None


def new_launch_token() -> str:
    global _launch_token
    _launch_token = secrets.token_urlsafe(32)
    return _launch_token


def clear_launch_token() -> None:
    global _launch_token
    _launch_token = None


class LaunchTokenMiddleware:
    def __init__(self, get_response) -> None:
        self.get_response = get_response

    def __call__(self, request):
        token = _launch_token
        if token is None:
            return self.get_response(request)
        offered = request.GET.get("launch")
        if offered is not None:
            # Compare bytes: compare_digest raises on non-ASCII str input.
            if not secrets.compare_digest(offered.encode(), token.encode()):
                return HttpResponseForbidden("This launch link has expired — reopen the Upscaler from the app.")
            response = HttpResponseRedirect(request.path)
            response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="Strict")
            return response
        if secrets.compare_digest(request.COOKIES.get(COOKIE_NAME, "").encode(), token.encode()):
            return self.get_response(request)
        return HttpResponseForbidden("The Upscaler only opens from the desktop app.")
