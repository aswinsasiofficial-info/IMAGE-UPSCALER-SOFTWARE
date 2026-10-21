"""Gemini and Groq callers that turn one image into Adobe-ready metadata.

Both providers are optional runtime dependencies: import happens inside the
call so the app still starts (and the upscaler still works) without them.
Errors from the SDKs/HTTP layer are wrapped in ``ProviderError`` — the
original exception classes are never reconstructed.
"""

from __future__ import annotations

import base64
import time

from .metadata_settings import (
    MetadataSettings, ImageMetadata, ProviderError,
    build_prompt, parse_model_json, prepare_analysis_bytes,
    DEFAULT_MODELS, GEMINI_MODELS, GROQ_MODELS,
)

# HTTP codes worth retrying with backoff. Everything else (401/403/404/400…)
# is a permanent problem — bad key, retired model, rejected image.
_TRANSIENT_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_HINTS = ("rate limit", "timed out", "timeout", "temporarily",
                    "unavailable", "overloaded", "connection", "reset by peer")

MAX_ATTEMPTS = 3
BACKOFF_BASE = 2.0  # seconds; doubles per attempt


def _is_transient(exc: Exception) -> bool:
    # An explicit flag set by generate_metadata wins: the *final* wrapped
    # ProviderError's message ("… failed after N attempts: …") no longer
    # matches the hint list, and network-flavoured SDK exceptions often
    # carry no HTTP code — without the flag the worker would misclassify
    # an exhausted transient failure as permanent and skip retry passes.
    if getattr(exc, "retryable", None) is True:
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code in _TRANSIENT_CODES
    msg = str(exc).lower()
    return any(h in msg for h in _TRANSIENT_HINTS) and "no longer available" not in msg


def is_retryable_error(exc: Exception) -> bool:
    """True when a later *batch-level* retry pass could plausibly succeed.

    Used by ``MetadataWorker`` to decide whether a failed image is worth an
    automatic retry (rate limits, timeouts, network blips, provider 5xx) or
    is a permanent problem that must be surfaced to the user immediately
    (bad API key, unreadable/unsupported image, retired model).
    """
    return _is_transient(exc)


def _retirement_hint(exc: Exception) -> str:
    """Google names the replacement model inside 404 retirement messages."""
    msg = str(exc)
    if "no longer available" in msg and "use models/" in msg:
        tail = msg.split("use models/")[-1].strip()
        replacement = tail.strip("\"'.,; ").split()[0]
        return f" Try switching the model to '{replacement}'."
    return ""


def missing_dependency(package: str) -> ProviderError:
    return ProviderError(
        f"The '{package}' package is not installed. Run: "
        f".venv\\Scripts\\python -m pip install {package}")


def _classify_http(provider: str, code: int | None, detail: str,
                   model: str = "") -> str:
    """Turn a raw HTTP failure into a distinguishable, actionable message."""
    if code in (401, 403):
        return (f"{provider} rejected your API key (HTTP {code}) — check the "
                f"{provider} key on the Metadata screen.")
    if code == 404:
        return (f"{provider} has no model '{model}' (HTTP 404) — it is retired "
                "or this key has no access; pick a current model from the list.")
    if code == 429:
        return (f"{provider} rate limit reached (HTTP 429) — too many "
                "requests; waiting before trying again.")
    if code == 400:
        return (f"{provider} rejected the request (HTTP 400) — the image or "
                f"prompt was not accepted: {detail}")
    return f"{provider} API error ({code}): {detail}"


def _resolve_model(settings: MetadataSettings) -> str:
    """Model for the *currently selected* provider.

    One settings object carries both providers' choices, so a model ID that
    belongs to the other provider (or an empty one) falls back to this
    provider's default instead of guaranteeing a 404.
    """
    if settings.provider == "Gemini":
        other = GROQ_MODELS
    else:
        other = GEMINI_MODELS
    model = (settings.model or "").strip()
    if not model or model in other:
        return DEFAULT_MODELS[settings.provider]
    return model


def generate_metadata(image_path, settings: MetadataSettings, api_key: str,
                      log=None) -> ImageMetadata:
    """Generate metadata for one image with bounded retries.

    ``log`` is an optional callable for human-readable progress lines.
    Raises ``ProviderError`` when every attempt fails.
    """
    if not api_key:
        raise ProviderError(
            f"No API key provided for {settings.provider}. Enter one, or set "
            f"the {'GEMINI_API_KEY' if settings.provider == 'Gemini' else 'GROQ_API_KEY'} "
            "environment variable.")
    # Raises ProviderError with a clear message for unreadable/unsupported images.
    data, mime = prepare_analysis_bytes(image_path)
    prompt = build_prompt(settings)

    last_error: Exception | None = None
    last_transient = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if settings.provider == "Gemini":
                text = _call_gemini(data, mime, prompt, settings, api_key)
            elif settings.provider == "Groq":
                text = _call_groq(data, mime, prompt, settings, api_key)
            else:
                raise ProviderError(f"Unknown provider: {settings.provider}")
            return parse_model_json(text, settings)
        except (ValueError, ProviderError) as exc:
            last_error = exc
            last_transient = _is_transient(exc)
            if not last_transient:
                hint = _retirement_hint(exc)
                raise ProviderError(f"{exc}{hint}",
                                    code=getattr(exc, "code", None)) from exc
        except Exception as exc:  # unexpected SDK/transport failure
            last_error = exc
            raise ProviderError(f"{settings.provider} request failed: {exc}",
                                code=getattr(exc, "code", None)) from exc
        if attempt < MAX_ATTEMPTS:
            delay = BACKOFF_BASE * (2 ** (attempt - 1))
            if log:
                log(f"  {settings.provider} hiccup ({last_error}); retrying in "
                    f"{delay:.0f}s — attempt {attempt + 1}/{MAX_ATTEMPTS}")
            time.sleep(delay)
    # Per-request attempts exhausted on a TRANSIENT error: carry that fact on
    # the final ProviderError so MetadataWorker still runs batch retry passes.
    final = ProviderError(f"{settings.provider} failed after {MAX_ATTEMPTS} attempts: "
                          f"{last_error}", code=getattr(last_error, "code", None))
    final.retryable = True if last_transient else None
    raise final


# -- provider calls ------------------------------------------------------------
def _call_gemini(data: bytes, mime: str, prompt: str, settings: MetadataSettings,
                 api_key: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise missing_dependency("google-genai") from exc

    client = genai.Client(api_key=api_key)
    model = _resolve_model(settings)
    try:
        response = client.models.generate_content(
            model=model,
            # Inline image bytes must go through Part.from_bytes with the real
            # MIME type. Passing a types.Image object in ``contents`` fails in
            # google-genai >= 2.x with "file uri and mime_type are required".
            contents=[prompt, types.Part.from_bytes(data=data, mime_type=mime)],
            config=types.GenerateContentConfig(temperature=0.4),
        )
    except Exception as exc:
        code = getattr(exc, "code", None)
        raise ProviderError(
            _classify_http("Gemini", code, f"{type(exc).__name__}: {exc}", model),
            code=code) from exc
    text = getattr(response, "text", None)
    if not text:
        raise ProviderError("Gemini returned an empty response.")
    return text


def _call_groq(data: bytes, mime: str, prompt: str, settings: MetadataSettings,
               api_key: str) -> str:
    """Groq exposes an OpenAI-compatible chat API — plain HTTPS, no SDK needed."""
    try:
        import requests
    except ImportError as exc:
        raise missing_dependency("requests") from exc

    model = _resolve_model(settings)
    image_b64 = base64.b64encode(data).decode("ascii")
    body = {
        "model": model,
        "temperature": 0.4,
        "max_tokens": 2048,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ],
        }],
    }
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=body, timeout=120,
        )
    except requests.Timeout as exc:
        raise ProviderError(
            f"Groq request timed out after 120s — check your connection "
            f"and retry: {exc}") from exc
    except requests.ConnectionError as exc:
        raise ProviderError(
            f"Could not reach the Groq API — check your internet "
            f"connection: {exc}") from exc
    except requests.RequestException as exc:
        raise ProviderError(f"Groq request failed: {exc}") from exc
    if resp.status_code != 200:
        detail = ""
        try:
            detail = str(resp.json().get("error", {}).get("message", ""))[:300]
        except Exception:
            detail = resp.text[:300]
        raise ProviderError(
            _classify_http("Groq", resp.status_code, detail or resp.reason, model),
            code=resp.status_code)
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise ProviderError(f"Groq returned an unexpected response shape: {exc}") from exc
