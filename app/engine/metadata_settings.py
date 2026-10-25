"""Metadata generation settings and provider-independent helpers.

Kept deliberately separate from the upscaler: the Metadata Generator works on
image files that already exist on disk and never touches upscale settings.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

# Adobe Stock contributor limits (used to clamp generated text).
ADOBE_TITLE_MAX = 70
ADOBE_KEYWORDS_MAX = 50          # Adobe's hard cap — the UI never offers more
ADOBE_KEYWORD_MAX = 40
ADOBE_DESCRIPTION_MAX = 2000

OLLAMA_LOCAL = "Ollama (Local)"
OLLAMA_CLOUD = "Ollama Cloud"
SUPPORTED_PROVIDERS = ("Gemini", "Groq", OLLAMA_LOCAL, OLLAMA_CLOUD)

# Ollama connection endpoints (verified against docs.ollama.com):
#  * local API  — http://127.0.0.1:11434, no authentication, images stay local
#  * cloud API  — https://ollama.com/api, Bearer API key, images are uploaded
OLLAMA_LOCAL_URL = "http://localhost:11434"
OLLAMA_CLOUD_URL = "https://ollama.com"

# Model catalogues — combos in the UI are editable so future retirements can be
# handled without a code change (a 404 from the API names the replacement).
# Only vision-capable models belong here: both screens send an image, and a
# text-only model rejects the request.
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-4-argon",
    "gemini-2.5-flash",
]
# Groq's llama-4 scout/maverick IDs were removed from the API (they now 404).
# qwen/qwen3.8-27b is Groq's current generally-available vision model; their
# other vision entry (meta-models/muse-spark-open) is limited-preview only, so
# it is deliberately not offered — the editable combo still accepts it for
# users who have preview access.
GROQ_MODELS = [
    "qwen/qwen3.8-27b",
]
# Ollama ships hundreds of models with per-account availability differences, so
# nothing is assumed here: the Metadata screen queries the server itself
# (``/api/tags`` — and ``/api/show`` capabilities on older local servers) and
# lists only models that report the ``vision`` capability. The entries below are
# the names Ollama's own documentation/API report as image-capable, offered as a
# starting point before the availability check has been run.
OLLAMA_LOCAL_MODELS = [
    "gemma4",
    "gemma3",
    "llama3.2-vision",
    "qwen3-vl",
    "llava",
]
# Cloud IDs are size-qualified and account/plan dependent, so the check refreshes
# this list from https://ollama.com/api/tags for the signed-in account. Every
# name below reported the ``vision`` capability on Ollama's cloud model list when
# this feature was built; treat them as a starting point, not a guarantee.
OLLAMA_CLOUD_MODELS = [
    "gemma4:31b",
    "kimi-k2.6",
    "minimax-m3",
    "mistral-large-4",
]

DEFAULT_MODELS = {
    "Gemini": GEMINI_MODELS[0],
    "Groq": GROQ_MODELS[0],
    OLLAMA_LOCAL: OLLAMA_LOCAL_MODELS[0],
    OLLAMA_CLOUD: OLLAMA_CLOUD_MODELS[0],
}

# Every provider's known vision models — used for the UI combos and to detect a
# model ID that clearly belongs to a different provider.
MODEL_CATALOGUES = {
    "Gemini": GEMINI_MODELS,
    "Groq": GROQ_MODELS,
    OLLAMA_LOCAL: OLLAMA_LOCAL_MODELS,
    OLLAMA_CLOUD: OLLAMA_CLOUD_MODELS,
}

KEYWORD_COUNT_DEFAULT = 45       # Gemini/Groq: leave Adobe's last slot free
OLLAMA_KEYWORD_COUNT = ADOBE_KEYWORDS_MAX   # Ollama: a full Adobe set (50)

# Provider families that talk to the Ollama API (local server or hosted cloud).
OLLAMA_PROVIDERS = (OLLAMA_LOCAL, OLLAMA_CLOUD)

# Environment variable named in every "missing key" message and consulted by
# credential_store. Local Ollama is absent on purpose: it needs no credential.
API_KEY_ENV_VARS = {
    "Gemini": "GEMINI_API_KEY",
    "Groq": "GROQ_API_KEY",
    OLLAMA_CLOUD: "OLLAMA_API_KEY",
}


def is_ollama(provider: str) -> bool:
    return provider in OLLAMA_PROVIDERS


def provider_requires_key(provider: str) -> bool:
    """Local Ollama runs on the user's own machine and needs no credentials."""
    return not is_ollama(provider) or provider == OLLAMA_CLOUD


LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Dutch", "Japanese", "Korean", "Chinese", "Hindi", "Russian", "Arabic",
]

# Longest image edge sent to an API — keeps requests small and never alters
# the file on disk.
ANALYSIS_MAX_EDGE = 1024


@dataclass
class MetadataSettings:
    """User-configurable options for the Metadata Generator screen."""
    provider: str = "Gemini"           # see SUPPORTED_PROVIDERS
    model: str = DEFAULT_MODELS["Gemini"]
    language: str = "English"
    keyword_count: int = 45            # target number of keywords to produce
    generate_description: bool = True
    # Ollama only: base URL of the local server (cloud mode always uses
    # OLLAMA_CLOUD_URL) and the Adobe-complete keyword target ("exactly N"
    # instead of "up to N").
    ollama_url: str = OLLAMA_LOCAL_URL
    exact_keyword_count: bool = False


@dataclass
class ImageMetadata:
    """The editable, exportable metadata attached to one image."""
    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"title": self.title, "description": self.description,
                "keywords": list(self.keywords)}

    @staticmethod
    def from_dict(data: dict) -> "ImageMetadata":
        return ImageMetadata(
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            keywords=[str(k) for k in data.get("keywords", [])],
        )


class ProviderError(RuntimeError):
    """Wraps any provider failure.

    Third-party SDK exception classes are NEVER reconstructed (their
    constructors may require positional args); we only carry the message and
    an optional HTTP code for retry classification.
    """

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


def clamp_text(value: str, limit: int) -> str:
    return " ".join(str(value).split())[:limit].strip()


def sanitize_keywords(keywords: list, limit: int) -> list[str]:
    """Trim, de-duplicate (case-insensitive) and cap the keyword list."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in keywords:
        kw = clamp_text(raw, ADOBE_KEYWORD_MAX)
        key = kw.lower()
        if not kw or key in seen:
            continue
        seen.add(key)
        out.append(kw)
        if len(out) >= limit:
            break
    return out


def sanitize_metadata(meta: ImageMetadata, settings: MetadataSettings) -> ImageMetadata:
    """Enforce contributor-format limits on whatever the model returned."""
    return ImageMetadata(
        title=clamp_text(meta.title, ADOBE_TITLE_MAX),
        description="" if not settings.generate_description
        else clamp_text(meta.description, ADOBE_DESCRIPTION_MAX),
        keywords=sanitize_keywords(meta.keywords, min(settings.keyword_count, ADOBE_KEYWORDS_MAX)),
    )


def build_prompt(settings: MetadataSettings) -> str:
    """Instruction sent alongside each image to the model."""
    n = min(settings.keyword_count, ADOBE_KEYWORDS_MAX)
    quantity = "exactly" if settings.exact_keyword_count else "up to"
    desc_line = (
        f"- description: a natural, factual sentence or two (max {ADOBE_DESCRIPTION_MAX} characters)"
        if settings.generate_description
        else "- description: return an empty string"
    )
    return (
        "You are preparing Adobe Stock contributor metadata for the attached image.\n"
        f"Write all values in {settings.language}.\n"
        "Respond with ONLY a JSON object, no markdown fences, using exactly this shape:\n"
        '{"title": "...", "description": "...", "keywords": ["..."]}\n'
        f"- title: max {ADOBE_TITLE_MAX} characters, descriptive, no punctuation at the end\n"
        f"{desc_line}\n"
        f"- keywords: {quantity} {n} single words or short phrases, most important first, "
        "no duplicates, no trademarks or artist names\n"
    )


def parse_model_json(text: str, settings: MetadataSettings) -> ImageMetadata:
    """Extract the JSON object a model returned, tolerating fences/prose.

    Raises ``ValueError`` when no usable JSON object is present.
    """
    import json
    import re

    cleaned = re.sub(r"```(?:json)?|```", "", text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(
            "the model's answer contained no complete JSON object "
            f"(truncated or extra-text reply): {cleaned[:160]!r}")
    payload = json.loads(cleaned[start:end + 1])
    meta = ImageMetadata(
        title=str(payload.get("title", "")),
        description=str(payload.get("description", "")),
        keywords=list(payload.get("keywords", []) or []),
    )
    return sanitize_metadata(meta, settings)


# Image formats every supported provider API accepts inline. Anything else that
# PIL can read (BMP, TIFF, GIF, …) is converted to JPEG — the most universally
# supported format. HEIC/HEIF need the optional pillow-heif plugin; without it
# PIL cannot decode them and prepare_analysis_bytes reports that clearly.
IMAGE_MIME_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def prepare_analysis_bytes(path) -> tuple[bytes, str]:
    """Return (image_bytes, mime_type) for an API request, in memory only.

    The source format is kept when a provider accepts it (PNG/JPEG/WEBP);
    other decodable formats are converted to JPEG. The MIME type is derived
    from the encoded output, never guessed from the file extension. Large
    images are down-scaled to <= ANALYSIS_MAX_EDGE on the long side; the file
    on disk is never modified. Raises ``ProviderError`` with an actionable
    message when the file is unreadable or in an unsupported format.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            source_format = (im.format or "").upper()
            im.load()
            target = source_format if source_format in IMAGE_MIME_TYPES else "JPEG"
            if target == "JPEG" and im.mode not in ("RGB", "L"):
                background = Image.new("RGB", im.size, (255, 255, 255))
                rgba = im.convert("RGBA")
                background.paste(rgba, mask=rgba.split()[-1])
                im = background
            if max(im.size) > ANALYSIS_MAX_EDGE:
                ratio = ANALYSIS_MAX_EDGE / max(im.size)
                new_size = (max(1, round(im.width * ratio)), max(1, round(im.height * ratio)))
                im = im.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, target, quality=88)
        return buf.getvalue(), IMAGE_MIME_TYPES[target]
    except UnidentifiedImageError as exc:
        raise ProviderError(
            f"'{path}' is not a readable image. Supported: PNG, JPEG, WEBP "
            "(BMP/TIFF/GIF are converted). HEIC/HEIF need extra support that "
            "isn't installed — convert the file to JPEG first.") from exc
    except (OSError, ValueError) as exc:
        raise ProviderError(f"Could not process image '{path}': {exc}") from exc
