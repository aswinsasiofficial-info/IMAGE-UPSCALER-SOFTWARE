"""Secure API-key storage for the Metadata Generator.

Lookup order per provider: in-session value → environment variable
(``GEMINI_API_KEY`` / ``GROQ_API_KEY`` / ``OLLAMA_API_KEY``) → Windows
Credential Manager via ``keyring``. Keys are never written to disk by the app
itself, never logged, and never included in exported files.

Local Ollama is deliberately absent from the table: it talks to a server on the
user's own machine and must never require or store a credential.
"""

from __future__ import annotations

import os

from .metadata_settings import API_KEY_ENV_VARS, provider_requires_key

_KEYRING_SERVICE = "Upscaler-Metadata"
_ENV_VARS = API_KEY_ENV_VARS

# Lives only as long as the process — filled from the UI's key field.
_session_keys: dict[str, str] = {}


def _keyring_available() -> bool:
    try:
        import keyring  # noqa: F401
        return True
    except Exception:
        return False


def set_session_key(provider: str, key: str) -> None:
    """Remember the typed key for the rest of this app session."""
    key = (key or "").strip()
    if key:
        _session_keys[provider] = key
    else:
        _session_keys.pop(provider, None)


def get_key(provider: str) -> str:
    """Best available key for *provider* ("" when none is configured)."""
    if not provider_requires_key(provider):
        return ""          # local Ollama never uses or stores credentials
    if provider in _session_keys:
        return _session_keys[provider]
    env = os.environ.get(_ENV_VARS.get(provider, ""), "").strip()
    if env:
        return env
    if _keyring_available():
        try:
            import keyring
            return (keyring.get_password(_KEYRING_SERVICE, provider) or "").strip()
        except Exception:
            return ""
    return ""


def save_key(provider: str, key: str) -> tuple[bool, str]:
    """Persist *key* in Windows Credential Manager. Returns (ok, message)."""
    key = (key or "").strip()
    if not provider_requires_key(provider):
        return False, f"{provider} runs locally and needs no key to store."
    if not key:
        return False, "Nothing to save — the key field is empty."
    if not _keyring_available():
        return False, "keyring is not installed — run: pip install keyring"
    try:
        import keyring
        keyring.set_password(_KEYRING_SERVICE, provider, key)
        return True, f"Key saved to Windows Credential Manager ({provider})."
    except Exception as exc:
        return False, f"Could not save key: {exc}"


def delete_key(provider: str) -> tuple[bool, str]:
    if not _keyring_available():
        return False, "keyring is not installed."
    try:
        import keyring
        keyring.delete_password(_KEYRING_SERVICE, provider)
        _session_keys.pop(provider, None)
        return True, f"Stored {provider} key deleted."
    except Exception as exc:
        return False, f"Could not delete key: {exc}"


def has_stored_key(provider: str) -> bool:
    if not _keyring_available():
        return False
    try:
        import keyring
        return bool(keyring.get_password(_KEYRING_SERVICE, provider))
    except Exception:
        return False
