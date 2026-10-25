"""Focused tests for the Ollama providers (local server + hosted cloud).

Run with:  .venv\\Scripts\\python -m tests.ollama_test

No Ollama install, no API key and no network are needed: every HTTP call is
stubbed, so these tests pin down the real request shape (official /api/chat
with base64 ``images``), the local/cloud split (no credential vs Bearer key),
vision-capability discovery, retry classification, batch result retention and
the all-or-nothing CSV export rule for folder batches.
"""

from __future__ import annotations

import base64
import csv
import io
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.engine.metadata_settings import (
    MetadataSettings, ImageMetadata, ProviderError, ADOBE_KEYWORDS_MAX,
    DEFAULT_MODELS, OLLAMA_LOCAL, OLLAMA_CLOUD, OLLAMA_LOCAL_URL,
    OLLAMA_CLOUD_URL, OLLAMA_KEYWORD_COUNT, API_KEY_ENV_VARS,
    OLLAMA_LOCAL_MODELS, OLLAMA_CLOUD_MODELS,
    build_prompt, parse_model_json, provider_requires_key,
)
from app.engine import metadata_providers
from app.engine import metadata_export
from app.engine import credential_store

REPLY = '{"title": "T", "description": "D", "keywords": ["a"]}'


def _ollama_settings(provider: str = OLLAMA_LOCAL, **kw) -> MetadataSettings:
    base = dict(provider=provider, model=DEFAULT_MODELS[provider],
                keyword_count=OLLAMA_KEYWORD_COUNT, language="English",
                exact_keyword_count=True)
    base.update(kw)
    return MetadataSettings(**base)


class FakeResp:
    def __init__(self, status: int = 200, payload: dict | None = None,
                 text: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.text = text
        self.reason = "Error"

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _ok(payload: dict | None = None) -> FakeResp:
    return FakeResp(200, payload if payload is not None
                    else {"message": {"content": REPLY}})


class OllamaHttp:
    """Canned Ollama endpoints used as a context manager.

    ``chat`` is the response/exception queue for POST /api/chat, ``tags`` the
    reply for GET /api/tags and ``show`` maps a model name to its /api/show
    response. Every call is recorded in ``calls``.
    """

    def __init__(self, chat: list | None = None, tags=None,
                 show: dict[str, FakeResp] | None = None) -> None:
        self.chat = list(chat or [])
        self.tags = tags
        self.show = show or {}
        self.calls: list[dict] = []

    def _next(self, bucket: list):
        if not bucket:
            raise AssertionError("unexpected extra Ollama call")
        return bucket.pop(0) if len(bucket) > 1 else bucket[0]

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "body": json,
                           "timeout": timeout})
        if url.endswith("/api/show"):
            model = str((json or {}).get("model", ""))
            return self.show.get(model, FakeResp(200, {"capabilities": []}))
        action = self._next(self.chat)
        if isinstance(action, Exception):
            raise action
        return action

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "body": None,
                           "timeout": timeout})
        if isinstance(self.tags, Exception):
            raise self.tags
        if self.tags is None:
            raise AssertionError("GET called with no /api/tags stub")
        return self.tags

    def chat_calls(self) -> list[dict]:
        return [c for c in self.calls if c["url"].endswith("/api/chat")]

    def __enter__(self) -> "OllamaHttp":
        import requests

        self._orig = (requests.post, requests.get, metadata_providers.time.sleep)
        requests.post, requests.get = self.post, self.get
        metadata_providers.time.sleep = lambda s: None
        return self

    def __exit__(self, *exc) -> bool:
        import requests

        requests.post, requests.get, metadata_providers.time.sleep = self._orig
        return False


def _image_in(tmp: str) -> Path:
    p = Path(tmp) / "x.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(p)
    return p


def test_ollama_local_request_shape() -> None:
    """Local mode: official /api/chat on the user's own server, no credential,
    the image travelling as base64 in the message's ``images`` array."""
    with tempfile.TemporaryDirectory() as tmp, OllamaHttp(chat=[_ok()]) as http:
        meta = metadata_providers.generate_metadata(
            _image_in(tmp), _ollama_settings(), "", log=lambda m: None)

    assert meta.title == "T" and meta.description == "D"
    req = http.chat_calls()[0]
    assert req["url"] == f"{OLLAMA_LOCAL_URL}/api/chat", req["url"]
    assert "Authorization" not in req["headers"], req["headers"]
    body = req["body"]
    assert body["model"] == DEFAULT_MODELS[OLLAMA_LOCAL]
    assert body["stream"] is False and body["format"] == "json"
    # Hidden reasoning must never eat the answer budget (real qwen3.5 server).
    assert body["think"] is False, body
    assert body["options"]["temperature"] == 0.4
    msg = body["messages"][0]
    assert "Adobe Stock" in msg["content"]
    raw = base64.b64decode(msg["images"][0])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", raw[:8]
    with Image.open(io.BytesIO(raw)) as im:
        assert im.format == "PNG"
    print("  ollama local request shape OK")


def test_ollama_cloud_request_shape_and_keys() -> None:
    """Cloud mode pins Ollama's official host with a Bearer key; local mode must
    never demand or store a credential."""
    with tempfile.TemporaryDirectory() as tmp, OllamaHttp(chat=[_ok()]) as http:
        img = _image_in(tmp)
        meta = metadata_providers.generate_metadata(
            img, _ollama_settings(OLLAMA_CLOUD), "ol-key")
        assert meta.title == "T"
        req = http.chat_calls()[0]
        assert req["url"] == f"{OLLAMA_CLOUD_URL}/api/chat", req["url"]
        assert req["headers"]["Authorization"] == "Bearer ol-key"

        # A leftover local address is ignored in cloud mode.
        metadata_providers.generate_metadata(
            img, _ollama_settings(OLLAMA_CLOUD,
                                  ollama_url="http://10.0.0.9:11434"), "ol-key")
        assert http.chat_calls()[-1]["url"] == f"{OLLAMA_CLOUD_URL}/api/chat"

        # Cloud without a key names only its own environment variable.
        try:
            metadata_providers.generate_metadata(img, _ollama_settings(OLLAMA_CLOUD), "")
            raise AssertionError("expected ProviderError")
        except ProviderError as exc:
            assert API_KEY_ENV_VARS[OLLAMA_CLOUD] in str(exc), str(exc)
            assert "GEMINI" not in str(exc) and "GROQ" not in str(exc)

        # Local mode runs keyless, and the key field is not consulted at all.
        credential_store._session_keys.clear()
        os.environ.pop("OLLAMA_API_KEY", None)
        assert metadata_providers.generate_metadata(img, _ollama_settings(), "").title == "T"
        assert credential_store.get_key(OLLAMA_LOCAL) == ""
        assert not provider_requires_key(OLLAMA_LOCAL)
        assert provider_requires_key(OLLAMA_CLOUD)
        ok, msg = credential_store.save_key(OLLAMA_LOCAL, "sneaky")
        assert not ok and "needs no key" in msg, msg
        assert credential_store._session_keys.get(OLLAMA_LOCAL) is None
    print("  ollama cloud request shape + keys OK")


def test_ollama_model_selection_rules() -> None:
    """Only a *foreign* model ID is swapped for the provider default; local
    quantised tags (what the availability check actually lists) pass through."""
    resolve = metadata_providers._resolve_model
    # No hard-coded name that Ollama does not report as image-capable.
    assert DEFAULT_MODELS[OLLAMA_LOCAL] in OLLAMA_LOCAL_MODELS
    assert DEFAULT_MODELS[OLLAMA_CLOUD] in OLLAMA_CLOUD_MODELS
    assert "gemma4:31b" in OLLAMA_CLOUD_MODELS      # vision on the cloud tag list
    assert resolve(_ollama_settings(OLLAMA_LOCAL, model="")) == DEFAULT_MODELS[OLLAMA_LOCAL]
    assert resolve(_ollama_settings(OLLAMA_LOCAL, model="gemma3:12b")) == "gemma3:12b"
    assert resolve(_ollama_settings(OLLAMA_LOCAL, model="gemma4:31b")) == "gemma4:31b"
    assert resolve(_ollama_settings(OLLAMA_CLOUD, model=OLLAMA_LOCAL_MODELS[0])) \
        == DEFAULT_MODELS[OLLAMA_CLOUD]
    # Gemini/Groq IDs never leak into an Ollama request (instant 404).
    assert resolve(_ollama_settings(OLLAMA_CLOUD, model="qwen/qwen3.8-27b")) \
        == DEFAULT_MODELS[OLLAMA_CLOUD]
    print("  ollama model selection rules OK")


def test_ollama_keyword_target() -> None:
    """Ollama asks for exactly the Adobe-complete keyword set; hosted providers
    keep their previous 'up to N' wording and 45-keyword default."""
    assert f"exactly {ADOBE_KEYWORDS_MAX}" in build_prompt(_ollama_settings())
    assert "up to 45" in build_prompt(MetadataSettings(provider="Gemini", keyword_count=45))

    payload = ('{"title": "T", "description": "D", "keywords": ['
               + ", ".join(f'"k{i}"' for i in range(60)) + "]}")
    meta = parse_model_json(payload, _ollama_settings())
    assert len(meta.keywords) == ADOBE_KEYWORDS_MAX, len(meta.keywords)
    assert meta.keywords[0] == "k0"

    fewer = parse_model_json('{"title": "T", "keywords": ["a", "a", "b"]}',
                             _ollama_settings())
    assert fewer.keywords == ["a", "b"], fewer.keywords   # deduped, never padded
    print("  ollama keyword target OK")


def test_ollama_keyword_top_up_completes_the_set() -> None:
    """A model that answers with 45 of the 50 requested keywords gets exactly one
    follow-up for the missing ones; a failing follow-up never discards the
    metadata that already arrived."""
    first = ('{"title": "T", "description": "D", "keywords": ['
             + ", ".join(f'"k{i}"' for i in range(45)) + "]}")
    second = '{"keywords": [' + ", ".join(f'"extra{i}"' for i in range(10)) + "]}"

    with tempfile.TemporaryDirectory() as tmp, OllamaHttp(
            chat=[FakeResp(200, {"message": {"content": first}}),
                  FakeResp(200, {"message": {"content": second}})]) as http:
        meta = metadata_providers.generate_metadata(_image_in(tmp),
                                                   _ollama_settings(), "")
    calls = http.chat_calls()
    assert len(calls) == 2, calls
    top_up = calls[1]["body"]["messages"][0]["content"]
    assert "exactly 5 additional" in top_up, top_up
    assert "k0" in top_up, top_up                      # never repeat what we have
    assert len(meta.keywords) == ADOBE_KEYWORDS_MAX, len(meta.keywords)
    assert meta.keywords[0] == "k0" and "extra0" in meta.keywords, meta.keywords
    assert calls[1]["body"]["format"] == "json" and calls[1]["body"]["think"] is False

    # Follow-up refused → the finished 45-keyword result is kept, not failed.
    with tempfile.TemporaryDirectory() as tmp, OllamaHttp(
            chat=[FakeResp(200, {"message": {"content": first}}),
                  FakeResp(404, {"error": "file model not found"})]) as http:
        meta = metadata_providers.generate_metadata(_image_in(tmp),
                                                   _ollama_settings(), "")
    assert meta.title == "T" and len(meta.keywords) == 45, meta.keywords
    assert len(http.chat_calls()) == 2, http.chat_calls()

    # A reply that ignored the count entirely is NOT re-asked keyword by
    # keyword — that is a regeneration job, not a cheap top-up.
    with tempfile.TemporaryDirectory() as tmp, OllamaHttp(chat=[_ok()]) as http:
        meta = metadata_providers.generate_metadata(_image_in(tmp),
                                                   _ollama_settings(), "")
    assert len(meta.keywords) == 1 and len(http.chat_calls()) == 1, http.chat_calls()
    print("  ollama keyword top-up OK")


def test_ollama_empty_answer_and_think_fallback() -> None:
    """A reasoning model that returns no visible text gives a clear, permanent
    error; a server that rejects the ``think`` field is asked again without it."""
    empty = {"message": {"content": "", "thinking": "lots of hidden reasoning"}}
    with tempfile.TemporaryDirectory() as tmp:
        img = _image_in(tmp)
        with OllamaHttp(chat=[FakeResp(200, empty)]) as http:
            try:
                metadata_providers.generate_metadata(img, _ollama_settings(), "")
                raise AssertionError("expected ProviderError")
            except ProviderError as exc:
                assert "answered with no text" in str(exc), str(exc)
                assert not metadata_providers.is_retryable_error(exc), str(exc)
                assert len(http.chat_calls()) == 1, http.chat_calls()

        # Older build that does not know the field → one retry without it.
        with OllamaHttp(chat=[FakeResp(400, {"error": 'unknown field "think"'}),
                              _ok()]) as http:
            meta = metadata_providers.generate_metadata(img, _ollama_settings(), "")
        assert meta.title == "T"
        calls = http.chat_calls()
        assert len(calls) == 2, calls
        assert "think" in calls[0]["body"] and "think" not in calls[1]["body"]
    print("  ollama empty answer + think fallback OK")


def test_ollama_permanent_failures_are_not_retried() -> None:
    """Missing model, stopped local server, bad cloud credentials, unsupported
    image and invalid settings stop immediately with an actionable message."""
    import requests

    shared = [
        # (trigger, must-appear substrings, retryable?)
        ([FakeResp(404, {"error": "file model not found"})],
         ["has no model", "not installed", "availability check"], False),
        ([FakeResp(400, {"error": "the model does not support images"})],
         ["not accepted", "images"], False),
        ([FakeResp(403, {"error": "forbidden"})], ["API key"], False),
        ([FakeResp(413, {"error": "image too large"})], ["too large"], False),
    ]
    # A local server that is not running is permanent; the same failure in cloud
    # mode is a network blip (covered in the transient test).
    local_only = [
        ([requests.ConnectionError("Connection aborted")],
         ["Ollama is not running", "ollama serve"], False),
    ]
    cloud_only = [
        ([FakeResp(401, {"error": "unauthorized"})],
         ["API key", "settings/keys", "OLLAMA_API_KEY"], False),
        ([FakeResp(402, {"error": "payment required"})], ["plan", "credits"], False),
    ]
    for provider in (OLLAMA_LOCAL, OLLAMA_CLOUD):
        cases = shared + (local_only if provider == OLLAMA_LOCAL else cloud_only)
        for queue, needles, retryable in cases:
            with tempfile.TemporaryDirectory() as tmp, \
                    OllamaHttp(chat=list(queue)) as http:
                img = _image_in(tmp)
                try:
                    metadata_providers.generate_metadata(
                        img, _ollama_settings(provider), "k")
                    raise AssertionError("expected ProviderError")
                except ProviderError as exc:
                    msg = str(exc).lower()
                    for needle in needles:
                        assert needle.lower() in msg, f"{provider}: {needle} !in {exc}"
                    assert len(http.chat_calls()) == 1, \
                        f"{provider}: permanent error retried ({exc})"
                    assert metadata_providers.is_retryable_error(exc) is retryable, str(exc)

    # Invalid settings fail before any network call is made.
    with tempfile.TemporaryDirectory() as tmp, OllamaHttp(chat=[_ok()]) as http:
        try:
            metadata_providers.generate_metadata(
                _image_in(tmp), _ollama_settings(ollama_url="not a server"), "")
            raise AssertionError("expected ProviderError")
        except ProviderError as exc:
            assert "valid ollama server address" in str(exc).lower(), str(exc)
            assert not metadata_providers.is_retryable_error(exc)
            assert http.chat_calls() == [], "bad address reached the network"
    print("  ollama permanent failures OK")


def test_ollama_transient_failures_retry() -> None:
    """Server hiccups retry per request, and an exhausted transient failure stays
    visible to the worker's batch-level automatic retry passes."""
    import requests

    with tempfile.TemporaryDirectory() as tmp:
        img = _image_in(tmp)
        s = _ollama_settings()

        with OllamaHttp(chat=[FakeResp(503, {"error": "model is loading"}),
                              FakeResp(429, {"error": "too many requests"}),
                              _ok()]) as http:
            assert metadata_providers.generate_metadata(img, s, "").title == "T"
            assert len(http.chat_calls()) == 3, http.chat_calls()

        with OllamaHttp(chat=[FakeResp(503, {"error": "server is overloaded"})]) as http:
            try:
                metadata_providers.generate_metadata(img, s, "")
                raise AssertionError("expected ProviderError")
            except ProviderError as exc:
                assert "failed after 3 attempts" in str(exc), str(exc)
                assert metadata_providers.is_retryable_error(exc), str(exc)
                assert len(http.chat_calls()) == 3, http.chat_calls()

        # A cold local model may simply answer slowly: keep retrying.
        with OllamaHttp(chat=[requests.Timeout("read timed out")]):
            try:
                metadata_providers.generate_metadata(img, s, "")
                raise AssertionError("expected ProviderError")
            except ProviderError as exc:
                assert "still be loading" in str(exc), str(exc)
                assert metadata_providers.is_retryable_error(exc), str(exc)

        # Cloud connectivity blips are temporary, unlike a dead local server.
        with OllamaHttp(chat=[requests.ConnectionError("no route to host")]):
            try:
                metadata_providers.generate_metadata(
                    img, _ollama_settings(OLLAMA_CLOUD), "k")
                raise AssertionError("expected ProviderError")
            except ProviderError as exc:
                assert "Ollama Cloud" in str(exc), str(exc)
                assert metadata_providers.is_retryable_error(exc), str(exc)
    print("  ollama transient retries OK")


def test_ollama_vision_model_discovery() -> None:
    """Only image-capable models are offered: the cloud reports capabilities in
    /api/tags, a local server is asked per model through /api/show."""
    tags = {"models": [{"name": "gemma4:latest"}, {"name": "qwen3:8b"}]}
    show = {"gemma4:latest": FakeResp(200, {"capabilities": ["completion", "vision"]}),
            "qwen3:8b": FakeResp(200, {"capabilities": ["completion"]})}

    with OllamaHttp(tags=FakeResp(200, tags), show=show) as http:
        models, msg = metadata_providers.check_ollama(OLLAMA_LOCAL_URL)
    assert models == ["gemma4:latest"], models
    assert "image-capable" in msg, msg
    asked = [c["body"]["model"] for c in http.calls if c["url"].endswith("/api/show")]
    assert asked == ["gemma4:latest", "qwen3:8b"], asked

    # Closing the app mid-check: the stop flag stops the per-model lookups.
    with OllamaHttp(tags=FakeResp(200, tags), show=show) as http:
        models, msg = metadata_providers.check_ollama(
            OLLAMA_LOCAL_URL, stopped=lambda: True)
    assert models == [] and "stopped" in msg.lower(), msg
    assert not [c for c in http.calls if c["url"].endswith("/api/show")], http.calls

    # A local server that is simply not running.
    with OllamaHttp(tags=requests_error()) as http:
        models, msg = metadata_providers.check_ollama(OLLAMA_LOCAL_URL)
    assert models == []
    assert "no ollama server answered" in msg.lower(), msg
    assert "start the ollama app" in msg.lower(), msg

    # Cloud: capabilities ride on the tag list, and the key authorises it.
    cloud_tags = {"models": [
        {"name": "gemma4:31b", "capabilities": ["completion", "vision"]},
        {"name": "glm-5.2", "capabilities": ["completion", "tools"]},
    ]}
    with OllamaHttp(tags=FakeResp(200, cloud_tags)) as http:
        models, msg = metadata_providers.check_ollama(
            OLLAMA_CLOUD_URL, "ol-key", cloud=True)
    assert models == ["gemma4:31b"], models
    tags_req = http.calls[0]
    assert tags_req["url"] == f"{OLLAMA_CLOUD_URL}/api/tags", tags_req
    assert tags_req["headers"]["Authorization"] == "Bearer ol-key"
    assert "skipped (no image support)" in msg, msg
    assert not [c for c in http.calls if c["url"].endswith("/api/show")], \
        "cloud capabilities must not need per-model lookups"

    # Cloud without a key is explained instead of attempted.
    models, msg = metadata_providers.check_ollama(OLLAMA_CLOUD_URL, "", cloud=True)
    assert models == [] and "needs an api key" in msg.lower(), msg

    # Rejected cloud credentials, and a model-less server.
    with OllamaHttp(tags=FakeResp(401, {"error": "bad key"})):
        models, msg = metadata_providers.check_ollama(OLLAMA_CLOUD_URL, "nope", cloud=True)
    assert models == [] and "rejected your api key" in msg.lower(), msg

    with OllamaHttp(tags=FakeResp(200, {"models": [{"name": "qwen3:8b"}]}),
                    show={"qwen3:8b": FakeResp(200, {"capabilities": ["completion"]})}):
        models, msg = metadata_providers.check_ollama(OLLAMA_LOCAL_URL)
    assert models == [] and "can analyse images" in msg, msg
    print("  ollama vision model discovery OK")


def requests_error():
    import requests

    return requests.ConnectionError("connection refused")


def test_worker_ollama_batch_keeps_successes(app: QApplication) -> None:
    """Ollama batch behaviour is unchanged: one request per image, successes kept
    and never regenerated, automatic retry passes for temporary failures only,
    and permanent model/server errors retried zero times."""
    from app.engine.metadata_worker import MetadataWorker, MAX_BATCH_PASSES
    import app.engine.metadata_worker as mw

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        names = ["flaky.png", "noserver.png", "nomodel.png", "steady.png"]
        for n in names:
            Image.new("RGB", (8, 8), (0, 0, 0)).save(root / n)

        calls: dict[str, int] = {}

        def fake_generate(path, settings, api_key, log=None):
            assert settings.provider == OLLAMA_LOCAL and api_key == ""
            name = Path(path).name
            calls[name] = calls.get(name, 0) + 1
            if name == "flaky.png" and calls[name] == 1:
                raise ProviderError("Local Ollama rate limit reached (HTTP 429)",
                                    code=429)
            if name == "noserver.png":
                raise ProviderError("Ollama is not running on this computer "
                                    "(http://localhost:11434) — start the Ollama app")
            if name == "nomodel.png":
                raise ProviderError("Local Ollama has no model 'gemma4' "
                                    "(HTTP 404) — not installed", code=404)
            return ImageMetadata(f"Meta {name}", "d", ["k"])

        originals = (mw.generate_metadata, mw.time.sleep)
        mw.generate_metadata = fake_generate
        mw.time.sleep = lambda s: None
        try:
            worker = MetadataWorker([root / n for n in names], _ollama_settings(), "")
            done: dict[str, str] = {}
            errs: dict[str, str] = {}
            passes: list = []
            counts: dict[str, int] = {}
            worker.item_done.connect(lambda p, m: done.update({Path(p).name: m["title"]}))
            worker.item_failed.connect(lambda p, e: errs.update({Path(p).name: e}))
            worker.item_started.connect(
                lambda p, i, n: counts.update({Path(p).name: i}))
            worker.retry_pass.connect(lambda *a: passes.append(a))
            worker.run()
        finally:
            mw.generate_metadata, mw.time.sleep = originals

        assert calls["flaky.png"] == 2, calls                 # healed on pass 2
        assert calls["steady.png"] == 1, calls                # never regenerated
        assert calls["noserver.png"] == 1, calls              # permanent: no retry
        assert calls["nomodel.png"] == 1, calls
        assert done.get("flaky.png") == "Meta flaky.png", done
        assert done.get("steady.png") == "Meta steady.png", done
        assert "not running" in errs.get("noserver.png", ""), errs
        assert "has no model" in errs.get("nomodel.png", ""), errs
        # Exactly one automatic retry pass ran — and it carried only the image
        # that failed temporarily, never the permanent ones.
        assert len(passes) == 1, passes
        assert passes[0][0] == 2 and passes[0][1] == MAX_BATCH_PASSES, passes
        assert passes[0][2] == 1, passes                      # only the flaky one
    print("  ollama worker batch retention OK")


def test_page_local_ollama_and_csv_gate(app: QApplication) -> None:
    """Screen level: a local Ollama folder batch starts with no API key at all,
    exports nothing while any image fails, and writes the complete CSV into the
    chosen folder as soon as every image succeeded."""
    from PySide6.QtCore import QEventLoop, QTimer
    from app.ui.metadata_page import MetadataPage, DONE, FAILED
    import app.engine.metadata_worker as mw

    def _drain(page: MetadataPage) -> None:
        loop = QEventLoop()
        page.worker.finished_all.connect(loop.quit)
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        assert page.worker.wait(5000), "worker did not finish"

    orig_gen = mw.generate_metadata
    credential_store._session_keys.clear()
    os.environ.pop("OLLAMA_API_KEY", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "shots"
            root.mkdir()
            for i in (1, 2, 3):
                Image.new("RGB", (8, 8), (i, i, i)).save(root / f"pic{i}.png")

            calls: list[str] = []

            def fake_generate(path, settings, api_key, log=None):
                name = Path(path).name
                calls.append(name)
                assert settings.provider == OLLAMA_LOCAL
                assert api_key == "", "local Ollama must not need a key"
                assert settings.exact_keyword_count, "Ollama targets a full Adobe set"
                assert settings.keyword_count == OLLAMA_KEYWORD_COUNT
                if name == "pic3.png":
                    raise ProviderError("Local Ollama has no model 'gemma4' "
                                        "(HTTP 404) — not installed", code=404)
                return ImageMetadata(f"Title {name}", "d", ["k1", "k2"])

            mw.generate_metadata = fake_generate
            page = MetadataPage()
            page.provider_combo.setCurrentText(OLLAMA_LOCAL)
            assert page.key_box.isHidden(), "local mode must not ask for a key"
            assert not page.ollama_box.isHidden()
            assert not page.ollama_url_widget.isHidden()
            assert page.keywords_spin.value() == OLLAMA_KEYWORD_COUNT

            page._add_paths([root])
            assert len(page.items) == 3
            page.generate_all()              # starts with no key configured
            _drain(page)

            assert {it.path.name for it in page.items.values()
                    if it.status == DONE} == {"pic1.png", "pic2.png"}
            failed = [it for it in page.items.values() if it.status == FAILED]
            assert [it.path.name for it in failed] == ["pic3.png"]
            assert "has no model" in failed[0].error, failed[0].error
            assert not list(root.glob("*.csv")), "partial folder batch exported"

            # Fix the permanent problem, then retry ONLY the failed image.
            def ok_generate(path, settings, api_key, log=None):
                calls.append(Path(path).name)
                return ImageMetadata("Title pic3.png", "d", ["k1", "k2"])

            mw.generate_metadata = ok_generate
            page.retry_failed()
            _drain(page)
            assert calls.count("pic3.png") == 2, calls
            assert calls.count("pic1.png") == 1, calls
            assert all(it.status == DONE for it in page.items.values())
            csvs = sorted(root.glob("*.csv"))
            assert [c.name for c in csvs] == ["metadata.csv"], csvs
            with csvs[0].open(newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.reader(fh))
            assert rows[0] == metadata_export.CSV_HEADERS
            assert len(rows) == 4, rows                     # header + all 3 images
            assert sorted(r[0] for r in rows[1:]) == [
                "pic1.png", "pic2.png", "pic3.png"]
            # Generated metadata stayed visible and editable before export:
            # dropping the selection first guarantees the editor reloads.
            page.grid.setCurrentRow(-1)
            page.grid.setCurrentRow(0)
            assert page.title_edit.text().startswith("Title "), page.title_edit.text()
            page.title_edit.setText("Hand-tuned title")
            key = page.grid.currentItem().data(Qt.UserRole + 1)
            assert page.items[key].meta.title == "Hand-tuned title"
            page.shutdown()
    finally:
        mw.generate_metadata = orig_gen
        credential_store._session_keys.clear()
    print("  page local ollama + all-or-nothing CSV OK")


def test_page_ollama_cloud_needs_key_and_lists_models(app: QApplication) -> None:
    """Cloud mode shows the key field and an upload warning, and the connection
    check feeds the model picker with image-capable cloud models."""
    from PySide6.QtCore import QEventLoop, QTimer
    from app.ui.metadata_page import MetadataPage

    orig_check = metadata_providers.check_ollama
    cloud_tags = {"models": [{"name": "gemma4:31b",
                              "capabilities": ["completion", "vision"]}]}
    try:
        with OllamaHttp(tags=FakeResp(200, cloud_tags)) as http:
            page = MetadataPage()
            page.provider_combo.setCurrentText(OLLAMA_CLOUD)
            assert not page.key_box.isHidden(), "cloud mode must offer a key field"
            assert page.ollama_url_widget.isHidden(), "cloud host is not editable"
            assert "uploaded" in page.ollama_notice.text().lower()

            page.key_edit.setText("ol-key")
            page.check_ollama()
            loop = QEventLoop()
            page.lookup.result.connect(loop.quit)
            QTimer.singleShot(10000, loop.quit)
            loop.exec()
            assert page.lookup.wait(5000), "lookup thread did not finish"
            assert page.model_combo.currentText() == "gemma4:31b", \
                page.model_combo.currentText()
            assert http.calls[0]["headers"]["Authorization"] == "Bearer ol-key"
            assert "upload the image" in page.ollama_notice.text().lower(), \
                page.ollama_notice.text()
            assert "image-capable model" in page.log.toPlainText(), \
                page.log.toPlainText()
            page.shutdown()
    finally:
        metadata_providers.check_ollama = orig_check
        credential_store._session_keys.clear()
    print("  page cloud key + model check OK")


def _release_qt(app: QApplication) -> None:
    """Destroy leftover widgets/threads while QApplication is still alive —
    parentless test pages finalized after Qt shuts down crash the process."""
    for w in list(app.topLevelWidgets()):
        w.close()
        w.deleteLater()
    for _ in range(8):
        app.processEvents()


def main() -> int:
    app = QApplication.instance() or QApplication([])
    try:
        test_ollama_local_request_shape()
        test_ollama_cloud_request_shape_and_keys()
        test_ollama_model_selection_rules()
        test_ollama_keyword_target()
        test_ollama_keyword_top_up_completes_the_set()
        test_ollama_empty_answer_and_think_fallback()
        test_ollama_permanent_failures_are_not_retried()
        test_ollama_transient_failures_retry()
        test_ollama_vision_model_discovery()
        test_worker_ollama_batch_keeps_successes(app)
        test_page_local_ollama_and_csv_gate(app)
        test_page_ollama_cloud_needs_key_and_lists_models(app)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"OLLAMA TEST FAILED: {exc}")
        _release_qt(app)
        return 1
    _release_qt(app)
    print("OLLAMA TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
