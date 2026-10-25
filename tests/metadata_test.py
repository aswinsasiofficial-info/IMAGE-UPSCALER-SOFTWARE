"""Tests for the Metadata Generator (no API keys, no network).

Run with:  python -m tests.metadata_test

Covers the pure engine pieces (parsing/sanitising, image pre-flight, exports,
credential lookup, retry classification), the worker with a stubbed provider,
and an offscreen smoke build of both feature screens behind the nav.
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from app.engine.metadata_settings import (
    MetadataSettings, ImageMetadata, ProviderError, ADOBE_TITLE_MAX,
    parse_model_json, sanitize_keywords, prepare_analysis_bytes,
    DEFAULT_MODELS, GEMINI_MODELS, GROQ_MODELS,
)
from app.engine import metadata_providers
from app.engine import metadata_export
from app.engine import credential_store


def _settings(**kw) -> MetadataSettings:
    base = dict(provider="Gemini", keyword_count=10, language="English")
    base.update(kw)
    return MetadataSettings(**base)


def test_parse_and_sanitize() -> None:
    s = _settings()
    # Plain JSON.
    meta = parse_model_json('{"title": "Sunset Beach", "description": "Dusk over sand.", '
                            '"keywords": ["beach", "sunset", "Beach", "", "ocean"]}', s)
    assert meta.title == "Sunset Beach"
    assert meta.keywords == ["beach", "sunset", "ocean"], meta.keywords  # dedup + drop empty

    # Markdown fences and surrounding prose must be tolerated.
    meta = parse_model_json('Sure!\n```json\n{"title": "T", "description": "", '
                            '"keywords": ["a"]}\n```', s)
    assert meta.title == "T" and meta.keywords == ["a"]

    # No JSON at all -> ValueError.
    try:
        parse_model_json("I cannot help with that.", s)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # Title clamped to Adobe limit; keyword count capped by the setting.
    import json
    long_title = "W" * (ADOBE_TITLE_MAX + 50)
    payload = json.dumps({"title": long_title, "description": "d",
                          "keywords": ["k%d" % i for i in range(30)]})
    meta = parse_model_json(payload, s)
    assert len(meta.title) <= ADOBE_TITLE_MAX
    assert len(meta.keywords) == 10

    # Description disabled -> dropped even if the model returned one.
    meta = parse_model_json('{"title": "T", "description": "kept?", "keywords": []}',
                            _settings(generate_description=False))
    assert meta.description == ""

    assert sanitize_keywords([" Alpha ", "alpha", "ALPHA", "beta"], 5) == ["Alpha", "beta"]
    print("  parse/sanitize OK")


def test_prepare_analysis_bytes(path_folder: Path) -> None:
    # Large RGBA PNG: downscaled to <=1024 long edge, kept as PNG (API-supported).
    src = path_folder / "big.png"
    arr = np.zeros((1600, 900, 4), dtype=np.uint8)
    arr[..., :3] = np.random.default_rng(2).integers(0, 255, (1600, 900, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGBA").save(src)
    before = src.read_bytes()

    data, mime = prepare_analysis_bytes(src)
    assert mime == "image/png", mime
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(io.BytesIO(data)) as im:
        assert max(im.size) <= 1024
    assert src.read_bytes() == before  # file on disk untouched

    # JPEG stays JPEG.
    jpg = path_folder / "photo.jpg"
    Image.new("RGB", (2000, 500), (10, 20, 30)).save(jpg)
    data, mime = prepare_analysis_bytes(jpg)
    assert mime == "image/jpeg" and data[:2] == b"\xff\xd8"
    with Image.open(io.BytesIO(data)) as im:
        assert max(im.size) <= 1024

    # WEBP stays WEBP — MIME comes from the decoded file, not the extension.
    webp = path_folder / "shot.webp"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(webp)
    data, mime = prepare_analysis_bytes(webp)
    assert mime == "image/webp" and data[0:4] == b"RIFF" and data[8:12] == b"WEBP"

    # Decodable but not API-supported (BMP with alpha) → converted to JPEG.
    bmp = path_folder / "old.bmp"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(bmp)
    data, mime = prepare_analysis_bytes(bmp)
    assert mime == "image/jpeg" and data[:2] == b"\xff\xd8"

    # Unreadable/non-image → clear ProviderError naming the supported formats.
    bad = path_folder / "broken.jpg"
    bad.write_bytes(b"definitely not image bytes")
    try:
        prepare_analysis_bytes(bad)
        raise AssertionError("expected ProviderError")
    except ProviderError as exc:
        assert "not a readable image" in str(exc), str(exc)

    # Missing file → ProviderError too (no raw OSError escape).
    try:
        prepare_analysis_bytes(path_folder / "gone.png")
        raise AssertionError("expected ProviderError")
    except ProviderError as exc:
        assert "Could not process image" in str(exc), str(exc)
    print("  prepare_analysis_bytes OK")


def test_csv_export(path_folder: Path) -> None:
    rows = [
        (Path("a,b.png"), ImageMetadata("Nice, Title", "Desc with, commas", ["k1", "k2"])),
        (Path("c.jpg"), ImageMetadata("", "", [])),
    ]
    dest = path_folder / "out" / "metadata.csv"
    metadata_export.write_csv(rows, dest)
    with dest.open(newline="", encoding="utf-8-sig") as fh:
        parsed = list(csv.reader(fh))
    assert parsed[0] == metadata_export.CSV_HEADERS
    assert parsed[1][:2] == ["a,b.png", "Nice, Title"]
    assert parsed[1][3] == "k1, k2"
    print("  csv export OK")


def test_embed_roundtrip(path_folder: Path) -> None:
    meta = ImageMetadata("Golden Hour", "Warm light over hills", ["sunset", "hills", "gold"])

    # PNG in place.
    png = path_folder / "img.png"
    Image.new("RGB", (16, 12), (200, 30, 30)).save(png)
    metadata_export.embed_metadata(png, meta)
    got = metadata_export.read_embedded(png)
    assert got["title"] == "Golden Hour" and got["keywords"] == ["sunset", "hills", "gold"]

    # JPEG into a copy — original must not gain the comment.
    jpg = path_folder / "img.jpg"
    Image.new("RGB", (16, 12), (30, 200, 30)).save(jpg)
    copy = path_folder / "copies" / "img.jpg"
    metadata_export.embed_metadata(jpg, meta, copy)
    assert copy.exists() and metadata_export.read_embedded(jpg).get("title", "") == ""
    assert metadata_export.read_embedded(copy)["title"] == "Golden Hour"
    with Image.open(copy) as im:
        assert im.size == (16, 12)

    # No stray temp files left behind from the in-place swap.
    assert not list(path_folder.glob("tmp*.png")), list(path_folder.glob("tmp*"))

    # Unsupported format raises a clear error.
    tif = path_folder / "img.tif"
    Image.new("RGB", (8, 8)).save(tif)
    try:
        metadata_export.embed_metadata(tif, meta)
        raise AssertionError("expected ValueError for .tif")
    except ValueError as exc:
        assert "PNG and JPEG" in str(exc)
    print("  embed roundtrip OK")


def test_credential_store() -> None:
    provider = "Gemini"
    credential_store._session_keys.clear()
    os.environ.pop("GEMINI_API_KEY", None)
    credential_store.set_session_key(provider, "  secret123 ")
    assert credential_store.get_key(provider) == "secret123"
    os.environ["GEMINI_API_KEY"] = "env-key"
    credential_store._session_keys.clear()
    assert credential_store.get_key(provider) == "env-key"  # env beats keyring, after session
    del os.environ["GEMINI_API_KEY"]
    print("  credential store OK")


def test_retry_classification() -> None:
    assert metadata_providers._is_transient(ProviderError("boom", code=429))
    assert metadata_providers._is_transient(ProviderError("HTTP 503 unavailable", code=503))
    assert not metadata_providers._is_transient(ProviderError("bad key", code=401))
    assert not metadata_providers._is_transient(
        ProviderError("404 This model models/gemini-x is no longer available. "
                      "Please update your code to use models/gemini-y"))
    hint = metadata_providers._retirement_hint(ProviderError(
        "404 This model models/old-1 is no longer available. "
        "Please update your code to use models/new-2'."))
    assert "new-2" in hint, hint
    assert metadata_providers._retirement_hint(ProviderError("plain 404")) == ""
    # Explicit retryable flag survives wrapping (set by generate_metadata).
    flagged = ProviderError("Groq failed after 3 attempts: nope")
    flagged.retryable = True
    assert metadata_providers._is_transient(flagged)
    assert metadata_providers.is_retryable_error(flagged)
    print("  retry classification OK")


def test_error_classification() -> None:
    # Each failure family gets its own distinguishable wording.
    assert "API key" in metadata_providers._classify_http("Groq", 401, "x", "m")
    assert "API key" in metadata_providers._classify_http("Gemini", 403, "x", "m")
    msg = metadata_providers._classify_http("Groq", 404, "x", "old/model")
    assert "old/model" in msg and "retired" in msg
    assert "rate limit" in metadata_providers._classify_http("Gemini", 429, "x", "m")
    assert "image" in metadata_providers._classify_http("Gemini", 400, "x", "m")
    assert "503" in metadata_providers._classify_http("Groq", 503, "upstream", "m")

    # Keys stay provider-specific: the missing-key error names only the
    # environment variable of the currently selected provider.
    for provider, env in (("Gemini", "GEMINI_API_KEY"), ("Groq", "GROQ_API_KEY")):
        try:
            metadata_providers.generate_metadata("x.png", _settings(provider=provider), "")
            raise AssertionError("expected ProviderError")
        except ProviderError as exc:
            assert env in str(exc) and ("GROQ" if provider == "Gemini" else "GEMINI") not in str(exc)
    print("  error classification OK")


def test_model_catalogues() -> None:
    # Every listed model must be a current, vision-capable ID.
    assert DEFAULT_MODELS["Gemini"] in GEMINI_MODELS
    assert DEFAULT_MODELS["Groq"] in GROQ_MODELS
    assert GROQ_MODELS == ["qwen/qwen3.8-27b"], GROQ_MODELS  # llama-4 IDs were removed
    assert "gemini-2.0-flash" not in GEMINI_MODELS           # shut down

    # A stale/foreign model choice falls back to the selected provider's own
    # default — a Groq run must never send a Gemini model ID (instant 404).
    resolve = metadata_providers._resolve_model
    assert resolve(_settings(provider="Groq", model=GEMINI_MODELS[0])) == DEFAULT_MODELS["Groq"]
    assert resolve(_settings(provider="Groq", model="")) == DEFAULT_MODELS["Groq"]
    assert resolve(_settings(provider="Groq", model="custom/vision-model")) == "custom/vision-model"
    assert resolve(_settings(provider="Gemini", model=GROQ_MODELS[0])) == DEFAULT_MODELS["Gemini"]
    print("  model catalogues OK")


def test_gemini_request_payload() -> None:
    """Gemini must receive the image as inline bytes via Part.from_bytes.

    google-genai >= 2.x rejects ``types.Image`` objects in ``contents`` with
    "file uri and mime_type are required" — the SDK's own t_contents (run by
    the fake client below) is what guards that regression.
    """
    from google import genai
    from google.genai import _transformers as t

    captured: dict = {}
    reply = '{"title": "T", "description": "D", "keywords": ["a"]}'

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            content = t.t_contents(kwargs["contents"])[0]  # raises on bad shapes
            blob = content.parts[1].inline_data
            captured["sent_mime"] = blob.mime_type
            captured["sent_data"] = blob.data

            class R:
                text = reply
            return R()

    class FakeClient:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.models = FakeModels()

    original = genai.Client
    genai.Client = FakeClient
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.webp"
            Image.new("RGB", (16, 16), (9, 9, 9)).save(p)
            meta = metadata_providers.generate_metadata(
                p, _settings(provider="Gemini", model="gemini-3.8-flash"), "g-key")
    finally:
        genai.Client = original

    assert meta.title == "T"
    assert captured["api_key"] == "g-key"
    assert captured["model"] == "gemini-3.8-flash"
    assert captured["sent_mime"] == "image/webp"      # derived from the file
    assert captured["sent_data"][:4] == b"RIFF"       # real bytes were sent
    assert "Adobe Stock" in captured["contents"][0]   # prompt travels as text
    print("  gemini request payload OK")


def test_groq_request_payload_and_http_errors() -> None:
    import base64
    import requests as _requests

    captured: dict = {}
    queue: list = []
    ok_body = {"choices": [{"message": {"content":
               '{"title": "T", "description": "", "keywords": ["a"]}'}}]}

    class FakeResp:
        def __init__(self, status=200, payload=None, text=""):
            self.status_code, self._payload, self.text, self.reason = status, payload, text, "Error"
        def json(self):
            if self._payload is None:
                raise ValueError("no json")
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        action = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(action, Exception):
            raise action
        return action

    original_post, original_sleep = _requests.post, metadata_providers.time.sleep
    _requests.post, metadata_providers.time.sleep = fake_post, lambda s: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.png"
            Image.new("RGB", (16, 16), (1, 2, 3)).save(p)
            s = _settings(provider="Groq")

            # Happy path: endpoint, key header, vision model, data URI with
            # the PNG's real MIME, and decodable image bytes.
            queue.append(FakeResp(200, ok_body))
            meta = metadata_providers.generate_metadata(p, s, "gq-key")
            assert meta.title == "T"
            assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
            assert captured["headers"]["Authorization"] == "Bearer gq-key"
            assert captured["body"]["model"] == DEFAULT_MODELS["Groq"]
            data_url = captured["body"]["messages"][0]["content"][1]["image_url"]["url"]
            assert data_url.startswith("data:image/png;base64,"), data_url[:40]
            raw = base64.b64decode(data_url.split(",", 1)[1])
            assert raw[:8] == b"\x89PNG\r\n\x1a\n"

            # Bad key → authentication wording, not retried.
            queue.clear(); queue.append(FakeResp(401, {"error": {"message": "invalid API key"}}))
            try:
                metadata_providers.generate_metadata(p, s, "bad")
                raise AssertionError("expected ProviderError")
            except ProviderError as exc:
                assert "API key" in str(exc) and exc.code == 401

            # Missing model → names the model and the fix, not retried.
            queue.clear(); queue.append(FakeResp(404, {"error": {"message": "model does not exist"}}))
            try:
                metadata_providers.generate_metadata(p, s, "gq-key")
                raise AssertionError("expected ProviderError")
            except ProviderError as exc:
                assert DEFAULT_MODELS["Groq"] in str(exc) and "retired" in str(exc)
                assert exc.code == 404

            # Rate limited twice, then success — backoff retries absorb it.
            queue.clear()
            queue.extend([FakeResp(429, {"error": {"message": "rate limit"}}),
                          FakeResp(429, {"error": {"message": "rate limit"}}),
                          FakeResp(200, ok_body)])
            assert metadata_providers.generate_metadata(p, s, "gq-key").title == "T"

            # Network failure → distinguishable, connection-focused message.
            queue.clear(); queue.append(_requests.ConnectionError("no route to host"))
            try:
                metadata_providers.generate_metadata(p, s, "gq-key")
                raise AssertionError("expected ProviderError")
            except ProviderError as exc:
                assert "internet" in str(exc) and "failed after" in str(exc)
                # Regression: the exhausted transient failure must stay visible
                # to the worker's batch-level automatic retry passes.
                assert metadata_providers.is_retryable_error(exc), str(exc)

            # Persistent 429 (all per-request attempts used) → final error's
            # message no longer hints "rate limit", but the retryable flag
            # must keep the batch retry passes alive.
            queue.clear(); queue.append(FakeResp(429, {"error": {"message": "rate limit"}}))
            try:
                metadata_providers.generate_metadata(p, s, "gq-key")
                raise AssertionError("expected ProviderError")
            except ProviderError as exc:
                assert "failed after" in str(exc)
                assert metadata_providers.is_retryable_error(exc), str(exc)

            # Permanent failure (bad key) must NOT be flagged retryable.
            queue.clear(); queue.append(FakeResp(401, {"error": {"message": "invalid API key"}}))
            try:
                metadata_providers.generate_metadata(p, s, "bad")
                raise AssertionError("expected ProviderError")
            except ProviderError as exc:
                assert not metadata_providers.is_retryable_error(exc), str(exc)
    finally:
        _requests.post, metadata_providers.time.sleep = original_post, original_sleep
    print("  groq request payload + http errors OK")


def test_worker_with_stubbed_provider(app: QApplication) -> None:
    from app.engine.metadata_worker import MetadataWorker
    import app.engine.metadata_worker as mw

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = root / "good.png"
        bad = root / "bad.png"
        Image.new("RGB", (8, 8), (0, 0, 0)).save(good)
        Image.new("RGB", (8, 8), (0, 0, 0)).save(bad)

        def fake_generate(path, settings, api_key, log=None):
            if Path(path).name == "bad.png":
                raise ProviderError("kaboom")
            return ImageMetadata("T", "D", ["one", "two"])

        original = mw.generate_metadata
        mw.generate_metadata = fake_generate
        try:
            worker = MetadataWorker([good, bad], _settings(), "key")
            results: dict[str, str] = {}
            errors: dict[str, str] = {}
            worker.item_done.connect(lambda p, m: results.update({p: m["title"]}))
            worker.item_failed.connect(lambda p, e: errors.update({p: e}))
            worker.run()  # synchronous
        finally:
            mw.generate_metadata = original

        assert results.get(str(good)) == "T", results
        assert errors.get(str(bad)) == "kaboom", errors
    print("  worker (stubbed provider) OK")


def test_worker_auto_retry_passes(app: QApplication) -> None:
    """Automatic batch retry: temporary failures are re-run in later passes
    (only them), permanent errors are never retried, and the retry limit
    finally marks unresolved images failed for the manual Retry Failed."""
    from app.engine.metadata_worker import (
        MetadataWorker, MAX_BATCH_PASSES, PASS_BACKOFF_BASE)
    import app.engine.metadata_worker as mw

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        names = ["flaky.png", "permanent.png", "doomed.png", "steady.png"]
        for n in names:
            Image.new("RGB", (8, 8), (0, 0, 0)).save(root / n)

        calls: dict[str, int] = {}

        def fake_generate(path, settings, api_key, log=None):
            name = Path(path).name
            calls[name] = calls.get(name, 0) + 1
            if name == "flaky.png" and calls[name] == 1:
                raise ProviderError("Gemini rate limit reached (HTTP 429) — "
                                    "too many requests", code=429)
            if name == "permanent.png":
                raise ProviderError("Gemini rejected your API key (HTTP 401) "
                                    "— check the key", code=401)
            if name == "doomed.png":
                raise ProviderError("Groq request timed out after 120s — retry",  # transient
                                    code=None)
            return ImageMetadata(f"Meta {name}", "d", ["k"])

        originals = (mw.generate_metadata, mw.time.sleep)
        mw.generate_metadata = fake_generate
        mw.time.sleep = lambda s: None   # skip the inter-pass backoff
        try:
            worker = MetadataWorker([root / n for n in names], _settings(), "key")
            done: dict[str, str] = {}
            errs: dict[str, str] = {}
            passes: list[tuple[int, int, int, float]] = []
            result: dict[str, int] = {}
            worker.item_done.connect(lambda p, m: done.update({Path(p).name: m["title"]}))
            worker.item_failed.connect(lambda p, e: errs.update({Path(p).name: e}))
            worker.retry_pass.connect(lambda n, t, c, w: passes.append((n, t, c, w)))
            worker.finished_all.connect(lambda o, f: result.update(ok=o, failed=f))
            worker.run()  # synchronous
        finally:
            mw.generate_metadata, mw.time.sleep = originals

        # Partial success is kept; the flaky image succeeded on pass 2 only.
        assert done.get("flaky.png") == "Meta flaky.png", done
        assert calls["flaky.png"] == 2, calls
        assert calls["steady.png"] == 1, calls        # never regenerated
        assert "flaky.png" not in errs and "steady.png" not in errs
        # Permanent error: reported with its clear message, retried zero times.
        assert calls["permanent.png"] == 1, calls
        assert "API key" in errs.get("permanent.png", ""), errs
        # Endless-loop guard: retried every pass, then gave up at the limit.
        assert calls["doomed.png"] == MAX_BATCH_PASSES, calls
        assert "gave up" in errs.get("doomed.png", ""), errs
        # Two retry passes ran, retrying ONLY the transient failures (flaky +
        # doomed on pass 2; only doomed left on pass 3) with increasing waits.
        assert len(passes) == MAX_BATCH_PASSES - 1, passes
        assert [p[0] for p in passes] == [2, 3], passes
        assert all(p[1] == MAX_BATCH_PASSES for p in passes)
        assert [p[2] for p in passes] == [2, 1], passes
        assert [p[3] for p in passes] == [
            PASS_BACKOFF_BASE, PASS_BACKOFF_BASE * 2], passes
        assert result == {"ok": 2, "failed": 2}, result
    print("  worker auto-retry passes OK")


def test_worker_cancel_stops_retries(app: QApplication) -> None:
    """Cancel during a batch must stop everything: no retry pass starts and
    images queued for automatic retry are finalised as cancelled."""
    from app.engine.metadata_worker import MetadataWorker
    import app.engine.metadata_worker as mw

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        one, two = root / "one.png", root / "two.png"
        Image.new("RGB", (8, 8), (0, 0, 0)).save(one)
        Image.new("RGB", (8, 8), (0, 0, 0)).save(two)

        calls: list[str] = []
        worker = MetadataWorker([one, two], _settings(), "key")

        def fake_generate(path, settings, api_key, log=None):
            calls.append(Path(path).name)
            worker.request_cancel()          # user hits Cancel right away
            raise ProviderError("Groq rate limit reached (HTTP 429) — too many "
                                "requests", code=429)

        originals = (mw.generate_metadata, mw.time.sleep)
        mw.generate_metadata = fake_generate
        mw.time.sleep = lambda s: None
        try:
            done: dict[str, str] = {}
            errs: dict[str, str] = {}
            passes: list = []
            was_cancelled: list[bool] = []
            worker.item_done.connect(lambda p, m: done.update({Path(p).name: "y"}))
            worker.item_failed.connect(lambda p, e: errs.update({Path(p).name: e}))
            worker.retry_pass.connect(lambda *a: passes.append(a))
            worker.cancelled.connect(lambda: was_cancelled.append(True))
            worker.run()
        finally:
            mw.generate_metadata, mw.time.sleep = originals

        assert calls == ["one.png"], calls                 # pass 1 stopped
        assert done == {}                                  # nothing succeeded
        assert "Cancelled" in errs.get("one.png", ""), errs  # finalised, not retried
        assert passes == [] and was_cancelled == [True]    # no retry pass after cancel
    print("  worker cancel stops retries OK")


def test_ui_smoke(app: QApplication) -> None:
    from app.ui.main_window import MainWindow
    from app.ui.metadata_page import MetadataPage, PENDING

    win = MainWindow()
    assert win.stack.count() == 2
    win.stack.setCurrentWidget(win.metadata_page)
    assert win.act_metadata.isChecked()

    page: MetadataPage = win.metadata_page
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.png"
        Image.new("RGB", (10, 10), (1, 2, 3)).save(p)
        page._add_paths([p])
        key = page._key(p)
        assert key in page.items and page.items[key].status == PENDING
        assert page.grid.count() == 1

        # Editing flows back into the item model.
        page.grid.setCurrentRow(0)
        page.title_edit.setText("My Title")
        page.kw_edit.setText("a, b ,c")
        assert page.items[key].meta.title == "My Title"
        assert page.items[key].meta.keywords == ["a", "b", "c"]

        # Status painting.
        page._set_status(key, "done", meta={"title": "G", "description": "", "keywords": ["z"]})
        assert page.items[key].meta.title == "G"
    win.close()
    print("  UI smoke OK")


def test_editor_switch_keeps_metadata_per_item(app: QApplication) -> None:
    """Selecting the next image must not paste the previous image's editor text
    over the newly selected image's generated metadata."""
    from app.ui.metadata_page import MetadataPage, DONE

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a, b = root / "a.png", root / "b.png"
        Image.new("RGB", (8, 8), (1, 1, 1)).save(a)
        Image.new("RGB", (8, 8), (2, 2, 2)).save(b)
        page = MetadataPage()
        page._add_paths([a, b])
        ka, kb = page._key(a), page._key(b)
        page._set_status(ka, DONE, meta={"title": "Alpha", "description": "da",
                                         "keywords": ["x"]})
        page._set_status(kb, DONE, meta={"title": "Beta", "description": "db",
                                         "keywords": ["y"]})

        page.grid.setCurrentItem(page._grid_item(ka))
        assert page.title_edit.text() == "Alpha", page.title_edit.text()
        page.grid.setCurrentItem(page._grid_item(kb))
        assert page.title_edit.text() == "Beta", page.title_edit.text()
        assert page.items[ka].meta.title == "Alpha"
        # Manual edits land on the image that is actually shown…
        page.title_edit.setText("Beta edited")
        assert page.items[kb].meta.title == "Beta edited"
        # …and survive moving to another row.
        page.grid.setCurrentItem(page._grid_item(ka))
        assert page.items[kb].meta.title == "Beta edited"
        assert page.items[ka].meta.title == "Alpha"
        page.shutdown()
    print("  editor per-item isolation OK")


def test_page_batch_and_auto_csv(app: QApplication) -> None:
    """A folder batch auto-exports its CSV only when EVERY image succeeds; a
    batch with any failure writes nothing (successful/failed rows stay visible
    for retry), and once Retry Failed makes the batch fully green the complete
    CSV is exported without overwriting. File-only batches keep the manual
    export workflow."""
    from PySide6.QtCore import QEventLoop, QTimer
    from app.ui.metadata_page import MetadataPage, DONE, FAILED
    import app.engine.metadata_worker as mw

    def _drain(page: MetadataPage) -> None:
        loop = QEventLoop()
        page.worker.finished_all.connect(loop.quit)
        QTimer.singleShot(10000, loop.quit)   # safety net: never hang CI
        loop.exec()
        assert page.worker.wait(5000), "worker did not finish"

    orig_gen = mw.generate_metadata
    credential_store._session_keys.clear()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "shots"
            root.mkdir()
            for i in (1, 2, 3):
                Image.new("RGB", (8, 8), (i, i, i)).save(root / f"pic{i}.png")
            sub = root / "sub"
            sub.mkdir()
            Image.new("RGB", (8, 8), (4, 4, 4)).save(sub / "deep.png")
            # A pre-existing CSV must never be silently overwritten.
            (root / "metadata.csv").write_text("existing", encoding="utf-8")

            def fake_generate(path, settings, api_key, log=None):
                if Path(path).name == "deep.png":
                    raise ProviderError("stub failure")
                return ImageMetadata(f"Title {Path(path).stem}", "d", ["k1"])

            mw.generate_metadata = fake_generate
            page = MetadataPage()
            page._add_paths([root])                # same as Add Folder…
            assert len(page.items) == 4, len(page.items)
            credential_store.set_session_key(page.provider_combo.currentText(), "test-key")

            page.generate_all()
            assert page.worker is not None, "worker not started"
            assert not page.generate_btn.isEnabled()   # double-click protection
            # A second start request while running must be ignored.
            started = page.worker
            class Running:
                def isRunning(self): return True
            page.worker = Running()
            page.generate_all()
            assert not isinstance(page.worker, started.__class__)
            page.worker = started
            _drain(page)

            done = {it.path.name for it in page.items.values() if it.status == DONE}
            failed = [it for it in page.items.values() if it.status == FAILED]
            assert done == {"pic1.png", "pic2.png", "pic3.png"}, done
            assert [it.path.name for it in failed] == ["deep.png"]
            assert failed[0].error == "stub failure"
            # Each row kept its own generated title.
            assert {it.meta.title for it in page.items.values()
                    if it.status == DONE} == {"Title pic1", "Title pic2", "Title pic3"}

            # Partial batch (deep.png failed) → NO CSV is auto-exported.
            assert (root / "metadata.csv").read_text(encoding="utf-8") == "existing"
            assert not (root / "metadata_2.csv").exists(), \
                "a folder batch with a failure must not auto-export a partial CSV"

            # Retry Failed regenerates ONLY the failed image; now the whole
            # folder succeeded, so the complete CSV is auto-exported.
            calls2: list[str] = []
            def ok_generate(path, settings, api_key, log=None):
                calls2.append(Path(path).name)
                return ImageMetadata("Deep Title", "d", ["k"])
            mw.generate_metadata = ok_generate
            page.retry_failed()
            _drain(page)
            assert calls2 == ["deep.png"], calls2
            assert all(it.status == DONE for it in page.items.values())
            auto = root / "metadata_2.csv"
            assert auto.exists(), "fully-successful folder batch did not auto-export"
            with auto.open(newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.reader(fh))
            assert rows[0] == metadata_export.CSV_HEADERS
            assert len(rows) == 5                          # header + all 4 rows
            assert sorted(r[0] for r in rows[1:]) == [
                "deep.png", "pic1.png", "pic2.png", "pic3.png"]

            # Individually selected files: full batch generates, but the
            # export workflow stays manual (no chosen export folder yet).
            solo = Path(tmp) / "solo"
            solo.mkdir()
            f1, f2 = solo / "one.png", solo / "two.png"
            Image.new("RGB", (8, 8), (1, 1, 1)).save(f1)
            Image.new("RGB", (8, 8), (2, 2, 2)).save(f2)
            page2 = MetadataPage()
            page2._add_paths([f1, f2])                     # same as Add Images…
            assert len(page2.items) == 2
            page2.generate_all()
            _drain(page2)
            assert all(it.status == DONE for it in page2.items.values())
            assert {it.meta.title for it in page2.items.values()} == {"Deep Title"}
            assert list(solo.glob("*.csv")) == []          # no auto-export
            assert not page2._run_roots
            page2.shutdown()
            page.shutdown()
    finally:
        mw.generate_metadata = orig_gen
        credential_store._session_keys.clear()
    print("  page batch + folder auto-CSV OK")


def test_page_auto_retry_full_csv(app: QApplication) -> None:
    """End-to-end page flow: a folder batch with one transient failure
    self-heals through an automatic retry pass and the complete CSV is
    exported WITHOUT the user clicking anything — successful metadata is
    kept and only the failed image is regenerated."""
    from PySide6.QtCore import QEventLoop, QTimer
    from app.ui.metadata_page import MetadataPage, DONE
    import app.engine.metadata_worker as mw

    orig_gen, orig_sleep = mw.generate_metadata, mw.time.sleep
    credential_store._session_keys.clear()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "shots"
            root.mkdir()
            Image.new("RGB", (8, 8), (1, 1, 1)).save(root / "one.png")
            Image.new("RGB", (8, 8), (2, 2, 2)).save(root / "two.png")

            calls: list[str] = []

            def fake_generate(path, settings, api_key, log=None):
                name = Path(path).name
                calls.append(name)
                if name == "two.png" and calls.count("two.png") == 1:
                    raise ProviderError("Gemini rate limit reached (HTTP 429) — "
                                        "too many requests", code=429)
                return ImageMetadata(f"Title {name}", "d", ["k"])

            mw.generate_metadata = fake_generate
            mw.time.sleep = lambda s: None     # no real backoff in tests
            page = MetadataPage()
            page._add_paths([root])
            credential_store.set_session_key(page.provider_combo.currentText(), "test-key")
            page.generate_all()
            loop = QEventLoop()
            page.worker.finished_all.connect(loop.quit)
            QTimer.singleShot(10000, loop.quit)
            loop.exec()
            assert page.worker.wait(5000), "worker did not finish"

            # One automatic retry pass with exactly the failed image (the
            # page's own retry_pass handler logged it before we connected).
            assert "Automatic retry pass 2/3: 1" in page.log.toPlainText(), \
                page.log.toPlainText()
            assert calls.count("one.png") == 1 and calls.count("two.png") == 2, calls
            assert all(it.status == DONE for it in page.items.values())
            titles = {it.path.name: it.meta.title for it in page.items.values()}
            assert titles == {"one.png": "Title one.png",
                              "two.png": "Title two.png"}, titles

            # Full batch green → complete CSV exported, no user interaction.
            csvs = sorted(root.glob("*.csv"))
            assert [c.name for c in csvs] == ["metadata.csv"], csvs
            with csvs[0].open(newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.reader(fh))
            assert rows[0] == metadata_export.CSV_HEADERS
            assert sorted(r[0] for r in rows[1:]) == ["one.png", "two.png"]
            page.shutdown()
    finally:
        mw.generate_metadata, mw.time.sleep = orig_gen, orig_sleep
        credential_store._session_keys.clear()
    print("  page auto-retry + full CSV OK")


def _release_qt(app: QApplication) -> None:
    """Destroy leftover widgets/threads while QApplication is still alive.

    Test pages are created without a parent, so without this their QThread and
    widget objects are finalized only during interpreter shutdown — after
    Qt/Shiboken state is gone, which crashes the process (access violation)
    even though every assertion passed.
    """
    for w in list(app.topLevelWidgets()):
        w.close()
        w.deleteLater()
    for _ in range(8):
        app.processEvents()


def main() -> int:
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "sub").mkdir()
        try:
            test_parse_and_sanitize()
            test_prepare_analysis_bytes(folder)
            test_csv_export(folder)
            test_embed_roundtrip(folder)
            test_credential_store()
            test_retry_classification()
            test_error_classification()
            test_model_catalogues()
            test_gemini_request_payload()
            test_groq_request_payload_and_http_errors()
            test_worker_with_stubbed_provider(app)
            test_worker_auto_retry_passes(app)
            test_worker_cancel_stops_retries(app)
            test_ui_smoke(app)
            test_editor_switch_keeps_metadata_per_item(app)
            test_page_batch_and_auto_csv(app)
            test_page_auto_retry_full_csv(app)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"METADATA TEST FAILED: {exc}")
            _release_qt(app)
            return 1
    _release_qt(app)
    print("METADATA TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
