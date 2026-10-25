"""Background worker for the Metadata Generator screen.

Runs on a ``QThread`` like the upscaler's BatchWorker, but the two pipelines
are completely independent: this one only reads existing image files and
calls a text/vision provider per image. The UI can start it for the whole
queue, for the failed items, or for an explicitly selected subset.

On top of that, folder/file batches now drive *automatic retry passes*:
when the first pass ends with images that failed for temporary reasons
(rate limits, timeouts, network blips, provider 5xx), the successful
metadata is kept and ONLY the retryable images are re-run after an
increasing pause, up to ``MAX_BATCH_PASSES`` passes in total — the user
never has to click "Retry Failed" for those. Permanent problems (bad API
key, unsupported image, retired model, an Ollama server that is not
running) are reported immediately and are never retried. Images still
failing after the last automatic pass stay marked failed so the existing
manual "Retry Failed" action picks them up.

``OllamaLookupWorker`` is the small companion thread behind the Metadata
screen's connection/model check: it asks an Ollama server (local or
ollama.com) which models exist and which of them can analyse images, so the
UI never blocks on HTTP.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .metadata_settings import MetadataSettings, OLLAMA_CLOUD, OLLAMA_LOCAL
from .metadata_providers import check_ollama, generate_metadata, is_retryable_error

# Whole-batch automatic passes: the first run plus two retry passes.
MAX_BATCH_PASSES = 3
# Increasing pause before each retry pass: 5s before pass 2, 10s before
# pass 3 — spreads requests instead of bursting them again.
PASS_BACKOFF_BASE = 5.0
_WAIT_SLICE = 0.1  # sleep granularity so Cancel stays responsive


class MetadataWorker(QThread):
    log_message = Signal(str)                  # human-readable log line
    progress = Signal(int, int)                # (done, total) within the pass
    item_started = Signal(str, int, int)       # (full path, 1-based index, total)
    item_done = Signal(str, dict)              # (full path, metadata dict)
    item_failed = Signal(str, str)             # (full path, final error message)
    retry_pass = Signal(int, int, int, float)  # (pass no, total passes, images, waited s)
    finished_all = Signal(int, int)            # (succeeded, failed)
    cancelled = Signal()

    def __init__(self, files: list[Path], settings: MetadataSettings,
                 api_key: str) -> None:
        super().__init__()
        self.files = [Path(f) for f in files]
        self.settings = settings
        self.api_key = api_key
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def _wait(self, seconds: float) -> bool:
        """Interruptible sleep — returns False as soon as cancel is seen."""
        remaining = seconds
        while remaining > 0:
            if self._cancel:
                return False
            step = min(_WAIT_SLICE, remaining)
            time.sleep(step)
            remaining -= step
        return not self._cancel

    def _abandon(self, retry_pending: list[tuple[Path, Exception]]) -> int:
        """Cancel hit before a queued retry pass: finalise those images."""
        for path, _exc in retry_pending:
            self.item_failed.emit(str(path), "Cancelled before automatic retry.")
            self.log_message.emit(f"✗ {path.name}: cancelled before automatic retry.")
        return len(retry_pending)

    def run(self) -> None:  # noqa: D102 (QThread entry point)
        total = len(self.files)
        succeeded = failed = 0
        self.log_message.emit(
            f"Generating metadata for {total} image(s) with {self.settings.provider} "
            f"({self.settings.model}, {self.settings.language})…")
        if self.settings.provider == OLLAMA_CLOUD:
            self.log_message.emit(
                "☁ Ollama Cloud is a hosted service — each image is uploaded "
                "to https://ollama.com for analysis.")
        elif self.settings.provider == OLLAMA_LOCAL:
            self.log_message.emit(
                f"⛁ Local Ollama at {self.settings.ollama_url}: images are only "
                "read on this computer and never uploaded.")
        items = list(self.files)
        cancelled = False
        for pass_no in range(1, MAX_BATCH_PASSES + 1):
            retry_pending: list[tuple[Path, Exception]] = []
            for idx, path in enumerate(items):
                if self._cancel:
                    cancelled = True
                    break
                # Paths (not names) identify items — two queued folders can hold
                # same-named files and results must land on the right row.
                self.item_started.emit(str(path), idx + 1, len(items))
                try:
                    meta = generate_metadata(
                        path, self.settings, self.api_key,
                        log=self.log_message.emit)
                except Exception as exc:
                    if is_retryable_error(exc) and pass_no < MAX_BATCH_PASSES:
                        # Not final — the automatic retry pass will re-run it.
                        retry_pending.append((path, exc))
                        self.log_message.emit(
                            f"↻ {path.name}: temporary failure ({exc}) — queued "
                            f"for automatic retry pass {pass_no + 1}")
                    elif is_retryable_error(exc):
                        failed += 1
                        self.item_failed.emit(
                            str(path),
                            f"{exc} (gave up after {MAX_BATCH_PASSES} automatic passes)")
                        self.log_message.emit(
                            f"✗ {path.name}: still failing after "
                            f"{MAX_BATCH_PASSES} automatic passes — {exc}")
                    else:
                        # Permanent (bad key, unsupported image, retired model):
                        # retrying can never help, report it right away.
                        failed += 1
                        self.item_failed.emit(str(path), str(exc))
                        self.log_message.emit(f"✗ {path.name}: {exc}")
                else:
                    succeeded += 1
                    self.item_done.emit(str(path), meta.as_dict())
                    self.log_message.emit(f"✓ {path.name}: {meta.title}")
                finally:
                    self.progress.emit(idx + 1, len(items))
            if self._cancel:
                cancelled = True
                failed += self._abandon(retry_pending)
                break
            if not retry_pending:
                break
            delay = PASS_BACKOFF_BASE * (2 ** (pass_no - 1))
            self.log_message.emit(
                f"{len(retry_pending)} image(s) hit temporary failures — keeping "
                f"{succeeded} successful result(s) and retrying them in "
                f"{delay:.0f}s (pass {pass_no + 1}/{MAX_BATCH_PASSES}).")
            if not self._wait(delay):
                cancelled = True
                failed += self._abandon(retry_pending)
                break
            self.retry_pass.emit(pass_no + 1, MAX_BATCH_PASSES,
                                 len(retry_pending), delay)
            items = [p for p, _ in retry_pending]
        if cancelled:
            self.log_message.emit("Cancelled by user.")
            self.cancelled.emit()
        else:
            self.log_message.emit(
                f"Metadata generation complete — {succeeded} ok, {failed} failed.")
        self.finished_all.emit(succeeded, failed)


class OllamaLookupWorker(QThread):
    """Runs the Ollama connection/vision-model check off the UI thread.

    Emits ``result(models, message)`` exactly once — an unreachable server,
    a rejected cloud key or a model-less install all come back as an empty
    list with an explained message, never as an exception in the UI.
    """

    log_message = Signal(str)
    result = Signal(list, str)          # (vision models, outcome message)

    def __init__(self, base_url: str, api_key: str = "", cloud: bool = False) -> None:
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.cloud = cloud
        self._stop = False

    def request_stop(self) -> None:
        """Skip the remaining per-model lookups (a close/cancel request)."""
        self._stop = True

    def stopped(self) -> bool:
        return self._stop

    def run(self) -> None:  # noqa: D102 (QThread entry point)
        try:
            models, message = check_ollama(self.base_url, self.api_key,
                                           cloud=self.cloud,
                                           log=self.log_message.emit,
                                           stopped=self.stopped)
        except Exception as exc:      # missing deps, unexpected SDK faults
            models, message = [], f"Ollama check failed: {exc}"
        self.result.emit(models, message)
