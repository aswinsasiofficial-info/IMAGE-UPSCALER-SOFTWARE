"""The Upscaler's queue and its background batch job.

One :class:`Workspace` lives in the desktop process (a single-user app) and
is shared by every request the embedded Django server handles. A batch runs
on a plain thread so the page stays responsive while it polls for progress.
The Real-ESRGAN model is loaded once per batch and released afterwards, and
cancellation is honoured between tiles and between images.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import imaging
from .engine import Cancelled, RealESRGAN
from .files import (DEFAULT_OUTPUT_DIR, OUTPUT_DIR_NAME, SequentialNamer, discover, extension,
                    find_root, output_dir_for, output_filename, path_key, unique_path)
from .geometry import MAX_SIDE, MODEL_SCALE, SizeOptions, describe, plan_size

LOG_LIMIT = 5000


@dataclass
class QueueItem:
    id: str
    path: Path
    width: int          # upright (EXIF-rotated) size
    height: int
    has_alpha: bool

    def as_json(self) -> dict:
        return {"id": self.id, "name": self.path.name, "path": str(self.path),
                "width": self.width, "height": self.height, "has_alpha": self.has_alpha}


@dataclass(frozen=True)
class JobOptions:
    size: SizeOptions = SizeOptions()
    pad_color: str = "white"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    save_beside: bool = False
    device: str = "auto"
    fp16: bool = True
    tile: int = 256
    output_format: str = "png"
    jpg_quality: int = 95
    suffix: str = "_x4"
    rename_batch: bool = False


def default_engine(options: JobOptions) -> RealESRGAN:
    return RealESRGAN(device=options.device, fp16=options.fp16, tile=options.tile)


EngineFactory = Callable[[JobOptions], RealESRGAN]


class Busy(RuntimeError):
    """A batch is already running."""


class Job:
    def __init__(self, items: list[QueueItem], roots: list[Path], options: JobOptions,
                 engine_factory: EngineFactory) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.items = list(items)
        self.roots = list(roots)
        self.options = options
        self.engine_factory = engine_factory
        self.total = len(self.items)
        self.done = self.ok = self.failed = 0
        self.state = "running"          # running | cancelling | done | cancelled | failed
        self.phase = "starting"         # starting | downloading | loading | processing | finished
        self.current = ""
        self.download: Optional[tuple[int, int]] = None
        self.results: dict[str, dict] = {}
        self._log: list[str] = []
        self._log_base = 0              # sequence number of _log[0] once trimmed
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="upscaler-batch", daemon=True)

    # -- control ----------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        # Check and change together, so a batch finishing at this very moment
        # can't be left stuck in "cancelling".
        with self._lock:
            if self.state == "running":
                self._cancel.set()
                self.state = "cancelling"

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self.state in ("running", "cancelling")

    # -- reporting --------------------------------------------------------------
    def log(self, text: str) -> None:
        with self._lock:
            self._append(text)

    def _append(self, text: str) -> None:
        """Add a log line; caller holds the lock."""
        self._log.append(text)
        if len(self._log) > LOG_LIMIT:
            drop = len(self._log) - LOG_LIMIT
            del self._log[:drop]
            self._log_base += drop

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            start = max(0, since - self._log_base)
            return {
                "id": self.id, "state": self.state, "running": self.running,
                "phase": self.phase, "current": self.current,
                "total": self.total, "done": self.done, "ok": self.ok, "failed": self.failed,
                "download": list(self.download) if self.download else None,
                "log": self._log[start:], "log_next": self._log_base + len(self._log),
                "results": dict(self.results),
            }

    def _set(self, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self, key, value)

    def _on_download(self, done: int, total: int) -> None:
        self._set(phase="downloading", download=(done, total))

    # -- the batch --------------------------------------------------------------
    def _run(self) -> None:
        opts = self.options
        engine = None
        try:
            # Fail early on an unusable output folder — but only create it if an
            # image will actually be saved there (folder-mode images aren't).
            if not opts.save_beside and any(find_root(i.path, self.roots) is None
                                            for i in self.items):
                opts.output_dir.mkdir(parents=True, exist_ok=True)
            if self.roots:
                self.log(f"Folder mode: images from these folders save inside their "
                         f"'{OUTPUT_DIR_NAME}' subfolder (structure preserved): "
                         + ", ".join(str(r) for r in self.roots))
            engine = self.engine_factory(opts)
            self._set(phase="loading")
            self.log("Loading model…")
            engine.load(log=self.log, progress=self._on_download)
        except Exception as exc:
            self.log(f"✗ Setup failed: {exc}")
            if engine is not None:
                engine.close()
            self._set(failed=self.total, phase="finished", state="failed")
            return

        namer = SequentialNamer() if opts.rename_batch else None
        ext = extension(opts.output_format)
        if namer is not None:
            self.log(f"Batch rename on: outputs will be named {namer.prefix}1{ext}, "
                     f"{namer.prefix}2{ext}, … (numbering restarts at 1 per folder).")
        try:
            for index, item in enumerate(self.items):
                if self._cancel.is_set():
                    self.log("Cancelled by user.")
                    break
                with self._lock:
                    self.phase, self.current = "processing", item.path.name
                    self.results[item.id] = {"status": "processing"}
                folder = None
                try:
                    folder = output_dir_for(item.path, self.roots, opts.save_beside, opts.output_dir)
                    out_path = (namer.next_path(folder, ext) if namer is not None else
                                unique_path(folder / output_filename(item.path, opts.suffix,
                                                                     opts.output_format)))
                    size = self._process(engine, item, out_path)
                    with self._lock:
                        self.ok += 1
                        self.results[item.id] = {"status": "done", "output": str(out_path),
                                                 "size": list(size)}
                except Cancelled:
                    if namer is not None and folder is not None:
                        namer.release(folder)
                    with self._lock:
                        self.results[item.id] = {"status": "cancelled"}
                    self.log("Cancelled by user.")
                    break
                except Exception as exc:
                    if namer is not None and folder is not None:
                        namer.release(folder)
                    with self._lock:
                        self.failed += 1
                        self.results[item.id] = {"status": "failed", "error": str(exc)}
                    self.log(f"✗ {item.path.name}: {exc}")
                finally:
                    self._set(done=index + 1)
        finally:
            engine.close()

        # Decide, log and publish the end state in one step: a late Cancel can't
        # slip in between, and the page sees the last log line before "done".
        with self._lock:
            cancelled = self._cancel.is_set()
            if not cancelled:
                self._append(f"Batch complete — {self.ok} ok, {self.failed} failed.")
            self.phase, self.current = "finished", ""
            self.state = "cancelled" if cancelled else "done"

    def _process(self, engine: RealESRGAN, item: QueueItem, out_path: Path) -> tuple[int, int]:
        opts = self.options
        src = imaging.load(item.path)
        sw, sh = src.size
        # Re-plan from the pixels actually loaded, in case the file changed since queueing.
        plan = plan_size(sw, sh, opts.size)
        if not plan.resolved:
            raise ValueError(f"{sw}x{sh} has a different shape from the "
                             f"{plan.output[0]}x{plan.output[1]} frame — choose Pad or Crop first")
        if max(plan.output) > MAX_SIDE:
            raise ValueError(f"output {plan.output[0]}x{plan.output[1]} exceeds "
                             f"{MAX_SIDE:,} px per side")

        ai = engine.upscale(src.rgb, cancelled=self._cancel.is_set, log=self.log)
        ai_size = (ai.shape[1], ai.shape[0])
        if ai_size != (sw * MODEL_SCALE, sh * MODEL_SCALE):
            raise RuntimeError(f"model returned {ai_size[0]}x{ai_size[1]}, expected exactly 4x")
        img = imaging.compose(ai, src.alpha, plan, opts.pad_color)
        imaging.save(img, out_path, opts.output_format, opts.jpg_quality, src.icc_profile)

        saved = imaging.saved_size(out_path)
        if saved != plan.output:
            raise RuntimeError(f"saved {saved[0]}x{saved[1]} but planned "
                               f"{plan.output[0]}x{plan.output[1]}")
        steps = f"{sw}x{sh} -> "
        if ai_size != saved:
            steps += f"AI {ai_size[0]}x{ai_size[1]} -> "
        self.log(f"[OK] {item.path.name}: {steps}{saved[0]}x{saved[1]} "
                 f"({describe(plan, opts.size)}) -> {out_path}")
        return saved


@dataclass
class AddResult:
    added: list[QueueItem] = field(default_factory=list)
    folders: list[tuple[Path, int]] = field(default_factory=list)   # (folder, images found)
    empty_folders: list[Path] = field(default_factory=list)
    unreadable: list[tuple[Path, str]] = field(default_factory=list)
    missing: list[Path] = field(default_factory=list)
    duplicates: int = 0

    def as_json(self) -> dict:
        return {
            "added": len(self.added),
            "folders": [{"path": str(p), "count": n} for p, n in self.folders],
            "empty_folders": [str(p) for p in self.empty_folders],
            "unreadable": [{"path": str(p), "error": e} for p, e in self.unreadable],
            "missing": [str(p) for p in self.missing],
            "duplicates": self.duplicates,
            "output_dir_name": OUTPUT_DIR_NAME,
        }


class Workspace:
    def __init__(self, engine_factory: EngineFactory = default_engine) -> None:
        self.engine_factory = engine_factory
        self.items: list[QueueItem] = []
        self.roots: list[Path] = []      # folders added with Add Folder / dropped
        self.job: Optional[Job] = None
        self._lock = threading.RLock()

    # -- queue ------------------------------------------------------------------
    def add(self, paths: Iterable[str | Path]) -> AddResult:
        result = AddResult()
        # Scan and probe without the lock (disk I/O), then merge under it.
        new_roots: list[Path] = []
        probed: list[tuple[str, Path, imaging.ImageInfo]] = []
        seen: set[str] = set()
        for raw in paths:
            text = str(raw).strip().strip('"')
            if not text:
                continue
            p = Path(text)
            if not p.exists():
                result.missing.append(p)
                continue
            found = discover([p])
            if p.is_dir():
                if not found:
                    result.empty_folders.append(p)
                    continue
                result.folders.append((p, len(found)))
                new_roots.append(p)
            for f in found:
                key = path_key(f)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    probed.append((key, f, imaging.probe(f)))
                except Exception as exc:
                    result.unreadable.append((f, str(exc)))

        with self._lock:
            root_keys = {path_key(r) for r in self.roots}
            for root in new_roots:
                if path_key(root) not in root_keys:
                    self.roots.append(root)
                    root_keys.add(path_key(root))
            known = {path_key(i.path) for i in self.items}
            for key, f, info in probed:
                if key in known:
                    result.duplicates += 1
                    continue
                item = QueueItem(uuid.uuid4().hex[:12], f, info.width, info.height, info.has_alpha)
                self.items.append(item)
                result.added.append(item)
                known.add(key)
        return result

    def remove(self, ids: Iterable[str]) -> None:
        drop = set(ids)
        with self._lock:
            self.items = [i for i in self.items if i.id not in drop]
            # Forget folder roots that no queued image lives under any more.
            self.roots = [r for r in self.roots
                          if any(find_root(i.path, [r]) for i in self.items)]

    def clear(self) -> None:
        with self._lock:
            self.items.clear()
            self.roots.clear()

    def snapshot_items(self) -> list[QueueItem]:
        with self._lock:
            return list(self.items)

    def root_for(self, item: QueueItem) -> Optional[Path]:
        with self._lock:
            match = find_root(item.path, self.roots)
        return match[0] if match else None

    # -- batch ------------------------------------------------------------------
    def start(self, options: JobOptions) -> Job:
        with self._lock:
            if self.job is not None and self.job.running:
                raise Busy("A batch is already running.")
            if not self.items:
                raise ValueError("Add some images first.")
            self.job = Job(self.items, self.roots, options, self.engine_factory)
            self.job.start()
            return self.job

    def cancel(self) -> None:
        with self._lock:
            if self.job is not None:
                self.job.cancel()

    def shutdown(self, timeout: float = 3.0) -> None:
        """Ask a running batch to stop and give it a moment to finish its file."""
        job = self.job
        if job is not None and job.running:
            job.cancel()
            job.join(timeout)


_workspace = Workspace()


def get_workspace() -> Workspace:
    return _workspace


def reset_workspace(engine_factory: EngineFactory = default_engine) -> Workspace:
    """Fresh queue (used by tests)."""
    global _workspace
    _workspace.shutdown()
    _workspace = Workspace(engine_factory)
    return _workspace
