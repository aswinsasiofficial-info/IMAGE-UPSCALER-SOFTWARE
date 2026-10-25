"""Page and JSON API of the Upscaler.

The page is a thin client: the queue, the size plan and the batch job all
live server-side, so the dimensions previewed before processing come from the
very code that later writes the files.
"""

from __future__ import annotations

import json
import re

from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from . import host as desktop
from .engine import WEIGHTS_FILE, probe_device, recommended_tile, weights_ready
from .files import DEFAULT_OUTPUT_DIR, OUTPUT_DIR_NAME
from .forms import UpscaleForm
from .geometry import (CROP, DEFAULT_FACTOR, LARGE_OUTPUT_MP, MAX_FACTOR, MAX_SIDE, MIN_FACTOR,
                       MODEL_SCALE, PAD, SCALE, Plan, SizeOptions, describe, orientation,
                       plan_size, ratio_label)
from .workspace import AddResult, Busy, QueueItem, get_workspace

_THEME_VALUE = {
    "color": re.compile(r"^#[0-9a-fA-F]{6}$"),
    "font-family": re.compile(r"^[\w .\-]{1,64}$"),
    "font-size": re.compile(r"^\d{1,2}(\.\d{1,2})?pt$"),
}


# -- helpers ------------------------------------------------------------------
def _body(request) -> dict:
    try:
        data = json.loads(request.body or b"{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _since(request) -> int:
    """Log position the page already has (``?since=``)."""
    try:
        return max(0, int(request.GET.get("since", 0)))
    except (TypeError, ValueError):
        return 0


def _error(status: int, message: str, **extra) -> JsonResponse:
    return JsonResponse({"error": message, **extra}, status=status)


def _theme_css(theme: dict[str, str]) -> str:
    """CSS custom properties from the host palette (validated — it lands in a <style>)."""
    rules = []
    for name, value in theme.items():
        kind = name if name in _THEME_VALUE else "color"
        if not re.fullmatch(r"[a-z\-]{1,32}", name) or not _THEME_VALUE[kind].match(value):
            continue
        rules.append(f'--{name}:"{value}";' if name == "font-family" else f"--{name}:{value};")
    return "".join(rules)


def _queue_json(items: list[QueueItem]) -> list[dict]:
    ws = get_workspace()
    rows = []
    for item in items:
        row = item.as_json()
        root = ws.root_for(item)
        row.update(ratio=ratio_label(item.width, item.height),
                   orientation=orientation(item.width, item.height),
                   folder_root=str(root) if root else None)
        rows.append(row)
    return rows


def _geometry(plan: Plan) -> dict:
    return {"frame": list(plan.output), "content": list(plan.content), "offset": list(plan.offset)}


def _kind(plan: Plan) -> str:
    if plan.mode == SCALE:
        return "scale"
    if not plan.conflict:
        return "exact"
    return plan.fit or "conflict"


def _summary(opts: SizeOptions | None, total: int, conflicts: int) -> str:
    if not total:
        return ""
    if opts is None:
        return "Enter a valid output size to see each image's result."
    if opts.mode == SCALE:
        return (f"Width and height are both multiplied by {opts.factor:g}, so every image "
                f"keeps its own shape and its whole picture.")
    frame = f"{opts.width} × {opts.height}"
    if not conflicts:
        return f"Every image already has the shape of {frame} — nothing is padded or cropped."
    if opts.fit == PAD:
        return (f"{conflicts} of {total} image(s) get borders to fill {frame}; "
                f"their whole picture is kept.")
    if opts.fit == CROP:
        return (f"{conflicts} of {total} image(s) are trimmed to fill {frame} — "
                f"part of their picture is cut off.")
    return (f"{conflicts} of {total} image(s) have a different shape from {frame}. "
            f"Choose how to handle them — nothing is cropped or stretched without your say-so.")


def _plan_rows(items: list[QueueItem], opts: SizeOptions | None) -> tuple[list[dict], int]:
    rows, conflicts = _queue_json(items), 0
    for row, item in zip(rows, items):
        if opts is None:
            row.update(output=None, kind="invalid", note="Enter a valid output size")
            continue
        plan = plan_size(item.width, item.height, opts)
        conflicts += plan.conflict
        warning = ""
        # Peak pixels held while processing: the network's 4x result, the resized
        # picture (bigger than the frame when cropping), or the output itself.
        peak_mp = max(plan.megapixels,
                      item.width * item.height * MODEL_SCALE ** 2 / 1_000_000,
                      plan.content[0] * plan.content[1] / 1_000_000)
        if max(plan.output) > MAX_SIDE:
            warning = f"Too large — the limit is {MAX_SIDE:,} px per side"
        elif peak_mp > LARGE_OUTPUT_MP:
            warning = f"Very large ({peak_mp:,.0f} MP while processing) — needs several GB of RAM"
        row.update(output=list(plan.output), kind=_kind(plan), note=describe(plan, opts),
                   warning=warning, geometry=_geometry(plan))
        if plan.conflict:
            row["alternatives"] = {}
            for fit in (PAD, CROP):
                alt_opts = SizeOptions(opts.mode, opts.factor, opts.width, opts.height, fit)
                alt = plan_size(item.width, item.height, alt_opts)
                row["alternatives"][fit] = {"geometry": _geometry(alt),
                                            "note": describe(alt, alt_opts)}
    return rows, conflicts


def _settings_form(data: dict, items: list[QueueItem]) -> UpscaleForm:
    settings = data.get("settings")
    return UpscaleForm(settings if isinstance(settings, dict) else {},
                       images=[(i.width, i.height) for i in items])


def _errors(form: UpscaleForm) -> dict[str, list[str]]:
    return {field: [str(m) for m in messages] for field, messages in form.errors.items()}


def _with_queue(result: AddResult | None = None, **extra) -> JsonResponse:
    payload = {"queue": _queue_json(get_workspace().snapshot_items()), **extra}
    if result is not None:
        payload["added"] = result.as_json()
    return JsonResponse(payload)


def _no_host() -> JsonResponse:
    return _error(501, "Native dialogs are only available inside the desktop app.")


# -- page ---------------------------------------------------------------------
@require_GET
@ensure_csrf_cookie
def index(request):
    host = desktop.current()
    theme = host.theme() if host else {}
    device = probe_device()
    config = {
        "api": reverse("upscaler:index") + "api/",
        "has_dialogs": host is not None,
        "model_ready": weights_ready(),
        "model_file": WEIGHTS_FILE,
        "default_output_dir": str(DEFAULT_OUTPUT_DIR),
        "output_dir_name": OUTPUT_DIR_NAME,
        "limits": {"min_factor": MIN_FACTOR, "max_factor": MAX_FACTOR, "max_side": MAX_SIDE},
        "device": device,
        "defaults": {
            "size_mode": SCALE, "scale_factor": str(DEFAULT_FACTOR),
            "exact_width": "", "exact_height": "", "fit": "", "pad_color": "white",
            "output_dir": "", "save_beside": False,
            "device": "cuda" if device["cuda"] else "auto",
            "fp16": device["cuda"],
            "tile": recommended_tile(device["vram_gb"]) if device["cuda"] else 256,
            "output_format": "png", "jpg_quality": 95, "suffix": f"_x{DEFAULT_FACTOR}",
            "rename_batch": False,
        },
    }
    return render(request, "upscaler/index.html", {
        "config": config,
        "theme_css": _theme_css(theme),
        "scheme": theme.get("scheme", "") if theme else "",
    })


# -- queue --------------------------------------------------------------------
@require_GET
def state(request):
    ws = get_workspace()
    job = ws.job.snapshot(_since(request)) if ws.job else None
    return JsonResponse({"queue": _queue_json(ws.snapshot_items()), "job": job})


@require_POST
def queue_add(request):
    paths = _body(request).get("paths")
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        return _error(400, "Expected a list of file or folder paths.")
    return _with_queue(get_workspace().add(paths))


@require_POST
def queue_remove(request):
    ids = _body(request).get("ids")
    if not isinstance(ids, list):
        return _error(400, "Expected a list of queue ids.")
    get_workspace().remove(str(i) for i in ids)
    return _with_queue()


@require_POST
def queue_clear(request):
    get_workspace().clear()
    return _with_queue()


# -- planning -----------------------------------------------------------------
@require_POST
def plan(request):
    items = get_workspace().snapshot_items()
    form = _settings_form(_body(request), items)
    valid = form.is_valid()
    opts = form.size_options()
    rows, conflicts = _plan_rows(items, opts)
    return JsonResponse({
        "valid": valid, "errors": _errors(form), "items": rows,
        "conflicts": conflicts, "summary": _summary(opts, len(items), conflicts),
    })


# -- native dialogs (desktop host) ----------------------------------------------
@require_POST
def dialog_images(request):
    host = desktop.current()
    if host is None:
        return _no_host()
    paths = host.pick_images()
    if not paths:
        return _with_queue(cancelled=True)
    return _with_queue(get_workspace().add(paths))


@require_POST
def dialog_folder(request):
    host = desktop.current()
    if host is None:
        return _no_host()
    folder = host.pick_folder()
    if not folder:
        return _with_queue(cancelled=True)
    return _with_queue(get_workspace().add([folder]))


@require_POST
def dialog_output(request):
    host = desktop.current()
    if host is None:
        return _no_host()
    current = str(_body(request).get("current") or "")
    return JsonResponse({"path": host.pick_output_folder(current) or ""})


@require_POST
def dialog_message(request):
    host = desktop.current()
    if host is None:
        return _no_host()
    data = _body(request)
    kind = "warning" if data.get("kind") == "warning" else "info"
    host.show_message(kind, str(data.get("title") or "Upscaler")[:120],
                      str(data.get("text") or "")[:2000])
    return JsonResponse({})


# -- batch --------------------------------------------------------------------
@require_POST
def job_start(request):
    ws = get_workspace()
    items = ws.snapshot_items()
    if not items:
        return _error(400, "Add some images first.", title="Nothing to do")
    form = _settings_form(_body(request), items)
    if not form.is_valid():
        errors = _errors(form)
        field, messages = next(iter(errors.items()))
        label = form.fields[field].label if field in form.fields else None
        return _error(400, f"{label}: {messages[0]}" if label else messages[0],
                      title="Check the settings", errors=errors)
    try:
        job = ws.start(form.job_options())
    except Busy as exc:
        return _error(409, str(exc), title="Already running")
    except ValueError as exc:       # queue emptied between the check above and now
        return _error(400, str(exc), title="Nothing to do")
    return JsonResponse({"job": job.snapshot()})


@require_GET
def job_status(request):
    job = get_workspace().job
    return JsonResponse({"job": job.snapshot(_since(request)) if job else None})


@require_POST
def job_cancel(request):
    get_workspace().cancel()
    job = get_workspace().job
    return JsonResponse({"job": job.snapshot(10 ** 9) if job else None})
