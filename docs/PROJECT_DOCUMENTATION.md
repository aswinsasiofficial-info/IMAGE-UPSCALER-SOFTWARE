# Batch Image Upscaler — Comprehensive Project Documentation

**Version:** 1.3 · **Model:** Real-ESRGAN `x4plus` · **Stack:** Python 3.10+, PyTorch (CUDA 12.6), PySide6 (Qt 6)

A fully local, GPU-accelerated Windows desktop application that batch-upscales images **4×**
using the **Real-ESRGAN x4plus** super-resolution model. It is tuned to run on an
**NVIDIA RTX 3050 (6 GB VRAM)** with 16 GB system RAM, processes 20–30+ images per batch,
and requires **no internet connection after the one-time model-weight download**.
Every saved output is **exactly 8000 × 5000 px by default** (configurable), reached by a
crop-to-fill resize that never stretches the image.

> **Note on resampling:** the neural network's internal upsampling uses
> **nearest-neighbor** (feature-space) and the alpha-channel resize uses
> **bicubic** — neither uses Lanczos. A **Lanczos (LANCZOS)** filter is used in
> exactly one place: the final **crop-to-fill resize** that makes the finished 4×
> image *exactly* cover the configured target dimensions (default **8000 × 5000**)
> before a centered crop trims the excess.
> See [Section 6 — Resampling Logic](#6-resampling-in-this-project) for details
> and rationale.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Layout](#2-project-layout)
3. [Python Module Reference](#3-python-module-reference)
4. [Dependencies](#4-dependencies)
5. [Real-ESRGAN Model Integration](#5-real-esrgan-model-integration)
6. [Resampling Logic](#6-resampling-in-this-project)
7. [Tiled Inference & VRAM Management](#7-tiled-inference--vram-management)
8. [User Interface](#8-user-interface)
9. [Local Setup — Step by Step](#9-local-setup--step-by-step)
10. [Usage — Step by Step](#10-usage--step-by-step)
11. [Testing & Verification](#11-testing--verification)
12. [Performance Tuning](#12-performance-tuning)
13. [Troubleshooting](#13-troubleshooting)
14. [License & Credits](#14-license--credits)

---

## 1. Architecture Overview

The application follows a clean **three-layer separation**: UI, orchestration, and inference.

```
┌────────────────────────────────────────────────────────────────┐
│  Presentation Layer               app/ui/main_window.py        │
│  PySide6 QMainWindow: queue list, settings panel, progress,    │
│  drag & drop, log view, status bar (CUDA probe)                │
└──────────────────────────┬─────────────────────────────────────┘
                           │ starts / signals / slots
┌──────────────────────────▼─────────────────────────────────────┐
│  Orchestration Layer              app/engine/worker.py         │
│  BatchWorker (QThread): loads model ONCE, iterates the queue,  │
│  emits progress/log/cancel signals per image                   │
└──────────────────────────┬─────────────────────────────────────┘
                           │ calls upscale_image() per file
┌──────────────────────────▼─────────────────────────────────────┐
│  Inference Layer       app/engine/realesrgan_upscaler.py       │
│  Weight download/cache · image I/O (EXIF, RGBA) · tiled        │
│  fp16 inference · output encoding (PNG/JPEG)                   │
│                          │ runs forward passes on               │
│                  app/engine/rrdbnet.py — vendored RRDBNet      │
│                  network architecture (pure PyTorch)           │
└────────────────────────────────────────────────────────────────┘
```

Key design decisions:

| Concern | Approach | Why |
|---|---|---|
| Model code | Vendored `RRDBNet` re-implementation | Avoids `basicsr`/`realesrgan` packages that break on modern torchvision |
| Batching | One model load per batch, reused for every image | Eliminates per-image load overhead — the core batching speedup |
| UI responsiveness | Heavy work on a `QThread`, never the main thread | Qt's GUI thread must not block; communication is via signals |
| Memory | Overlapping tile-based inference | Bounds peak VRAM for arbitrarily large images on a 6 GB GPU |
| Precision | fp16 (half) on CUDA, fp32 on CPU | ~2× speed and ~half the VRAM on RTX 30xx cards |
| Privacy | 100% local processing | No uploads, no accounts; internet only for the one-time weight download |

Data flow for one image: file path → EXIF-aware PIL load → RGB float32 NumPy array in `[0,1]`
(+ optional detached alpha channel) → `(1,3,H,W)` tensor → device/fp16 → RRDBNet forward pass
(whole or tiled) → clamp to `[0,1]` → uint8 → alpha re-merge → Lanczos crop-to-fill resize to
the exact `target_width`×`target_height` (default 8000×5000) → centered crop → PNG/JPEG save.

---

## 2. Project Layout

```
d:\Upscaler\
├── app/
│   ├── main.py                      # GUI entry point (python -m app.main)
│   ├── engine/
│   │   ├── rrdbnet.py               # Real-ESRGAN x4plus network architecture
│   │   ├── realesrgan_upscaler.py   # weights, tiled inference, image I/O
│   │   └── worker.py                # QThread batch runner (progress + cancel)
│   └── ui/
│       └── main_window.py           # PySide6 desktop UI
├── models/
│   └── RealESRGAN_x4plus.pth        # pretrained weights (~67 MB, auto-downloaded)
├── tests/
│   ├── smoke_test.py                # headless end-to-end engine check
│   ├── worker_batch_test.py         # BatchWorker path test (offscreen Qt)
│   ├── target_size_test.py          # exact-target crop-to-fill + transparency test
│   └── folder_mode_test.py          # folder-mode scan/output-path test (no model)
├── setup.ps1                        # one-time environment installer
├── run.bat                          # double-click launcher
├── run.ps1                          # PowerShell launcher
├── requirements.txt                 # pinned deps incl. CUDA index URL
└── README.md                        # quick-start guide
```

---

## 3. Python Module Reference

### 3.1 `app/main.py` — Entry Point

`main()` performs one critical ordering constraint: **`torch` is imported before
`QApplication`**. On some Windows CUDA setups, importing torch after Qt triggers a
DLL-initialisation race; importing torch first avoids it. If the torch import fails,
the app prints remediation instructions and exits with code `2` instead of crashing
inside the GUI. Otherwise it constructs a `QApplication` (style: `Fusion`), shows
`MainWindow`, and enters the Qt event loop.

Run via: `python -m app.main` (module form is required so the `app.*` package
imports resolve).

### 3.2 `app/engine/rrdbnet.py` — Network Architecture

A faithful, self-contained re-implementation of `basicsr/archs/rrdbnet_arch.py`.
State-dict keys line up **1:1** with the official checkpoint, so `load_state_dict`
works with zero key remapping. Classes:

- **`ResidualDenseBlock`** — 5 conv layers with dense channel connections
  (`num_feat=64`, `num_grow_ch=32`), LeakyReLU slope 0.2, and the empirical
  `x5 * 0.2 + x` residual scaling used during training.
- **`RRDB`** — "Residual in Residual Dense Block": three RDBs chained, plus a
  `out * 0.2 + x` shortcut.
- **`RRDBNet`** — the full x4plus generator:
  `conv_first → 23× RRDB body → conv_body (global residual) → two 2× nearest-neighbour
  upsample stages (conv_up1, conv_up2) → conv_hr → conv_last`, giving **4× total**.
  The constructor supports `scale=1/2/4` via `pixel_unshuffle` input packing, but this
  project only uses `scale=4`.
- **`_make_layer`** — vendored equivalent of basicsr's `make_layer` helper.

### 3.3 `app/engine/realesrgan_upscaler.py` — Inference Engine

Constants: `MODEL_URL` (official GitHub release), `MODEL_FILENAME`,
`MIN_VALID_BYTES = 40 MB` (guards against truncated downloads), `DEFAULT_MODEL_DIR`
(`<project>/models`, derived from the file's location via `Path(__file__).resolve().parents[2]`).

- **`UpscaleSettings`** (dataclass) — all tunable knobs surfaced by the GUI: `device`,
  `fp16`, `tile`, `tile_pad`, `output_format`, `jpg_quality`, `filename_suffix`,
  `keep_original_name`, `output_in_source_dir`, `target_width` (default 8000) and
  `target_height` (default 5000) — the **exact** dimensions every saved image gets
  via crop-to-fill (`0` on either axis disables the operation and keeps the raw 4×
  size). Includes
  `recommended_tile_for_vram()` — a stepped heuristic mapping VRAM tiers
  (≥12 GB → 512, ≥8 → 384, ≥6 → 256, ≥4 → 192, else 128) to safe tile sizes.
- **`ensure_weights()`** — idempotent weight acquisition: returns the cached file if
  valid, deletes and re-downloads if truncated, otherwise streams a 256 KB-chunked
  download to a `.part` temp file (atomic rename on success) with an optional
  `progress_cb(downloaded, total)` callback. Network failures raise
  `WeightDownloadError` and clean up the partial file.
- **`RealESRGANUpscaler`** — the engine class:
  - `load()` — resolves weights, picks the device (`resolve_device()` falls back
    CUDA → CPU transparently), builds `RRDBNet(3,3,64,23,32,scale=4)`, loads the
    state dict from CPU (`params_ema` preferred over `params`; strips any
    DataParallel `module.` prefix), enforces `strict=True` loading, then
    `eval()` + `.to(device)` + optional `.half()`.
  - `_read_image()` — EXIF-orientation-aware load; extracts and detaches the alpha
    channel from RGBA/LA/PA/palette-with-transparency images; returns RGB as
    float32 NumPy in `[0,1]` plus the raw alpha `Image`.
  - `_merge_alpha()` — resizes the alpha channel to output dimensions with
    **bicubic** resampling and re-stacks it onto the upscaled RGB (see Section 6).
  - `compute_fill_dimensions()` / `_fill_to_target()` — the **final crop-to-fill
    resize**: computes the smallest cover size `max(tw/w, th/h)` (aspect preserved,
    enlargement allowed), resizes with **Lanczos**, then takes a centered crop of
    exactly `target_width`×`target_height` (see Section 6.3). Returns
    `(array, was_cropped)` for logging.
  - `_tile_inference()` — overlapping-tile inference (see Section 7).
  - `_infer_tensor()` — wrapped in `@torch.inference_mode`; bypasses tiling when
    `tile == 0` or the image already fits within one tile.
  - `upscale_image()` — the full per-file pipeline; clamps output to `[0,1]`,
    converts to uint8, merges alpha, crop-to-fills to the exact target size,
    saves, and logs the **exact saved dimensions**, calling out when a centered
    crop occurred (e.g. `2000x1250 -> 8000x5000 -> center-cropped to exact target 8000x5000`).
  - `cleanup()` — releases the model and calls `torch.cuda.empty_cache()`.

### 3.4 `app/engine/worker.py` — Batch Orchestration

- `IMAGE_EXTENSIONS` = `.jpg .jpeg .png .bmp .tif .tiff .webp`.
- `UPSCALED_DIR_NAME` = `"upscaled"` — the per-folder output subfolder used by
  **folder mode**; folder scans always skip it so generated results are never
  re-processed on a later run.
- **`collect_images()`** — expands a mix of files and folders (recursive `rglob`)
  into a de-duplicated, order-preserving image list, filtering by extension and
  excluding anything inside the scanned folder's own `upscaled` subfolder
  (case-insensitive top-level match via `_under_output_dir()`).
- **`resolve_output_path()`** — builds `"{stem}{suffix}{ext}"` in the output dir.
- **`BatchWorker(QThread)`** — signals: `log_message`, `progress(done,total)`,
  `image_started`, `image_done`, `image_failed`, `download_progress`,
  `finished_all(ok,failed)`, `cancelled`. Constructor additionally accepts
  `folder_roots` (the folders the user picked with *Add Folder…*, or dropped
  onto the window). Its `run()` method:
  1. Creates the output dir, constructs the upscaler, **loads the model once**.
  2. Loops over the queue, checking the cooperative `_cancel` flag **between images**.
  3. Per image: resolves the output directory via `output_base_dir()`, then the
     output path; avoids collisions (`_avoid_collision()` appends `(1)`, `(2)`, … —
     never silently overwrites); upscales, emits signals.
  4. On any per-image exception: counts it as failed, logs, and **continues the batch**.
  5. Calls `cleanup()` and emits `finished_all`.

  **`output_base_dir()`** implements the folder-mode precedence: if the source
  lives under a remembered folder root (deepest matching root wins for nested
  selections), the result is written to `<root>/upscaled/<same relative
  subfolder>` — e.g. `photos/trip/image.jpg` → `<root>/upscaled/photos/trip/image_x4.jpg`
  — so identically named files from different subfolders never collide. Images
  that did not come from a folder selection keep the classic behaviour (beside
  the original when `output_in_source_dir` is set, otherwise the global output
  folder). Output directories are auto-created by the engine's `_save()`.

  A setup failure (e.g. weights unreachable) aborts the whole batch early with
  `finished_all(0, total)`.

### 3.5 `app/ui/main_window.py` — Desktop GUI

See Section 8. Also exposes `cuda_info()`, a best-effort torch probe returning
`{available, name, vram_gb, torch}` used to preconfigure the UI (default device,
fp16 checkbox, recommended tile size) and populate the status bar.

---

## 4. Dependencies

### 4.1 Pinned dependencies (`requirements.txt`)

| Package | Version | Role | Notes |
|---|---|---|---|
| `torch` | `2.14.1+cu126` | Deep-learning runtime (CUDA 12.6 build) | Installed from PyTorch's own index via `--extra-index-url https://download.pytorch.org/whl/cu126` |
| `torchvision` | `0.29.1+cu126` | Paired with torch | Present for completeness; the vendored architecture avoids torchvision ops that break `realesrgan` |
| `PySide6` | `>=6.8` | Qt 6 GUI toolkit | `QMainWindow`, `QThread`, signals/slots, widgets |
| `Pillow` | `>=10.0` | Image decode/encode, EXIF transpose, alpha handling, resampling | |
| `numpy` | `>=1.26` | Array conversion between PIL and tensors | |

The CUDA wheel set is a **~2.6 GB download** — expect several GB of free disk (~4 GB total).
Because of the `+cu126` local version specifiers, plain `pip install -r requirements.txt`
works in one shot only on machines that can reach the PyTorch CUDA index.

### 4.2 What is deliberately *not* used

| Package | Why excluded |
|---|---|
| `basicsr` | Legacy; breaks on modern torchvision (imports removed APIs) |
| `realesrgan` | Imports `torchvision.transforms.functional_tensor`, **removed** in current torchvision; pins ancient deps |
| `opencv-python` | Pillow + NumPy cover all image I/O needs |

Instead of the official inference packages, the ~120-line network definition is vendored
in `app/engine/rrdbnet.py`, which produces **identical output** with the official weights.

### 4.3 System requirements

- Windows with an **NVIDIA GPU** and a **CUDA 12.x driver** (RTX 3050 6 GB is the target).
- **Python 3.10+** (developed and tested on 3.14).
- Internet on first run only (PyTorch install + one-time 67 MB weight download).

---

## 5. Real-ESRGAN Model Integration

### 5.1 Model identity

- **Checkpoint:** `RealESRGAN_x4plus.pth` (~67 MB) from the official
  [xinntao/Real-ESRGAN v0.1.0 GitHub release].
- **Architecture:** `RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)`
  — 23 RRDB blocks, two 2× upsample stages, 4× output scale.
- **License:** BSD-3-Clause (see Section 14).

### 5.2 Weight acquisition lifecycle

```
ensure_weights(models/)
  ├─ file exists and ≥ 40 MB  → reuse cache (zero network use)
  ├─ file exists and < 40 MB  → delete truncated file → re-download
  └─ file missing             → stream download to RealESRGAN_x4plus.part
                                 ├─ chunked 256 KB reads + progress_cb → GUI bar
                                 ├─ failure → delete .part, raise WeightDownloadError
                                 └─ success → atomic rename .part → .pth
```

The 40 MB `MIN_VALID_BYTES` threshold detects truncated/interrupted downloads that
would otherwise fail later with a confusing `load_state_dict` error.

### 5.3 Loading procedure (`RealESRGANUpscaler.load`)

1. Obtain weights path via `ensure_weights()` (progress forwarded to the UI).
2. Resolve device — `"cpu"`, `"cuda"`, or `"auto"`; CUDA requested but unavailable
   silently falls back to CPU with a log warning.
3. `torch.load(weights, map_location="cpu")` — always load onto CPU first for
   portability, then move.
4. Key selection: the official checkpoint ships **both** `params` and `params_ema`;
   the **EMA weights (`params_ema`) are preferred** as they are the smoothed
   generator used for inference.
5. Defensive strip of any `module.` prefix (checkpoints saved under `DataParallel`).
6. `model.load_state_dict(state, strict=True)` — strictness guarantees the vendored
   architecture is bit-compatible with the checkpoint; any mismatch fails loudly.
7. `model.eval()` → `.to(device)` → `.half()` if fp16 is enabled **and** on CUDA
   (fp16 is silently skipped on CPU).

The model is loaded **exactly once per batch** by `BatchWorker` and reused for every
image — verified by `tests/worker_batch_test.py`, which asserts the
`"Model loaded and ready."` log appears exactly one time for a 6-image batch.

### 5.4 Inference pipeline (per image)

1. PIL open → `ImageOps.exif_transpose` (camera rotation applied).
2. If alpha is present (RGBA/LA/PA or palette-with-transparency): split into an
   RGB array + a separate alpha `Image` (alpha is *not* fed through the network —
   x4plus is a 3-channel model).
3. RGB → float32 NumPy `[0,1]` → tensor `(1,3,H,W)` → device, cast to fp16 if enabled.
4. Forward pass inside `torch.inference_mode()` (skips autograd bookkeeping);
   whole-image if it fits one tile, otherwise tiled (Section 7).
5. Result → float32 → `torch.clamp(0,1)` → uint8.
6. Alpha resized to output size with bicubic and re-stacked (RGBA preserved in PNG).
7. **Final crop-to-fill resize:** the composited image is resized with **Lanczos**
   to *cover* the exact `target_width`×`target_height` (default **8000×5000**) using
   scale `max(tw/w, th/h)` — enlarging small 4× results if needed, never stretching —
   then a centered crop trims the excess so the output is *exactly* the target size
   (target `0` disables; see Section 6.3).
8. Saved as PNG (lossless, alpha-capable) or JPEG at the chosen quality
   (alpha flattened onto white, since JPEG cannot store transparency).

### 5.5 Manual weight placement

If the download is blocked (firewall/proxy), place `RealESRGAN_x4plus.pth` manually
into `models/` from the [official release] — the app will detect and reuse it.

[official release]: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0

---

## 6. Resampling in This Project

A common assumption for upscaling tools is Lanczos resampling. Inside the **neural
network** there is still **no Lanczos** — the two upsampling mechanisms actually used
by the model are nearest-neighbor (feature space) and bicubic (alpha channel). Lanczos
appears only in the separate **final crop-to-fill resize** that pins the output to the
exact target size (Section 6.3).

### 6.1 Nearest-neighbor — inside the network (`rrdbnet.py`)

The network's spatial upsampling uses `F.interpolate(feat, scale_factor=2, mode="nearest")`
applied **twice** (feature space), each followed by a 3×3 conv + LeakyReLU:

```python
feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
```

**Why nearest (and why Lanczos is not applicable here):** at this point the data is a
64-channel *learned feature tensor*, not a displayable image. Classic kernel-based
resamplers (Lanczos/bicubic — which are designed for band-limited scalar image signals)
would inject ringing artifacts into activations the network never saw during training.
The x4plus generator was trained with exactly this nearest+conv pattern: the conv layers
*learn* the interpolation kernel, so nearest simply seeds the conv with replicated
values and the learned weights do the reconstruction. Changing the resampler here would
produce outputs inconsistent with the pretrained weights.

(For the 2×/1× variants the architecture instead uses `F.pixel_unshuffle` input packing —
also not kernel resampling.)

### 6.2 Bicubic — for the alpha channel (`realesrgan_upscaler.py`)

```python
alpha_resized = alpha.resize((w, h), Image.Resampling.BICUBIC)
```

The alpha mask is the one place where classical PIL resampling is used: the network is
3-channel and never sees transparency, so the 4×-larger alpha plane is produced with a
smooth 4-lobed cubic convolution kernel and re-composited onto the RGB result. Bicubic
was chosen over Lanczos deliberately: **Lanczos is an approximating sinc-kernel filter
that overshoots at sharp opacity edges**, which can push pixel values out of range and
create visible halos/fringing around transparent boundaries; bicubic is monotone and
clamped, giving a clean gradient at the cost of slightly less edge sharpness — the right
trade-off for a mask channel where color fidelity of the RGB is already handled by the CNN.

**If you want Lanczos output quality:** the model's learned reconstruction (Section 5.4
steps 4–5) is what provides photo realism; a Lanczos resize (e.g.,
`Image.resize(..., Image.Resampling.LANCZOS)`) would only be a non-AI classical
alternative, strictly lower quality at 4× for photographic content.

### 6.3 Lanczos — final crop-to-fill resize to the exact target (`realesrgan_upscaler.py`)

Because the model always runs at 4×, a large source (e.g. 5000×2500) produces a
20000×10000 result. After inference and alpha re-merge, `_fill_to_target()` resizes the
finished image so it **fully covers** the configured `target_width` × `target_height`
(default **8000 × 5000**), then takes a **centered crop** of exactly that size. The
scale is `max(target_w/w, target_h/h)` — a single uniform factor, so the aspect ratio
is preserved and the image is **never stretched**; a 4× result *smaller* than the
target is enlarged as needed to fill it. A 20000×10000 result becomes 10000×5000 and
is then center-cropped to 8000×5000; a result with the exact 8:5 ratio maps 1:1 with
no crop at all.

```python
img = img.resize((fill_w, fill_h), Image.Resampling.LANCZOS)
img = img.crop((left, top, left + target_w, top + target_h))  # centered
```

Lanczos is the high-quality choice for this resize in both directions: when throwing
pixels away it acts as an anti-aliasing kernel suppressing moiré, and when enlarging
it stays sharper than bilinear/bicubic. It runs on the fully composited RGB/RGBA
array, so **PNG transparency is preserved** and the existing JPEG behaviour (flatten
onto white) is unchanged. Setting either target to `0` disables crop-to-fill entirely
(the raw 4× size is saved). Crucially, this filter acts only on the final output — the
network's internal nearest/bicubic resampling (Sections 6.1–6.2) is untouched.

### 6.4 Summary table

| Where | Kernel | Data type | Rationale |
|---|---|---|---|
| `RRDBNet` upsample stages | **nearest-neighbor** + learned convs | 64-ch feature tensor | Matches training; the conv layers learn the reconstruction — Lanczos would ring and mismatch weights |
| Alpha channel resize | **bicubic** (`Image.Resampling.BICUBIC`) | 8-bit mask plane | Smooth, monotone, no overshoot halos at transparency edges (Lanczos overshoot risk) |
| Final crop-to-fill resize | **Lanczos** (`Image.Resampling.LANCZOS`) | RGB/RGBA output image | High-quality *uniform-scale* resize to cover `target_width`×`target_height`, then centered crop; keeps aspect ratio and PNG alpha, never stretches |
| JPEG/PNG encoding | none (lossless pixel write / DCT encode) | uint8 image | Resampling already done; no further resize occurs |

---

## 7. Tiled Inference & VRAM Management

Rendering an e.g. 6000×4000 image at 4× in one pass would allocate output feature maps
far beyond 6 GB of VRAM. `_tile_inference()` bounds peak memory regardless of input size:

1. The output canvas `B×3×(H·4)×(W·4)` is pre-allocated once **on CPU** in float32.
2. The full input image is moved to the device in the working dtype **once**
   (`base = t.to(device).half()`), avoiding repeated host→device copies.
3. The image is swept in row-major tile order. For each tile at grid `(tx, ty)`:
   - The **input** region is expanded by `tile_pad=16` px on each side (clamped at
     image borders) and fed to the network.
   - Only the tile's **core** (non-overlap) output region is sliced out using the
     computed paste offsets and written into the output canvas. The padded margin
     exists only to give the convolutions real context, which is what **hides seams**.
   - Intermediate tensors are `del`eted per iteration to free GPU memory promptly.
4. With `tile=256` and `pad=16`, each forward pass handles at most a 288×288 patch —
   comfortably within 6 GB even in fp16.

Special cases: `tile=0` disables tiling (single whole-image pass — only safe for small
inputs); images already smaller than one tile skip tiling entirely.

---

## 8. User Interface

Built with **PySide6** (`QMainWindow`, 940×620 default, Fusion style). Layout:

```
┌──────────────────────────────────────────────────────────────┐
│ Toolbar: [Add Images] [Add Folder] [Clear List] | [About]    │
├──────────────────────────────┬───────────────────────────────┤
│ Images to process            │ Settings                      │
│  • QListWidget (queue)       │  • Output folder + Browse…    │
│  • drag & drop target        │  • ☐ Save beside original     │
│  • [Add Images…] [Add        │  • Device: Auto/CUDA/CPU      │
│    Folder…] [Remove Sel.]    │  • ☑ Half precision (fp16)    │
│    [Clear]                   │  • Tile size spin (0–1024)    │
│                              │  • Target size 8000 × 5000    │
│                              │  • Format png/jpg + quality   │
│                              │  • Filename suffix (_x4)      │
├──────────────────────────────┴───────────────────────────────┤
│ [Start Upscaling] [Cancel]  status label   ▓▓▓▓░░ progress   │
├──────────────────────────────────────────────────────────────┤
│ Read-only log pane (QPlainTextEdit, capped at 2000 blocks)   │
├──────────────────────────────────────────────────────────────┤
│ Status bar: GPU name/VRAM/CUDA state or CPU-fallback warning │
└──────────────────────────────────────────────────────────────┘
```

Behavior highlights:

- **Startup CUDA probe** (`cuda_info()`): if CUDA is detected the device combo
  defaults to *CUDA (GPU)*, fp16 is pre-checked, and tile size is pre-set from
  `recommended_tile_for_vram()`; otherwise a ⚠ CPU-fallback warning shows in the status bar.
- **Drag & drop:** the window accepts URL drops; folders in a drop are expanded
  recursively through `collect_images()` and are remembered as folder-mode roots,
  exactly like *Add Folder…*. Duplicates are ignored (case-insensitive).
- **Folder mode:** *Add Folder…* reuses the existing toolbar/button action — no
  separate button. It scans the folder recursively (skipping the folder's own
  `upscaled` subfolder), queues every supported image and logs that results will
  be saved under `<folder>/upscaled/` with subfolders preserved. An empty (or
  images-only-in-`upscaled`) folder shows a clear "No images found" dialog and
  queues nothing. The Settings panel carries a hint label explaining the
  `<folder>/upscaled/` destination; individually added images still follow the
  global Output folder / "Save beside each original" settings.
- **Model readiness hint:** the log notes at startup whether weights are cached or
  will auto-download.
- **Reactive controls:** the JPG quality spin enables only for `jpg`; the output-folder
  field disables when "Save beside each original" is ticked; during a run all settings
  widgets lock and Start/Cancel swap enabled states.
- **Live feedback:** per-image status (`Processing name…`), integer progress bar wired
  to `progress(done,total)`, first-run weight-download percentage via
  `download_progress(bytes,total)`, and a full scrolling log.
- **Cooperative cancel:** Cancel calls `worker.request_cancel()` — the batch finishes
  the current image, then stops and emits `cancelled` (no thread kill).
- **Safe shutdown:** `closeEvent` requests cancellation and `wait(3000)`s for the worker
  before allowing the window to close.
- **Completion dialog:** informational (`N ok`) or warning (`N failed — see log`).

Threading model: all inference lives in `BatchWorker.run()` on the `QThread`; the GUI
thread only ever receives queued signals — the window never freezes during a 30-image batch.

---

## 9. Local Setup — Step by Step

### 9.1 Prerequisites check

1. Verify Python: `python --version` → **3.10 or newer**.
2. Verify GPU driver: open *NVIDIA Control Panel* → system information shows
   **CUDA Version ≥ 12.x** (driver-level; the app's torch bundles its own CUDA runtime).

### 9.2 Automated setup (recommended)

Open PowerShell in `d:\Upscaler` and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

What `setup.ps1` does, in order:

1. Creates `.venv` in the project root (reuses it if present — the script is idempotent).
2. Upgrades `pip` inside the venv.
3. Installs the **CUDA 12.6** build of `torch` + `torchvision` from
   `download.pytorch.org/whl/cu126` (~2.6 GB — allow time).
4. Installs `PySide6`, `Pillow`, `numpy`.
5. Runs a CUDA verification one-liner; expect:

   ```
   torch 2.14.1+cu126
   CUDA available: True
   Device: NVIDIA GeForce RTX 3050 ...
   ```

   If it prints `CUDA available: False`, see Section 13 before continuing.

### 9.3 Manual setup (alternative)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install PySide6 Pillow numpy
```

(Or equivalently `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`,
which carries the CUDA index URL itself.)

### 9.4 Optional: pre-place the model weights

To avoid any download inside the app, download `RealESRGAN_x4plus.pth` (~67 MB) from the
[official release] and put it in `d:\Upscaler\models\`. Otherwise the app fetches it
automatically on first batch start.

### 9.5 Launch

- **Double-click `run.bat`**, or
- **PowerShell:** `.\run.ps1`, or
- **Direct:** `.\.venv\Scripts\python.exe -m app.main`

All three paths use the project's `.venv` and exit with guidance if it's missing.

---

## 10. Usage — Step by Step

1. **Add images** — via **Add Images…** (multi-select files), **Add Folder…** (recursive
   scan for supported formats: JPG/JPEG/PNG/BMP/TIFF/WEBP), or **drag & drop** files/folders
   onto the window. The queue count shows in the status label; duplicates are skipped.
   Folder scans deliberately ignore the folder's existing `upscaled` output subfolder,
   so a re-run never re-processes previous results. Selecting a folder that contains no
   supported images shows a clear message and adds nothing to the queue.
2. **Choose output location** — either browse to an **Output folder** (default suggestion:
   `~\Upscaled`), or tick **Save beside each original**. Images added through
   **Add Folder…** (or dropped as folders) instead always go to
   **`<selected folder>/upscaled/<same relative subfolder>/<name><suffix>.<ext>`** —
   e.g. `photos/trip/image.jpg` → `upscaled/photos/trip/image_x4.jpg` — which keeps
   subfolder structure and prevents name collisions between subfolders. Existing files
   are never overwritten in any mode — collisions get `(1)`, `(2)`, … suffixes
   automatically.
3. **Configure the engine**
   - **Device:** *CUDA (GPU)* (preselected when detected) or *CPU* (much slower).
   - **fp16:** keep checked on NVIDIA GPUs — roughly halves VRAM and speeds up inference.
   - **Tile size:** `256` is the recommended default for a 6 GB RTX 3050; `0` disables
     tiling (small images only). Lower to `192`/`128` if you hit out-of-memory errors.
   - **Exact target size:** width × height (default **8000 × 5000**). Every output
     is resized (Lanczos, uniform scale — never stretched) to *cover* these
     dimensions and then **center-cropped to exactly** width × height; 4× results
     smaller than the target are enlarged to fill it. `0` disables crop-to-fill
     and keeps the raw 4× size. Note that images whose aspect ratio differs from
     8:5 lose some content at the overspilled edges.
   - **Format:** `png` (lossless, preserves transparency) or `jpg` (+ quality 50–100;
     transparency is flattened onto white).
   - **Filename suffix:** appended to each output name, e.g. `photo_x4.png`.
4. **Click Start Upscaling.** The model loads once (with a progress readout in the status
   bar on the very first run while weights download), then images process back-to-back.
   Watch the per-image `[OK] name: WxH -> 4Wx4H -> center-cropped to exact target
   8000x5000` log lines (or `... -> resized to exact target ...` when no cropping was
   needed) and the progress bar.
5. **Cancel anytime** — the current image finishes, then the batch stops gracefully.
6. When the completion dialog appears, outputs are in your chosen location — every
   one of them **exactly** the configured target size (default 8000×5000), with
   preserved color mode (RGB stays RGB; RGBA stays RGBA in PNG).

**Expected quality:** photos, screenshots, artwork, anime-style content all benefit;
the model hallucinates plausible high-frequency detail, so faces/text in very degraded
inputs may look "sharpened-plausible" rather than perfectly faithful.

---

## 11. Testing & Verification

All tests are self-contained scripts (no pytest needed); they use temporary directories
and generate synthetic inputs.

### 11.1 Engine smoke test (headless, no GUI)

```powershell
.\.venv\Scripts\python.exe -m tests.smoke_test
.\.venv\Scripts\python.exe -m tests.smoke_test --cpu     # force CPU
.\.venv\Scripts\python.exe -m tests.smoke_test --tile 32 # custom tile
```

Creates 3 small RGB PNGs (with crisp diagonal lines) + 1 RGBA PNG, runs a real 4× batch
through `RealESRGANUpscaler` (crop-to-fill disabled via `target_width=target_height=0`),
and asserts every output is exactly **4× the source dimensions with the same color
mode**. Prints `SMOKE TEST PASSED` / exit code 0 on success.

### 11.2 Worker batch test (GUI code path, offscreen Qt)

```powershell
.\.venv\Scripts\python.exe -m tests.worker_batch_test
```

Runs `BatchWorker` synchronously (`QT_QPA_PLATFORM=offscreen`) over a 6-image batch with
the **default 8000×5000 target left enabled**, asserting all 6 outputs exist at **exactly
8000×5000**, zero failures, exactly one `"Model loaded and ready."` log (proving the
**load-once-per-batch** invariant), and 6 progress events emitted.

### 11.3 Target-size / crop-to-fill / transparency test (headless, no model)

```powershell
.\.venv\Scripts\python.exe -m tests.target_size_test
```

Focused on the exact-output-size guarantee. Verifies `compute_fill_dimensions()` covers
the target with a uniform scale for many shapes (downscale, upscale, same-ratio, extreme
aspect ratios), that `_fill_to_target()` lands on **exactly** the target dimensions for
every shape (including tiny sources that must be enlarged) and disables cleanly at `0×0`,
that the crop is provably **centered** (edge content cut, middle content kept, symmetric
output), and exercises `_save()` on real RGBA arrays to confirm PNG transparency survives
crop-to-fill while JPEG still flattens to RGB. Prints `TARGET SIZE TEST PASSED` /
exit code 0 on success.

### 11.4 Folder-mode test (headless, no model)

```powershell
.\.venv\Scripts\python.exe -m tests.folder_mode_test
```

Exercises the *Add Folder…* path logic without loading the network: recursive
scanning with `upscaled`-folder exclusion, empty-folder handling,
structure-preserving `output_base_dir()` mapping (`<folder>/upscaled/<sub>`),
non-collision of same-named files from different subfolders, honouring of
suffix/format settings, fallback to the classic output modes for individually
added images, and the `(1)` collision suffix. Prints `FOLDER MODE TEST PASSED`
/ exit code 0 on success.

---

## 12. Performance Tuning

| GPU VRAM | Recommended tile | Notes |
|---|---|---|
| ≥ 12 GB | 512 | Fewer tiles → fewer seams, fastest |
| 8–12 GB | 384 | Good balance |
| **6 GB (RTX 3050)** | **256** | Project default; auto-selected |
| 4–6 GB | 192 | Lower if OOM occurs |
| < 4 GB / CPU | 128 | CPU will be *very* slow at 4× |

- `tile = 0` (whole-image pass) is fastest for small inputs but unsafe for large ones.
- Keep **fp16 on** with CUDA — the largest single speed/VRAM win.
- `tile_pad` is fixed at 16 in the GUI; increasing it (code change) further reduces
  seam visibility at the cost of redundant compute.
- Batch throughput scales with queue size because model load (~seconds) is amortized
  over all images; prefer one big batch over many small ones.

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA available: False` after setup | CPU-only torch wheel installed, or outdated NVIDIA driver | Update the driver (CUDA 12.x capable); reinstall with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` (note: `--index-url`, not `--extra-index-url`, to force the CUDA build) |
| `CUDA out of memory` | Tile too large for the scene | Reduce **Tile size** (256 → 192 → 128); keep fp16 on |
| `PyTorch is not installed correctly` at launch / `.venv` missing | Setup never ran | Run `setup.ps1` (Section 9.2) |
| `Failed to download weights` / "looks incomplete" | Firewall/proxy blocking GitHub; interrupted download | Manually place `RealESRGAN_x4plus.pth` in `models/` from the [official release]; truncated files are auto-purged on next attempt |
| Seams visible in output | Very small tiles / low-contrast content | Increase tile size or lower it less aggressively; default 256 rarely shows seams |
| Alpha edges haloed in PNG | Expected with bicubic mask smoothing | Cosmetic; crop/soften alpha beforehand if critical |
| JPEG output loses transparency | JPEG format has no alpha channel | Use **png** to preserve RGBA (the app flattens onto white for jpg by design) |
| App window freezes | Should not occur (inference is on a QThread) | Check log for a per-image exception loop; update GPU driver |
| Wrong rotation from phones | Usually handled by EXIF transpose | If a camera embeds unusual metadata, re-save once via Paint/PIL |

---

## 14. License & Credits

- **Real-ESRGAN** is by Xintao Wang et al. ([xinntao/Real-ESRGAN], **BSD-3-Clause**).
  This project reuses the public pretrained `x4plus` weights and re-implements the
  network architecture for local inference only.
- **PyTorch** (BSD-style), **PySide6** (LGPLv3/commercial), **Pillow** (MIT-CMU),
  **NumPy** (BSD) — review each upstream license for distribution/commercial use.
- Output images are for personal use under the model's license terms; the BSD-3
  license requires retaining the original copyright notice in redistributions of
  the model/code.

[official release]: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0
[xinntao/Real-ESRGAN]: https://github.com/xinntao/Real-ESRGAN
