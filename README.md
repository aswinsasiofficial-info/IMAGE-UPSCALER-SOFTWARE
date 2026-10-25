# Batch Upscaler & Metadata Generator

A local Windows **desktop app** with two separate tools:

- **Upscaler** — enlarges images **1×–5×** with the **Real-ESRGAN `x4plus`**
  model on **PyTorch + CUDA**. Every image keeps its own aspect ratio and its
  whole picture.
- **Metadata** — writes Adobe Stock–ready titles, descriptions and keywords
  with **Gemini**, **Groq** or **Ollama** (your own local server *or* Ollama's
  hosted cloud), and exports them as CSV.

The two never trigger each other: upscale and save first, then open
**Metadata** and choose the images or folder you want to describe. Upscaling
runs entirely on your machine; the Metadata screen only contacts the provider
you pick, for the images you pick (local Ollama keeps even those on your PC).
There is no website or web server — it's a desktop program.

---

## Upscaler

- **Factors 1×, 2×, 3×, 4×, 5×** — the same factor is applied to width and
  height, so nothing is cropped, stretched, zoomed or distorted. 1× keeps the
  original size; a 1024 × 1536 image at 4× becomes exactly 4096 × 6144.
- **Honest about the model** — Real-ESRGAN x4plus has exactly one native scale,
  **4×** (its network ends in two fixed 2× upsampling stages). The screen says
  how each factor is produced:
  - **4×** — native.
  - **1×, 2×, 3×** — the model runs at 4×, then the result is reduced with a
    high-quality Lanczos filter (1× = a clean-up at the original size).
  - **5×** — the model runs at 4×; the last step is ordinary Lanczos
    interpolation, which adds pixels but no new detail.

  More pixels don't automatically mean more real detail: the AI reconstructs
  plausible detail, it can't recover what the original never captured.
- **See it before you run it** — every queued image shows its thumbnail, its
  original size and shape, and the exact pixel size it will be saved at; the
  selected image is shown large. Outputs that would be too large (over
  65,500 px a side) or need a lot of memory are flagged.
- **JPEG by default** (quality 95), PNG optional — PNG keeps transparency;
  JPEG flattens it onto white.
- **Batch rename by default** — one random 4–5 letter prefix for the batch,
  numbered from 1 in each output folder (`KQATZ1.jpg`, `KQATZ2.jpg`, …). Untick
  it to keep the original names. **Existing files are never overwritten** —
  taken numbers are skipped, kept names get `(1)`, `(2)`, ….
- **Folder mode** — images from an added folder save to `<folder>/upscaled/`
  with subfolders preserved; that folder is skipped on later scans.
- **Fast and safe on a 6 GB GPU** — the model loads once per batch, runs in
  fp16 with overlapping tiles, writes into an 8-bit canvas, and if CUDA still
  runs out of memory the tile size is halved and the image retried.
- EXIF rotation, transparency and RGB colour profiles (e.g. Display P3) are kept.

## Metadata

- **Images and folders** — add images, or a folder: a **folder batch holds up
  to 25 images**. If a folder has more, a chooser shows all of them as
  thumbnails (the first 25 ticked) and you pick up to 25.
- **One tile per image** — thumbnail, filename, status (not generated /
  generating / done / failed), keyword count and title, or the error. Filters
  (All / Pending / Done / Failed) with counts; overall progress below.
- **Large preview + editor** — the selected image next to its editable title,
  description, keywords, Adobe Stock **category** (suggested by the AI, pick
  another from the list) and **releases** (blank unless you type the release
  file names — the AI never fills them), with counters against Adobe's limits.
  Edits are kept per image.
- **Automatic retries, failed images only** — temporary problems (rate limits,
  timeouts, provider 5xx) are retried automatically with growing waits, only
  for the images that failed; finished images are never regenerated.
  Permanent problems (invalid key, missing model, unsupported image, a local
  server that isn't running) stop right away with an explanation.
- **Complete CSV only** — once *every* image in a folder batch has metadata,
  `metadata.csv` is saved into that folder automatically. If any image is
  still unsuccessful, or the batch is cancelled, no partial CSV is written.
- **Export** — saves the current metadata, including your edits, straight to
  CSV. It asks where only if no location is chosen yet (after an automatic
  save it updates that file); *Export As…* picks another. The same menu offers
  **Embed Metadata into Images…** (PNG tEXt / JPEG comment — as copies or in
  place; this is where the description goes).
- **Adobe Stock CSV format** — exactly the template's columns, in order:
  `Filename,Title,Keywords,Category,Releases`. Keywords share one cell,
  comma-separated, most important first, at most 49 (extra ones are left out
  and the banner says so); Category is the category number (1–21); Releases is
  blank unless you entered it. Values with commas or quotes are quoted, so a
  keyword list never spills into extra columns. UTF-8 with BOM, so Excel shows
  non-English text correctly.
- **Providers** — Google **Gemini** (official SDK), **Groq** (OpenAI-compatible
  vision models), **Ollama (Local)** (`http://localhost:11434`, no key, nothing
  uploaded — the check button lists installed models that can read images) and
  **Ollama Cloud** (hosted, Bearer key; the screen says that images are
  uploaded). Ollama asks for exactly 49 keywords; Gemini/Groq up to 45.
- **Keys** — typed keys stay masked, each provider keeps its own key, optionally
  remembered in the **Windows Credential Manager** or read from
  `GEMINI_API_KEY` / `GROQ_API_KEY` / `OLLAMA_API_KEY`. Keys are never exported.

## Design

- **Inter Tight** everywhere (bundled, SIL Open Font License), one colour
  palette for light and dark mode (follows Windows; `UPSCALER_THEME=light|dark`
  forces one), consistent cards, buttons and spacing on both screens.
- **Icon buttons** (Lucide icons, ISC license) replace long button text. Each
  has a tooltip that says what it does (and its shortcut) and an accessible
  name for screen readers.
- **Keyboard** — `Ctrl+1` / `Ctrl+2` switch screens; `Ctrl+O` add images,
  `Ctrl+Shift+O` add a folder, `Del` remove selected, `Ctrl+Enter`
  Upscale / Generate, `Esc` cancel, `Ctrl+E` export.
- Layouts adapt to the window size (the Metadata screen shows the editor and
  provider settings as tabs when narrower).

## Get the app

### Packaged app (no Python needed)

Build it once (see below), then copy the `dist\BatchUpscaler` folder anywhere
and start **`BatchUpscaler.exe`**. The folder contains everything — Python,
PyTorch with CUDA, Qt, the fonts and icons, and the model weights — so it works
offline. An NVIDIA GPU with a current driver is used automatically; without
one it runs on the CPU (much slower).

### From source

Requirements: Windows, an NVIDIA GPU with a CUDA 12.x driver (an RTX 3050
works; CPU also works, slowly), Python 3.10+ (developed on 3.14) and about
4 GB of disk for the CUDA PyTorch download.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1   # once: creates .venv and installs everything
.\run.bat                                              # or .\run.ps1
```

`setup.ps1` installs the CUDA 12.6 build of PyTorch, PySide6, Pillow, NumPy,
Django and the Metadata providers, then prints whether CUDA is detected. On
the first upscale the app downloads `RealESRGAN_x4plus.pth` (67 MB) into
`models/`.

### Build the packaged app

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Uses PyInstaller from the `.venv` to make `dist\BatchUpscaler\BatchUpscaler.exe`
(a one-folder app, several GB because of CUDA PyTorch), copies the model
weights next to it, then runs the packaged app's self-test. You can repeat the
self-test any time:

```powershell
dist\BatchUpscaler\BatchUpscaler.exe --self-test report.json
```

It checks the font and icons, PyTorch/CUDA, and runs the real model at every
factor 1×–5× (sizes, aspect ratio, JPEG + batch names), writing the results
to `report.json` (exit code 0 = passed).

## How to use

**Upscaler** — add images or a folder (or drop them on the window), pick a
factor, check the sizes in the list, choose where to save, and press
**Upscale**. Watch the progress bar; open the activity log for details;
**Cancel** stops after the current tile and keeps finished images.

**Metadata** — switch to **Metadata**, add images or a folder (≤ 25 images per
folder batch), choose the provider (enter a key, or use Ollama locally), press
**Generate**. Select an image (or use the **Failed** filter) to see what went
wrong; fix it and use **Retry failed**. Edit anything, then **Export**.

## Tests

All run from the project folder with `.\.venv\Scripts\python.exe -m tests.<name>`:

| Test | What it checks |
|---|---|
| `upscaler_test` | No model: 1×–5× sizes and the factor explanations, the Django settings form, real batches for portrait/landscape at every factor (size, shape, whole picture via shrink-back + corner markers), JPEG + rename defaults, no overwriting, folder mode, transparency, colour profiles, cancel — and that no web server/website remains |
| `upscaler_ui_test` | The Upscaler screen in the real window (offscreen): defaults, removed options, Inter Tight, every icon button has an icon + tooltip + accessible name, thumbnails and preview, sizes for 1×–5×, a batch run from the Upscale button, the too-large message |
| `upscaler_model_test` | The real Real-ESRGAN network at every factor on portrait, landscape, EXIF-rotated and transparent images (`--example` adds 1024 × 1536 → 4096 × 6144) |
| `metadata_ui_test` | Export (edits included, asks once, Export As), 25-image folder batches with automatic retries of failed images only, complete-only CSV (not after a failure or cancel), the 25-image limit and chooser, thumbnails, filters, layouts |
| `metadata_test`, `ollama_test` | Providers, parsing, retries, credentials, CSV/embedding, Ollama local + cloud — all with network calls stubbed |

`smoke_test`, `target_size_test`, `worker_batch_test`, `folder_mode_test` and
`batch_rename_test` cover the previous upscaler modules, which the app no
longer uses.

## Project layout

```
app/
  main.py                 # entry point (python -m app.main / BatchUpscaler.exe; --self-test)
  selftest.py             # headless check of a source or packaged build
  paths.py                # bundled assets and model folder (source vs packaged)
  design.py               # colours, font, radii (light + dark)
  imagefiles.py           # finding supported images (shared by both screens)
  assets/                 # Inter Tight font, Lucide icons, app icon (+ licenses)
  upscaler/
    geometry.py           # 1×–5× size planning + how each factor is produced
    engine.py · network.py  # Real-ESRGAN x4plus: weights, device, tiles, OOM retry
    imaging.py            # EXIF-aware load, resize to plan, atomic save
    files.py              # output folders, batch rename, no overwrite
    workspace.py          # queue + background batch (progress, log, cancel)
    forms.py              # settings validation (Django forms, standalone)
  engine/
    metadata_settings.py · metadata_providers.py · metadata_worker.py
    metadata_export.py · credential_store.py
    worker.py · realesrgan_upscaler.py · rrdbnet.py   # previous upscaler (unused)
  ui/
    main_window.py        # top navigation + the two screens
    upscaler_screen.py    # Upscaler screen
    metadata_page.py      # Metadata screen
    folder_picker.py      # choose up to 25 images of a big folder
    widgets.py · icons.py · thumbnails.py · theme.py
    upscale_page.py       # previous upscaler screen (unused)
packaging/                # PyInstaller spec, launcher, icon generator
tests/
setup.ps1 · run.bat · run.ps1 · build.ps1 · requirements.txt
```

## Troubleshooting

- **`CUDA available: False`** → update the NVIDIA driver, or the CPU build of
  torch is installed; reinstall with the `cu126` command in `setup.ps1`.
- **Out of memory** → the Upscaler halves the tile size and retries by itself;
  if it still fails, lower **Tile size** and keep **fp16** on.
- **"…would be larger than 65,500 px on a side"** → pick a smaller factor for
  that image (JPEG can't be bigger, and PNGs that size need huge amounts of RAM).
- **App won't start (from source)** → run `setup.ps1` first, then `run.bat`
  from this folder; the console shows the Python error.
- **Weights download blocked** → put `RealESRGAN_x4plus.pth` from the
  [official release](https://github.com/xinntao/Real-ESRGAN/releases) into
  `models\` (next to `BatchUpscaler.exe` for the packaged app).
- **Metadata: 401/403** → wrong or expired API key for that provider.
- **Metadata: 404 "no longer available"** → the model was retired; the message
  names the replacement — type it into the (editable) model box.
- **Ollama (Local): "Ollama is not running on this computer"** → start Ollama
  (or `ollama serve`), confirm `ollama list` shows a model, and press the check
  button again. This is never retried automatically.
- **Ollama (Local): "none of its model(s) can analyse images"** → pull a model
  that reads images, e.g. `ollama pull gemma4`.
- **Ollama: "answered with no text" / "no complete JSON object"** → try a
  larger image-capable model or fewer keywords; incomplete JSON is treated as
  temporary and retried on its own.
- **Ollama Cloud: 402 / "rejected your API key"** → the account has no plan or
  credits for that model, or the key is wrong
  (<https://ollama.com/settings/keys>).

## Credits / licenses

- Real-ESRGAN by Xintao Wang et al. (xinntao/Real-ESRGAN, BSD-3-Clause) — the
  public x4plus weights; the network is re-implemented for inference.
- Inter Tight by The Inter Project Authors — SIL Open Font License 1.1
  (`app/assets/fonts/OFL.txt`).
- Lucide icons — ISC License (`app/assets/icons/LICENSE-lucide.txt`).
- PyTorch (BSD-style), PySide6/Qt (LGPLv3), Pillow (MIT-CMU), NumPy (BSD),
  Django (BSD) — check each license before redistributing the packaged app.
