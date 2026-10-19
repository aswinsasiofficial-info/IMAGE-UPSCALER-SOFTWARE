# Batch Image Upscaler (Real-ESRGAN x4plus)

A local, GPU-accelerated desktop app for **batch-upscaling images 4×** with the
**Real-ESRGAN `x4plus`** model, built on **PyTorch + CUDA**. Designed to squeeze
the most out of an **NVIDIA RTX 3050 (6 GB VRAM)** and 16 GB RAM.

Everything runs on your machine — no uploads, no accounts, no internet needed
after the model weights are downloaded once.

---

## Features

- **Batch processing** — queue 20-30+ images at once (files or whole folders,
  recursively). The model is loaded **once** and reused for every image.
- **Real-ESRGAN x4plus** — high-quality 4× super-resolution for photos,
  screenshots, artwork and anime-style images.
- **CUDA + fp16** — runs on the GPU in half precision for speed and lower VRAM.
- **VRAM-safe tiling** — large images are automatically split into overlapping
  tiles so they fit in 6 GB of VRAM (no out-of-memory crashes on big photos).
- **RGBA / transparency preserved** — alpha channels are upscaled cleanly.
- **Max output size** — the 4× result is downscaled (high-quality Lanczos) to fit
  within a configurable cap (default **8000 × 5000 px**), preserving aspect ratio.
- **PNG or JPEG output**, custom output folder (or save beside originals),
  configurable suffix, live progress bar, per-image logging and cancel.
- **Drag & drop** or browse to add images.

## Requirements

- Windows with an **NVIDIA GPU** and a **CUDA 12.x driver** (RTX 3050 works).
- **Python 3.10+** (developed/tested on 3.14).
- ~4 GB free disk space for the CUDA PyTorch wheel + dependencies.
- Internet on first run (PyTorch download, then one-time 67 MB weight download).

## Setup (one time)

Open PowerShell in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This creates a `.venv`, installs the **CUDA 12.6** build of PyTorch, PySide6,
Pillow and NumPy, then prints whether CUDA is detected. If it says
`CUDA available: True`, you're good.

<details>
<summary>Manual setup</summary>

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install PySide6 Pillow numpy
```
</details>

## Run

Double-click **`run.bat`**, or from PowerShell:

```powershell
.\run.ps1
```

Then:

1. **Add Images…** / **Add Folder…** (or drag & drop onto the window).
2. Pick an **output folder**, or tick **Save beside each original**.
3. Choose **Device** (CUDA), keep **fp16** on, a **tile size** (256 is the
   recommended default for a 6 GB GPU; `0` = no tiling), and the **max output
   size** (default 8000 × 5000 — the 4× result is downscaled to fit within it).
4. Click **Start Upscaling**. Watch the progress bar and log; **Cancel** anytime.

On the very first run the app downloads `RealESRGAN_x4plus.pth` into `models/`
and shows download progress in the status bar.

## How it works

| Concern | Approach |
|---|---|
| Model | Self-contained `RRDBNet` re-implementation (identical architecture & weights to the official `RealESRGAN_x4plus.pth`), so we avoid the `basicsr`/`realesrgan` packages that break on modern torchvision. |
| Batching | One model load for the whole queue; images are processed back-to-back on the GPU. |
| VRAM | Tile-based inference with an overlap pad (`tile=256`, `pad=16`) bounds peak memory for arbitrarily large inputs. |
| Speed | `torch.inference_mode` + fp16 + cuDNN on the RTX 3050. |
| Transparency | RGB is upscaled by the network; the alpha channel is resized with bicubic and recombined. |
| Output size | After 4× inference the composited image is downscaled with Lanczos to fit the configurable max (default 8000 × 5000), aspect ratio preserved. |

### Tuning for your GPU

- **RTX 3050 6 GB (yours):** keep **tile = 256**. If you hit an out-of-memory
  error on an enormous image, lower it (e.g. `192`/`128`). Larger tiles = fewer
  seams but more VRAM.
- **More VRAM / faster:** `384`–`512` on 8-12 GB cards.
- **`tile = 0`:** process the whole image in one pass — only safe for small
  inputs; disables tiling.

## Verify the engine (optional, no GUI)

```powershell
.\.venv\Scripts\python.exe -m tests.smoke_test
```

Generates a few test images, runs a real 4× batch and asserts the outputs are
exactly 4× with the correct color mode.

```powershell
.\.venv\Scripts\python.exe -m tests.max_size_test
```

Checks the max-output-size behaviour: aspect-ratio/fit-inside math, the reference
`20000×10000 → 8000×4000` case, and that a PNG with transparency keeps its alpha
after the final Lanczos downscale (no GPU/model required).

## Project layout

```
app/
  main.py                     # GUI entry point (python -m app.main)
  engine/
    rrdbnet.py                # Real-ESRGAN x4plus network architecture
    realesrgan_upscaler.py    # weights, tiled inference, image I/O
    worker.py                 # QThread batch runner (progress + cancel)
  ui/
    main_window.py            # PySide6 desktop UI
tests/
  smoke_test.py               # headless end-to-end check
  max_size_test.py            # max-output-size + transparency resize check
models/                       # downloaded weights (auto-created)
setup.ps1 · run.bat · run.ps1 · requirements.txt
```

## Troubleshooting

- **`CUDA available: False`** → update your NVIDIA driver, or the installed
  torch is the CPU build. Reinstall with the `cu126` index command above.
- **Out of memory** → reduce **Tile size**, keep **fp16** enabled.
- **App won't launch** → ensure you ran `setup.ps1`; run `run.bat` from this
  folder. Check the console output for the Python error.
- **Weights download blocked** → manually place `RealESRGAN_x4plus.pth` from the
  [official release](https://github.com/xinntao/Real-ESRGAN/releases) into the
  `models/` folder.

## Credits / license

Real-ESRGAN is by Xintao Wang et al. (xinntao/Real-ESRGAN, BSD-3-Clause). This
project reuses the public pretrained weights and re-implements the network for
inference. Output is for personal use under the model's license terms.
