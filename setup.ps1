<#
.SYNOPSIS
    One-time environment setup for the Batch Image Upscaler.
.DESCRIPTION
    Creates a local .venv and installs the CUDA build of PyTorch plus the GUI
    dependencies. Run this once before using run.bat / run.ps1.
.NOTES
    Requires Python 3.10+ (tested on 3.14) and an NVIDIA GPU with a CUDA 12.x
    driver. Requires internet access on first run (large PyTorch download).
#>

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "== Batch Image Upscaler setup ==" -ForegroundColor Cyan

if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "Virtual environment already exists - reusing it." -ForegroundColor Yellow
} else {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
}

$py = ".venv\Scripts\python.exe"

Write-Host "Upgrading pip..."
& $py -m pip install --upgrade pip

Write-Host "Installing PyTorch (CUDA 12.6 build) - this is a ~2.6 GB download..."
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Write-Host "Installing GUI + imaging dependencies..."
& $py -m pip install PySide6 Pillow numpy

Write-Host "Installing settings validation dependency (Django)..."
& $py -m pip install "Django>=5.2"

Write-Host "Installing Metadata Generator dependencies..."
& $py -m pip install google-genai requests keyring

Write-Host "`nVerifying CUDA availability..."
& $py -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "`nSetup complete. Launch the app with run.bat (double-click) or run.ps1." -ForegroundColor Green
