# Launches the Batch Image Upscaler GUI using the local .venv.
# If the environment is missing, it tells you to run setup.ps1 first.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup.ps1" -ForegroundColor Yellow
    exit 1
}

& $py -m app.main
