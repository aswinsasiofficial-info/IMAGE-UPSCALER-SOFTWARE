@echo off
REM Double-click to launch the Batch Image Upscaler (uses the local .venv).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run setup first:  powershell -ExecutionPolicy Bypass -File setup.ps1
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m app.main
