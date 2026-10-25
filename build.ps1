<#
.SYNOPSIS
    Packages the desktop app as dist\BatchUpscaler\BatchUpscaler.exe.
.DESCRIPTION
    Builds a one-folder Windows app with PyInstaller from the local .venv
    (run setup.ps1 first), copies the Real-ESRGAN weights next to the exe so
    it works offline from the first start, then runs the packaged app's
    headless self-test. Copy or zip the whole dist\BatchUpscaler folder.
#>

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "== Building BatchUpscaler ==" -ForegroundColor Cyan
& $py -m pip install --quiet --upgrade "pyinstaller>=6.16"
& $py -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\BatchUpscaler.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$app = "dist\BatchUpscaler"
if (Test-Path "models\RealESRGAN_x4plus.pth") {
    New-Item -ItemType Directory -Force -Path "$app\models" | Out-Null
    Copy-Item "models\RealESRGAN_x4plus.pth" "$app\models\" -Force
    Write-Host "Model weights included."
} else {
    Write-Host "No local weights - the app downloads them on first use." -ForegroundColor Yellow
}

Write-Host "Self-testing the packaged app..."
$report = Join-Path (Resolve-Path "dist") "self-test.json"
$proc = Start-Process -FilePath "$app\BatchUpscaler.exe" -ArgumentList "--self-test", "`"$report`"" -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    Write-Host "Self-test FAILED - see $report" -ForegroundColor Red
    exit 1
}
Write-Host "Self-test passed ($report)." -ForegroundColor Green
Write-Host "Done: $app\BatchUpscaler.exe" -ForegroundColor Green
