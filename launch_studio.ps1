# Tabular & NLP Studio Server Launcher (PowerShell Version)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  TABULAR & NLP STUDIO - Competitive Machine Learning       " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$modelsDir = Join-Path $PSScriptRoot "models"
if (Test-Path $modelsDir) {
    $env:HF_HOME = $modelsDir
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:HF_DATASETS_OFFLINE = "1"
    Write-Host "[OFFLINE READY] HuggingFace cache diatur ke: $modelsDir" -ForegroundColor Green
}

$venvPy = $null
$candidates = @(
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
    (Join-Path $PSScriptRoot "spaceship-titanic\.venv\Scripts\python.exe"),
    (Join-Path $PSScriptRoot "venv\Scripts\python.exe")
)

foreach ($c in $candidates) {
    if (Test-Path $c) {
        $venvPy = $c
        break
    }
}

if (-not $venvPy) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $venvPy = "python"
    } else {
        Write-Host "[INFO] Environment belum terpasang. Menjalankan install_offline.ps1..." -ForegroundColor Yellow
        & (Join-Path $PSScriptRoot "install_offline.ps1")
        $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    }
}

Write-Host "Menggunakan Python: $venvPy" -ForegroundColor DarkGray
Write-Host "Memulai FastAPI Engine pada http://localhost:8000 ..." -ForegroundColor Green
Write-Host "Tekan Ctrl+C pada konsol ini untuk menghentikan server."
Write-Host ""

Start-Process "http://localhost:8000"
& $venvPy -m uvicorn server:app --host 127.0.0.1 --port 8000
