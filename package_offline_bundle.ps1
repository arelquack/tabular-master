# Package Full Offline Bundle for Gemastik Final (PowerShell Version)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  PACKAGE FULL OFFLINE BUNDLE (RUN ON PC DOSEN WITH FAST INTERNET)     " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Langkah ini akan menyiapkan seluruh kebutuhan offline proyek ini:"
Write-Host " 1. Menginstal & memverifikasi environment Python (.venv)"
Write-Host " 2. Mengunduh semua file wheel (.whl) ke ./wheels/ untuk instalasi offline"
Write-Host " 3. Mengunduh bobot model Transformer (IndoBERT dll) ke ./models/"
Write-Host ""
Write-Host "Setelah skrip ini selesai:"
Write-Host " Salin SELURUH folder ini ke Flashdisk"
Write-Host " Maka platform siap dijalankan 100% OFFLINE di PC Panitia!"
Write-Host "======================================================================"
Write-Host ""

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python tidak terdeteksi pada sistem PATH!" -ForegroundColor Red
    Write-Host "Silakan install Python 3.10+ pada PC Dosen terlebih dahulu." -ForegroundColor Yellow
    Read-Host "Tekan Enter untuk keluar"
    exit 1
}

$venvDir = Join-Path $PSScriptRoot ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "[1/4] Membuat Virtual Environment (.venv)..." -ForegroundColor Green
    & python -m venv "$venvDir"
} else {
    Write-Host "[1/4] Virtual Environment (.venv) sudah tersedia." -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/4] Menginstal & memperbarui dependencies dasar..." -ForegroundColor Green
& $venvPy -m pip install --upgrade pip
& $venvPip install -r requirements.txt

Write-Host ""
Write-Host "[3/4] Mengunduh seluruh Python Wheels (.whl) ke folder ./wheels/ ..." -ForegroundColor Green
$wheelsDir = Join-Path $PSScriptRoot "wheels"
if (-not (Test-Path $wheelsDir)) {
    New-Item -ItemType Directory -Path $wheelsDir | Out-Null
}
& $venvPip download -r requirements.txt -d "$wheelsDir"

Write-Host ""
Write-Host "[4/4] Mengunduh bobot model Pretrained Transformer ke ./models/ ..." -ForegroundColor Green
$downloadScript = Join-Path $PSScriptRoot "scripts\download_models.py"
& $venvPy "$downloadScript" @args

Write-Host ""
$verifyScript = Join-Path $PSScriptRoot "scripts\verify_bundle.py"
& $venvPy "$verifyScript"

Write-Host ""
Write-Host "Selesai. Tekan Enter untuk menutup jendela ini..." -ForegroundColor Cyan
Read-Host
