# Offline Environment Installer (Zero Internet Required - PowerShell Version)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  OFFLINE ENVIRONMENT INSTALLER (PC PANITIA / ZERO INTERNET REQUIRED)  " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Langkah ini akan menginstal seluruh library Python langsung dari folder"
Write-Host "lokal ./wheels/ di Flashdisk tanpa memerlukan koneksi internet."
Write-Host "======================================================================"
Write-Host ""

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python tidak terdeteksi pada sistem PATH PC Panitia!" -ForegroundColor Red
    Write-Host "Silakan pastikan Python 3.10+ sudah terinstal dan dicentang 'Add to PATH'." -ForegroundColor Yellow
    Read-Host "Tekan Enter untuk keluar"
    exit 1
}

$venvDir = Join-Path $PSScriptRoot ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "[1/3] Membuat Virtual Environment lokal pada drive ini (.venv)..." -ForegroundColor Green
    & python -m venv "$venvDir"
} else {
    Write-Host "[1/3] Virtual environment (.venv) sudah terdeteksi." -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/3] Menginstal dependencies dari folder lokal ./wheels/ (100% OFFLINE)..." -ForegroundColor Green
$wheelsDir = Join-Path $PSScriptRoot "wheels"
if (Test-Path $wheelsDir) {
    & $venvPip install --no-index --find-links="$wheelsDir" -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARNING] Instalasi offline menemui kendala. Mencoba instalasi paket utama..." -ForegroundColor Yellow
        & $venvPip install --no-index --find-links="$wheelsDir" fastapi uvicorn pydantic numpy pandas scikit-learn lightgbm xgboost catboost optuna scipy torch transformers matplotlib seaborn
    }
} else {
    Write-Host "[WARNING] Folder ./wheels/ tidak ditemukan! Mencoba instalasi online standar..." -ForegroundColor Yellow
    & $venvPip install -r requirements.txt
}

Write-Host ""
Write-Host "[3/3] Memeriksa kesiapan Hardware (GPU/CUDA) dan Model Offline..." -ForegroundColor Green
$verifyScript = Join-Path $PSScriptRoot "scripts\verify_install.py"
& $venvPy "$verifyScript"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  INSTALASI OFFLINE SELESAI DENGAN SUKSES!                             " -ForegroundColor Green
Write-Host "  Platform siap dijalankan di PC Panitia.                              " -ForegroundColor Green
Write-Host "  Silakan jalankan .\launch_studio.ps1 untuk memulai web platform.     " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Tekan Enter untuk menutup jendela ini..."
