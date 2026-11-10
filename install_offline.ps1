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

# 1. Deteksi Python 3.11 Interpreter
$basePy = $null

# Cek py launcher untuk versi 3.11
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    $testPy311 = & py -3.11 -c "import sys; print(sys.version_info[1])" 2>$null
    if ($testPy311 -and $testPy311.Trim() -eq "11") {
        $basePy = "py -3.11"
        Write-Host "[OK] Python 3.11 terdeteksi via Python Launcher (py -3.11)" -ForegroundColor Green
    }
}

# Cek path instalasi standar Python 3.11
if (-not $basePy) {
    $stdPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Python311\python.exe"
    )
    foreach ($p in $stdPaths) {
        if (Test-Path $p) {
            $basePy = "& `"$p`""
            Write-Host "[OK] Python 3.11 ditemukan di: $p" -ForegroundColor Green
            break
        }
    }
}

# Cek perintah 'python' default di PATH
if (-not $basePy) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $verMinor = & python -c "import sys; print(sys.version_info[1])" 2>$null
        if ($verMinor -and $verMinor.Trim() -eq "11") {
            $basePy = "python"
            Write-Host "[OK] Default Python di PATH adalah Python 3.11" -ForegroundColor Green
        } else {
            Write-Host "======================================================================" -ForegroundColor Yellow
            Write-Host "[PERINGATAN KRUSIAL] Terdeteksi Python 3.$verMinor pada PATH!" -ForegroundColor Red
            Write-Host "Pustaka Machine Learning (PyTorch, CatBoost, LightGBM) dan file wheels" -ForegroundColor Yellow
            Write-Host "pada proyek ini secara spesifik dikompilasi untuk Python 3.11." -ForegroundColor Yellow
            Write-Host "Python 3.$verMinor TIDAK KOMPATIBEL dengan binary wheels offline!" -ForegroundColor Red
            Write-Host "Silakan install Python 3.11.9 (64-bit) di PC Panitia." -ForegroundColor Yellow
            Write-Host "======================================================================" -ForegroundColor Yellow
            $basePy = "python"
        }
    } else {
        Write-Host "[ERROR] Python tidak terdeteksi pada sistem PATH PC Panitia!" -ForegroundColor Red
        Write-Host "Silakan install Python 3.11.9 (64-bit) dan centang 'Add python.exe to PATH'." -ForegroundColor Yellow
        Read-Host "Tekan Enter untuk keluar"
        exit 1
    }
}

$venvDir = Join-Path $PSScriptRoot ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "[1/3] Membuat Virtual Environment lokal pada drive ini (.venv)..." -ForegroundColor Green
    if ($basePy -eq "py -3.11") {
        & py -3.11 -m venv "$venvDir"
    } elseif ($basePy -like "& *") {
        Invoke-Expression "$basePy -m venv `"$venvDir`""
    } else {
        & python -m venv "$venvDir"
    }
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
