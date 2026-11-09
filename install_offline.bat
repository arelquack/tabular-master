@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Offline Environment Installer (Zero Internet Required)

echo ======================================================================
echo   OFFLINE ENVIRONMENT INSTALLER (PC PANITIA / ZERO INTERNET REQUIRED)
echo ======================================================================
echo.
echo Langkah ini akan menginstal seluruh library Python langsung dari folder
echo lokal ./wheels/ di Flashdisk tanpa memerlukan koneksi internet.
echo ======================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak terdeteksi pada sistem PATH PC Panitia!
    echo Silakan pastikan Python 3.10+ sudah terinstal dan dicentang 'Add to PATH'.
    pause
    exit /b 1
)

set "VENV_DIR=%~dp0.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/3] Membuat Virtual Environment lokal pada drive ini (.venv)...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] Gagal membuat virtual environment lokal.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment (.venv) sudah terdeteksi.
)

echo.
echo [2/3] Menginstal dependencies dari folder lokal ./wheels/ (100%% OFFLINE)...
if exist "wheels" (
    "%VENV_DIR%\Scripts\pip.exe" install --no-index --find-links=./wheels -r requirements.txt
    if %errorlevel% neq 0 (
        echo [WARNING] Instalasi offline menemui kendala. Mencoba instalasi toleran...
        "%VENV_DIR%\Scripts\pip.exe" install --no-index --find-links=./wheels fastapi uvicorn pydantic numpy pandas scikit-learn lightgbm xgboost catboost optuna scipy torch transformers matplotlib seaborn
    )
) else (
    echo [WARNING] Folder ./wheels/ tidak ditemukan! Mencoba instalasi online standar...
    "%VENV_DIR%\Scripts\pip.exe" install -r requirements.txt
)

echo.
echo [3/3] Memeriksa kesiapan Hardware (GPU/CUDA) dan Model Offline...
"%VENV_DIR%\Scripts\python.exe" -c "
import sys, os

print(f'Python Interpreter : {sys.executable} ({sys.version.split()[0]})')

try:
    import torch
    has_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if has_cuda else 'CPU Multi-Core Mode'
    print(f'Akselerator Compute: PyTorch {torch.__version__} | CUDA: {has_cuda} ({dev_name})')
except Exception as e:
    print('Akselerator Compute: Gagal memeriksa PyTorch:', e)

try:
    import transformers
    print(f'Transformers Engine: {transformers.__version__}')
except Exception as e:
    print('Transformers Engine:', e)

models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
if os.path.exists(models_dir):
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    print(f'Model Offline Siap : {len(subdirs)} model -> {subdirs}')
else:
    print('Model Offline      : Folder ./models/ belum ada.')
"

echo.
echo ======================================================================
echo   INSTALASI OFFLINE SELESAI DENGAN SUKSES!
echo   Platform siap dijalankan di PC Panitia.
echo   Silakan jalankan launch_studio.bat untuk memulai web platform.
echo ======================================================================
echo.
pause
