@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Package Full Offline Bundle for Gemastik Final

echo ======================================================================
echo   PACKAGE FULL OFFLINE BUNDLE (RUN ON PC DOSEN WITH FAST INTERNET)
echo ======================================================================
echo.
echo Langkah ini akan menyiapkan seluruh kebutuhan offline proyek ini:
echo  1. Menginstal & memverifikasi environment Python (.venv)
echo  2. Mengunduh semua file wheel (.whl) ke ./wheels/ untuk instalasi offline
echo  3. Mengunduh bobot model Transformer (IndoBERT dll) ke ./models/
echo.
echo Setelah skrip ini selesai:
echo  Salin SELURUH folder ini ke Flashdisk (kecuali .venv)
echo  Maka platform siap dijalankan 100%% OFFLINE di PC Panitia!
echo ======================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak terdeteksi pada sistem PATH!
    echo Silakan install Python 3.10+ pada PC Dosen terlebih dahulu.
    pause
    exit /b 1
)

set "VENV_DIR=%~dp0.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/4] Membuat Virtual Environment (.venv)...
    python -m venv "%VENV_DIR%"
) else (
    echo [1/4] Virtual Environment (.venv) sudah tersedia.
)

echo.
echo [2/4] Menginstal & memperbarui dependencies dasar...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\pip.exe" install -r requirements.txt

echo.
echo [3/4] Mengunduh seluruh Python Wheels (.whl) ke folder ./wheels/ ...
if not exist "wheels" mkdir "wheels"
"%VENV_DIR%\Scripts\pip.exe" download -r requirements.txt -d "./wheels"

echo.
echo [4/4] Mengunduh bobot model Pretrained Transformer ke ./models/ ...
"%VENV_DIR%\Scripts\python.exe" scripts\download_models.py

echo.
echo ======================================================================
echo   MEMERIKSA KELENGKAPAN OFFLINE BUNDLE...
echo ======================================================================
"%VENV_DIR%\Scripts\python.exe" -c "
import os, glob
wheels = glob.glob('wheels/*.whl')
models = [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))] if os.path.exists('models') else []
print(f'Total Binary Wheels (.whl) Terunduh : {len(wheels)}')
print(f'Total Model Offline Terunduh        : {len(models)} -> {models}')
if len(wheels) > 0 and len(models) >= 3:
    print('\n>>> STATUS: 100% SUKSES! BUNDLE SIAP DISALIN KE FLASHDISK! <<<')
    print('Petunjuk ke PC Panitia:')
    print('  1. Salin seluruh isi folder ini ke Flashdisk (folder .venv opsional ditinggal).')
    print('  2. Di PC Panitia, jalankan install_offline.bat')
    print('  3. Jalankan launch_studio.bat untuk membuka platform.')
else:
    print('\n[PERINGATAN] Ada file yang belum lengkap. Silakan jalankan ulang script ini.')
"
echo ======================================================================
echo.
pause
