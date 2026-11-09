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
echo  Salin SELURUH folder ini ke Flashdisk
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
"%VENV_DIR%\Scripts\python.exe" scripts\verify_bundle.py

echo.
echo Selesai. Tekan tombol apa saja untuk menutup jendela ini...
pause
