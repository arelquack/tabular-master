@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Setup Tabular & NLP Studio Environment

echo ============================================================
echo   TABULAR & NLP STUDIO - AUTOMATED ENVIRONMENT SETUP
echo ============================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak terdeteksi pada sistem PATH!
    echo Silakan install Python 3.10+ terlebih dahulu.
    pause
    exit /b 1
)

set "VENV_DIR=%~dp0.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/3] Membuat Python Virtual Environment (.venv)...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] Gagal membuat virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment (.venv) sudah tersedia.
)

echo.
echo [2/3] Menginstal dependencies dari requirements.txt...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\pip.exe" install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Terjadi kendala saat instalasi requirements. Melanjutkan proses...
)

echo.
echo [3/3] Memeriksa dan mengunduh pre-trained NLP models ke folder ./models/ ...
"%VENV_DIR%\Scripts\python.exe" scripts\download_models.py

echo.
echo ============================================================
echo   INSTALASI DAN PRE-CACHING MODEL SELESAI DENGAN SUKSES!
echo   Platform siap digunakan 100%% offline di laptop / PC Dosen.
echo   Jalankan launch_studio.bat untuk memulai web studio.
echo ============================================================
echo.
pause
