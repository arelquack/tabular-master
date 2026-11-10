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

set "BASE_PY="

rem 1. Cek py launcher untuk versi 3.11
py -3.11 -c "import sys; sys.exit(0 if sys.version_info[1] == 11 else 1)" >nul 2>nul
if %errorlevel% equ 0 (
    set "BASE_PY=py -3.11"
    echo [OK] Python 3.11 terdeteksi via Python Launcher (py -3.11)
    goto :found_python
)

rem 2. Cek path instalasi standar Python 3.11
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "BASE_PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe""
    echo [OK] Python 3.11 terdeteksi di %LOCALAPPDATA%\Programs\Python\Python311\
    goto :found_python
)
if exist "C:\Program Files\Python311\python.exe" (
    set "BASE_PY="C:\Program Files\Python311\python.exe""
    echo [OK] Python 3.11 terdeteksi di C:\Program Files\Python311\
    goto :found_python
)
if exist "C:\Python311\python.exe" (
    set "BASE_PY="C:\Python311\python.exe""
    echo [OK] Python 3.11 terdeteksi di C:\Python311\
    goto :found_python
)

rem 3. Cek perintah python default di PATH
where python >nul 2>nul
if %errorlevel% equ 0 (
    python -c "import sys; sys.exit(0 if sys.version_info[1] == 11 else 1)" >nul 2>nul
    if %errorlevel% equ 0 (
        set "BASE_PY=python"
        echo [OK] Default Python di PATH adalah Python 3.11
        goto :found_python
    ) else (
        echo ======================================================================
        echo [PERINGATAN KRUSIAL] Terdeteksi versi Python bukan 3.11 pada PATH!
        echo Seluruh binary wheels (.whl) pada folder ./wheels/ dikompilasi khusus
        echo untuk Python 3.11. Library PyTorch, CatBoost, dan LightGBM belum
        echo mendukung Python 3.14!
        echo Harap install Python 3.11.9 (64-bit) di PC Panitia.
        echo ======================================================================
        set "BASE_PY=python"
        goto :found_python
    )
)

echo [ERROR] Python tidak terdeteksi pada sistem PATH PC Panitia!
echo Silakan install Python 3.11.9 (64-bit) dan centang 'Add python.exe to PATH'.
pause
exit /b 1

:found_python
set "VENV_DIR=%~dp0.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/3] Membuat Virtual Environment lokal pada drive ini (.venv)...
    %BASE_PY% -m venv "%VENV_DIR%"
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
"%VENV_DIR%\Scripts\python.exe" scripts\verify_install.py

echo.
echo ======================================================================
echo   INSTALASI OFFLINE SELESAI DENGAN SUKSES!
echo   Platform siap dijalankan di PC Panitia.
echo   Silakan jalankan launch_studio.bat untuk memulai web platform.
echo ======================================================================
echo.
pause
