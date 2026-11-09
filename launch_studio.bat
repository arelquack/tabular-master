@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Tabular & NLP Studio Server - Gemastik Final

echo ============================================================
echo   TABULAR & NLP STUDIO - Competitive Machine Learning Platform
echo ============================================================
echo.

set "SCRIPT_DIR=%~dp0"
set "VENV_PY="

rem Set offline HuggingFace cache to local models directory
if exist "%SCRIPT_DIR%models" (
    set "HF_HOME=%SCRIPT_DIR%models"
    set "TRANSFORMERS_OFFLINE=1"
    set "HF_DATASETS_OFFLINE=1"
    echo [OFFLINE READY] HuggingFace cache diatur ke folder lokal: %SCRIPT_DIR%models
)

rem Prioritas pencarian Python:
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%spaceship-titanic\.venv\Scripts\python.exe" (
    set "VENV_PY=%SCRIPT_DIR%spaceship-titanic\.venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "VENV_PY=%SCRIPT_DIR%venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "VENV_PY=python"
    ) else (
        echo [INFO] Environment belum terpasang. Menjalankan install_offline.bat / setup.bat...
        if exist "%SCRIPT_DIR%install_offline.bat" (
            call "%SCRIPT_DIR%install_offline.bat"
        ) else (
            call "%SCRIPT_DIR%setup.bat"
        )
        set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
    )
)

echo Menggunakan Python: !VENV_PY!
echo Memulai FastAPI Engine pada http://localhost:8000 ...
echo Tekan Ctrl+C pada konsol ini untuk menghentikan server.
echo.

rem Otomatis buka browser setelah server siap
start "" "http://localhost:8000"
"!VENV_PY!" -m uvicorn server:app --host 127.0.0.1 --port 8000

pause
