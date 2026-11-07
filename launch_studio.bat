@echo off
setlocal
cd /d "%~dp0"
title Tabular & NLP Studio Server

echo ============================================================
echo   TABULAR & NLP STUDIO - Competitive Machine Learning Platform
echo ============================================================
echo.

set SCRIPT_DIR=%~dp0
set VENV_PY=

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%spaceship-titanic\.venv\Scripts\python.exe" (
    set "VENV_PY=%SCRIPT_DIR%spaceship-titanic\.venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "VENV_PY=%SCRIPT_DIR%venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "VENV_PY=python"
    ) else (
        echo [INFO] Environment belum terpasang. Menjalankan setup.bat otomatis...
        call "%SCRIPT_DIR%setup.bat"
        set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
    )
)

echo Menggunakan Python: %VENV_PY%
echo Memeriksa kesiapan FastApi Backend pada http://localhost:8000 ...
echo Tekan Ctrl+C pada konsol ini untuk menghentikan server.
echo.

start "" "http://localhost:8000"
"%VENV_PY%" -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
