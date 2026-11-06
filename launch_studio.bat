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
        echo [ERROR] Python environment tidak ditemukan!
        echo Silakan siapkan virtual environment:
        echo   python -m venv .venv
        echo   .\.venv\Scripts\pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

echo Using Python: %VENV_PY%
echo Starting FastAPI Engine on http://localhost:8000 ...
echo Press Ctrl+C in this console to stop the platform.
echo.

start "" "http://localhost:8000"
"%VENV_PY%" -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
