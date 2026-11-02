@echo off
setlocal
cd /d "%~dp0"
title Tabular Mining Studio Server

echo ============================================================
echo   TABULAR STUDIO - Competitive Machine Learning Platform
echo ============================================================
echo.
echo Starting FastAPI Engine on http://localhost:8000 ...
echo Press Ctrl+C in this console to stop the platform.
echo.

start "" "http://localhost:8000"
".\spaceship-titanic\.venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
