@echo off
echo ========================================================
echo Starting Tabular & NLP Studio Nexus via Docker Compose
echo ========================================================

where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop or run locally using:
    echo   spaceship-titanic\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
    pause
    exit /b 1
)

docker compose up --build -d
echo.
echo Nexus Studio is running at: http://localhost:8000
echo Logs can be viewed with: docker compose logs -f
pause
