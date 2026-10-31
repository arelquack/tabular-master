@echo off
set SCRIPT_DIR=%~dp0
set VENV_PY=%SCRIPT_DIR%spaceship-titanic\.venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    echo Error: Python environment not found at %VENV_PY%
    exit /b 1
)

if "%~1"=="" (
    echo Menjalankan run_s6e10.py secara default...
    "%VENV_PY%" "%SCRIPT_DIR%run_s6e10.py"
) else (
    "%VENV_PY%" "%SCRIPT_DIR%%~1" %2 %3 %4 %5
)
