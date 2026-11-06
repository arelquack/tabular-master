@echo off
setlocal
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
        exit /b 1
    )
)

if "%~1"=="" (
    echo Menjalankan scripts\runners\run_s6e10.py secara default...
    "%VENV_PY%" "%SCRIPT_DIR%scripts\runners\run_s6e10.py"
) else (
    if exist "%SCRIPT_DIR%scripts\runners\%~1" (
        "%VENV_PY%" "%SCRIPT_DIR%scripts\runners\%~1" %2 %3 %4 %5
    ) else if exist "%SCRIPT_DIR%scripts\%~1" (
        "%VENV_PY%" "%SCRIPT_DIR%scripts\%~1" %2 %3 %4 %5
    ) else (
        "%VENV_PY%" "%SCRIPT_DIR%%~1" %2 %3 %4 %5
    )
)
