@echo off
setlocal EnableExtensions
title BG3 Anim Compare UI

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo ========================================
echo   BG3 Animation Compare - Web UI
echo ========================================
echo.
echo  Opens http://127.0.0.1:8765
echo  Pick 2 files in the browser, press Run.
echo  Leave this window open while using it.
echo.

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo ERROR: Python 3 not found.
    pause
    exit /b 1
)

%PY% -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing numpy...
    %PY% -m pip install numpy -q
)

%PY% "%SCRIPT_DIR%compare_server.py"
echo.
pause
