@echo off
setlocal EnableExtensions
title BG3 GR2 Animation Compare

REM Old one-shot launcher. Prefer START_COMPARE_UI.bat (pick files in browser).
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo.
echo Tip: use START_COMPARE_UI.bat to pick any 2 files in the browser.
echo Running default Karlach pair once...
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

%PY% "%SCRIPT_DIR%gr2_anim_compare.py"
set "RC=%ERRORLEVEL%"
if %RC% neq 0 (
    echo FAILED.
    pause
    exit /b %RC%
)

if exist "%SCRIPT_DIR%comparison_report.html" start "" "%SCRIPT_DIR%comparison_report.html"
echo.
pause
exit /b 0
