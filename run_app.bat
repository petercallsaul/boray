@echo off
title SignAI Workstation Launcher
cd /d "%~dp0\SignAI"
echo Starting SignAI Modern Workstation...
..\.venv\Scripts\python.exe app_modern.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
