@echo off
title MT5 AI Bridge Connector
cd /d "%~dp0"

echo =======================================================
echo          STARTING LOCAL MT5 AI CONNECTOR
echo =======================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found!
    echo Please create it first using: py -3.11 -m venv .venv
    pause
    exit /b 1
)

:: Prompt for Render URL or use default
set /p RENDER_URL="Enter Render Server URL [Press Enter for default localhost:8000]: "
if "%RENDER_URL%"=="" set RENDER_URL=http://127.0.0.1:8000

echo.
echo Connecting to AI Cloud at: %RENDER_URL%
echo.

.\.venv\Scripts\python.exe mt5_local_connector.py --url %RENDER_URL% --interval 15

pause
