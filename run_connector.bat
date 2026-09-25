@echo off
title MT5 Real-Time 1-Second AI Trading Bot
cd /d "%~dp0"

echo ======================================================================
echo          REAL-TIME 1-SECOND QUANTITATIVE MT5 AI TRADING BOT
echo ======================================================================
echo Target MT5 Account : #12345868444 (Elefin-Trade)
echo Active Symbols     : EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, XAUUSD
echo Scan Frequency     : EVERY 1.0 SECOND (Zero-Lag Local RAM Inference)
echo Daily Profit Goal  : $10.00 - $15.00 (Max 5 Trades/Day)
echo ======================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found!
    pause
    exit /b 1
)

echo [ACTIVE] Connecting to MT5 and starting 1-second live decision engine...
echo.

.\.venv\Scripts\python.exe -u mt5_local_connector.py

pause
