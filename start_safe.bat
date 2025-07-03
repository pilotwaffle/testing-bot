@echo off
REM File: start_safe.bat
REM Location: E:\Trade Chat Bot\G Trading Bot\start_safe.bat
REM Description: Elite Trading Bot V3.0 - Safe Startup Script
REM Purpose: Start bot with proper Unicode and encoding support

echo Elite Trading Bot V3.0 - Safe Startup
echo =====================================

:: Set UTF-8 encoding for Windows
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=0
chcp 65001 >nul

:: Navigate to bot directory
cd /d "E:\Trade Chat Bot\G Trading Bot"

echo Starting server with Unicode support...
echo Server will be available at: http://localhost:8000

:: Start server with UTF-8 support and reduced logging
python -X utf8 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level warning

echo.
echo Server stopped. Press any key to exit...
pause >nul
