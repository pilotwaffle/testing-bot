@echo off
REM File: start_bot_fixed.bat
REM Location: E:\Trade Chat Bot\G Trading Bot\start_bot_fixed.bat
REM Description: Elite Trading Bot V3.0 startup script with Unicode fixes
REM Purpose: Start bot with proper UTF-8 encoding support

echo Starting Elite Trading Bot V3.0 with fixes...

:: Set UTF-8 encoding
set PYTHONIOENCODING=utf-8
chcp 65001

:: Navigate to bot directory
cd /d "E:\Trade Chat Bot\G Trading Bot"

:: Start with UTF-8 support
python -X utf8 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
