@echo off
echo 🚀 Elite Trading Bot V3.0 - Industrial Dashboard
echo =====================================================

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo Starting Elite Trading Bot V3.0...
python main.py

pause
