@echo off
REM Trading Bot Dashboard Startup Script
REM FILE LOCATION: E:\Trade Chat Bot\G Trading Bot\start_trading_system.bat

title Trading Bot Dashboard System

echo.
echo ========================================
echo    🚀 Trading Bot Dashboard System
echo ========================================
echo.
echo 📁 Location: %CD%
echo 🌐 Dashboard: http://localhost:5000
echo 👤 Login: admin / trading123
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if required packages are installed
echo 🔍 Checking packages...
python -c "import flask, pandas, numpy, plotly" 2>nul
if errorlevel 1 (
    echo 📦 Installing required packages...
    pip install flask flask-login plotly pandas numpy
    if errorlevel 1 (
        echo ❌ Failed to install packages!
        pause
        exit /b 1
    )
)

REM Check if required files exist
if not exist "optimized_model_trainer.py" (
    echo ❌ optimized_model_trainer.py not found!
    echo This should be your enhanced trading bot file.
    pause
    exit /b 1
)

if not exist "dashboard_app.py" (
    echo ❌ dashboard_app.py not found!
    echo Please save the dashboard code as this file.
    pause
    exit /b 1
)

if not exist "run_integrated_system.py" (
    echo ❌ run_integrated_system.py not found!
    echo Please save the integration code as this file.
    pause
    exit /b 1
)

echo ✅ All files found!
echo.
echo 🚀 Starting Trading Bot Dashboard...
echo.
echo Dashboard will open automatically in your browser.
echo Use Ctrl+C to stop the system.
echo.

REM Start the integrated system
python run_integrated_system.py

echo.
echo 🛑 Trading Bot Dashboard stopped.
pause