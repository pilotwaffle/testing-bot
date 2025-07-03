@echo off
echo Starting Enhanced Trading Bot...

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models" mkdir models
dir /b models | findstr /r ".*" >nul
if errorlevel 1 (
    echo WARNING: No trained models found. Training models first...
    python enhanced_model_trainer.py --full-train
)

REM Start the bot
echo Starting trading bot...
python main_trading_bot.py
pause
