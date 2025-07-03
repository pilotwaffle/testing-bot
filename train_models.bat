@echo off
echo Training ML Models...

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found
    pause
    exit /b 1
)

python enhanced_model_trainer.py --full-train --verbose
pause
