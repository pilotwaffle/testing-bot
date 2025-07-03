@echo off
REM File: E:\Trade Chat Bot\G Trading Bot\fix_dashboard_simple.bat
REM Location: E:\Trade Chat Bot\G Trading Bot\fix_dashboard_simple.bat
REM Simple dashboard template fix - calls Python script

title Elite Trading Bot - Dashboard Fix
color 0E

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              Dashboard Template Fix                          ║
echo ║             Elite Trading Bot V3.0                          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🔧 Fixing dashboard template Jinja errors...
echo.

REM Check if Python script exists
if not exist "fix_template.py" (
    echo ❌ fix_template.py not found in current directory
    echo.
    echo Please make sure you have saved the fix_template.py file
    echo in this directory: %CD%
    echo.
    pause
    exit /b 1
)

echo 📄 Running template fix script...
python fix_template.py

echo.
echo ✅ Dashboard template fix complete!
echo.
echo 🚀 Next steps:
echo 1. Restart your bot (Stop with Ctrl+C, then: python main.py)
echo 2. Open: http://localhost:8000
echo 3. Test the Market Data API button
echo 4. Verify no more template errors
echo.
pause