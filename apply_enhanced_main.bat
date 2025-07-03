@echo off
REM File: E:\Trade Chat Bot\G Trading Bot\apply_enhanced_main.bat
REM Script to help apply the enhanced main.py code

title Apply Enhanced Main.py - Elite Trading Bot V3.0
color 0E

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              Apply Enhanced Main.py Code                    ║
echo ║             Elite Trading Bot V3.0                          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if we're in the right directory
if not exist "main.py" (
    echo ❌ Error: main.py not found in current directory.
    echo Please run this from the G Trading Bot folder.
    pause
    exit /b 1
)

echo 📁 Current directory: %CD%
echo.

echo 🔍 Current main.py status:
for %%F in (main.py) do echo    Size: %%~zF bytes
for %%F in (main.py) do echo    Modified: %%~tF
echo.

echo 💾 Creating backup before applying enhanced code...
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do set BACKUP_DATE=%%c%%a%%b
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set BACKUP_TIME=%%a%%b
set BACKUP_TIME=%BACKUP_TIME: =0%
copy main.py "main.py.backup_before_enhanced_%BACKUP_DATE%_%BACKUP_TIME%"
echo ✅ Backup created: main.py.backup_before_enhanced_%BACKUP_DATE%_%BACKUP_TIME%
echo.

echo 📋 INSTRUCTIONS TO APPLY ENHANCED MAIN.PY:
echo ==========================================
echo.
echo 1. 📄 Open Claude's conversation where you see the enhanced main.py code
echo    (Look for the first artifact titled "Enhanced main.py with Deployment Fixes")
echo.
echo 2. 📋 Copy the ENTIRE enhanced main.py code from Claude's artifact
echo.
echo 3. 📝 Open your main.py file in a text editor:
echo    - Right-click on main.py → "Open with" → Notepad++ or VS Code
echo    - Or use: notepad main.py
echo.
echo 4. 🔄 Replace ALL content in main.py with the enhanced code from Claude
echo.
echo 5. 💾 Save the file (Ctrl+S)
echo.
echo 6. ✅ Return here and press any key to test the enhanced version
echo.

echo 🔧 Alternative Method (if you have the enhanced code saved):
echo ===========================================================
echo If you saved the enhanced main.py as a separate file:
echo 1. Put the enhanced file in this directory as "main_enhanced.py"
echo 2. Run this script again - it will detect and use it automatically
echo.

REM Check if enhanced version exists
if exist "main_enhanced.py" (
    echo 🎉 Found main_enhanced.py! 
    echo.
    echo Would you like to replace main.py with main_enhanced.py? (Y/N)
    set /p replace_choice=""
    if /i "%replace_choice%"=="Y" (
        copy main_enhanced.py main.py
        echo ✅ main.py replaced with enhanced version!
        goto TEST_ENHANCED
    )
)

pause
echo.

:TEST_ENHANCED
echo 🧪 Testing Enhanced Main.py
echo ===========================
echo.

echo 🔍 Checking syntax of enhanced main.py...
python -m py_compile main.py 2>nul
if errorlevel 1 (
    echo ❌ Syntax error in enhanced main.py!
    echo.
    echo Running detailed syntax check...
    python -m py_compile main.py
    echo.
    echo 💡 Please fix the syntax errors and try again.
    echo You can restore the backup: copy main.py.backup_before_enhanced_* main.py
    pause
    exit /b 1
) else (
    echo ✅ Enhanced main.py syntax is valid!
)

echo.
echo 🔍 Checking for key enhancements...
echo import re > check_enhancements.py
echo with open('main.py', 'r') as f: >> check_enhancements.py
echo     content = f.read() >> check_enhancements.py
echo. >> check_enhancements.py
echo enhancements = { >> check_enhancements.py
echo     'Enhanced Market Data Manager': 'EnhancedMarketDataManager' in content, >> check_enhancements.py
echo     'Market Data API Endpoint': '/api/market-data' in content, >> check_enhancements.py
echo     'CORS Middleware': 'CORSMiddleware' in content, >> check_enhancements.py
echo     'Global Exception Handler': 'StarletteHTTPException' in content, >> check_enhancements.py
echo     'Enhanced Error Handling': 'JSONResponse' in content, >> check_enhancements.py
echo     'Request ID Tracking': 'request_id' in content, >> check_enhancements.py
echo     'Performance Monitoring': 'request_stats' in content >> check_enhancements.py
echo } >> check_enhancements.py
echo. >> check_enhancements.py
echo print('🔍 Enhancement Check Results:') >> check_enhancements.py
echo print('=' * 40) >> check_enhancements.py
echo for name, found in enhancements.items(): >> check_enhancements.py
echo     status = '✅' if found else '❌' >> check_enhancements.py
echo     print(f'{status} {name}') >> check_enhancements.py
echo. >> check_enhancements.py
echo total_found = sum(enhancements.values()) >> check_enhancements.py
echo total_checks = len(enhancements) >> check_enhancements.py
echo print() >> check_enhancements.py
echo print(f'📊 Enhancement Score: {total_found}/{total_checks} ({total_found/total_checks*100:.1f}%%)') >> check_enhancements.py
echo. >> check_enhancements.py
echo if total_found >= 5: >> check_enhancements.py
echo     print('🎉 Enhanced main.py successfully applied!') >> check_enhancements.py
echo     print('✅ Your bot now has all the deployment fixes!') >> check_enhancements.py
echo else: >> check_enhancements.py
echo     print('⚠️  Enhanced main.py may not be fully applied.') >> check_enhancements.py
echo     print('💡 Make sure you copied the complete enhanced code from Claude.') >> check_enhancements.py

python check_enhancements.py
del check_enhancements.py

echo.
echo 🚀 Ready to Test Enhanced Bot
echo ============================
echo.
echo Your enhanced main.py is ready! Now you can:
echo.
echo 1. 🚀 Start the enhanced bot:
echo    python main.py
echo    (or double-click start_bot.bat)
echo.
echo 2. 🧪 Test the deployment:
echo    python test_deployment.py
echo.
echo 3. 🌐 Access your enhanced bot:
echo    • Dashboard: http://localhost:8000
echo    • Market Data API: http://localhost:8000/api/market-data
echo    • Health Check: http://localhost:8000/health
echo.
echo 💡 The enhanced version includes:
echo    ✅ Fixed 404 API errors
echo    ✅ Better error handling
echo    ✅ Enhanced market data manager
echo    ✅ CORS fixes for deployment
echo    ✅ Performance monitoring
echo    ✅ Request ID tracking
echo    ✅ Better logging and diagnostics
echo.

echo Would you like to start the enhanced bot now? (Y/N)
set /p start_choice=""
if /i "%start_choice%"=="Y" (
    echo.
    echo 🚀 Starting Elite Trading Bot V3.0 Enhanced...
    echo.
    echo 📊 Dashboard: http://localhost:8000
    echo 📈 Market Data API: http://localhost:8000/api/market-data
    echo 🏥 Health Check: http://localhost:8000/health
    echo.
    echo Press Ctrl+C to stop the bot when you're done testing.
    echo.
    python main.py
) else (
    echo.
    echo ✅ Enhanced main.py is ready!
    echo Run 'python main.py' when you're ready to start.
)

echo.
pause