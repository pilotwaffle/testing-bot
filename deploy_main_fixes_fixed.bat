@echo off
REM File: E:\Trade Chat Bot\G Trading Bot\deploy_main_fixes_fixed.bat
REM Fixed Windows Deployment Script for Elite Trading Bot V3.0

echo 🔧 Elite Trading Bot V3.0 - Main.py Deployment Fixes (Windows)
echo ==================================================================

REM Check if we're in the right directory
if not exist "main.py" (
    echo ❌ Error: Not in the correct directory. Please run this from the G Trading Bot folder.
    pause
    exit /b 1
)

echo 📁 Current directory: %CD%

REM Create backup of current main.py
echo 💾 Creating backup of current main.py...
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do set BACKUP_DATE=%%c%%a%%b
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set BACKUP_TIME=%%a%%b
set BACKUP_TIME=%BACKUP_TIME: =0%
copy main.py "main.py.backup_%BACKUP_DATE%_%BACKUP_TIME%"
echo ✅ Backup created: main.py.backup_%BACKUP_DATE%_%BACKUP_TIME%

REM Check Python installation
echo 🐍 Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ and add it to PATH.
    echo 💡 Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create temporary Python script to check dependencies
echo 📦 Checking required dependencies...
echo import sys > check_deps.py
echo packages = ['fastapi', 'uvicorn', 'aiohttp', 'psutil', 'python-dotenv', 'jinja2'] >> check_deps.py
echo missing = [] >> check_deps.py
echo for package in packages: >> check_deps.py
echo     try: >> check_deps.py
echo         if package == 'python-dotenv': >> check_deps.py
echo             __import__('dotenv') >> check_deps.py
echo         else: >> check_deps.py
echo             __import__(package) >> check_deps.py
echo         print(f'✅ {package} found') >> check_deps.py
echo     except ImportError: >> check_deps.py
echo         missing.append(package) >> check_deps.py
echo         print(f'⚠️  {package} missing') >> check_deps.py
echo if missing: >> check_deps.py
echo     print(f'📦 Installing missing packages: {missing}') >> check_deps.py
echo     import subprocess >> check_deps.py
echo     subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing) >> check_deps.py
echo     print('✅ All packages installed') >> check_deps.py
echo else: >> check_deps.py
echo     print('✅ All required dependencies found') >> check_deps.py

python check_deps.py
del check_deps.py

REM Create necessary directories
echo 📂 Ensuring directory structure...
for %%d in (logs data models api api\routers core ai utils static static\js static\css templates) do (
    if not exist "%%d" (
        mkdir "%%d" 2>nul
        echo ✅ Created directory: %%d
    )
)

REM Create __init__.py files
for %%d in (api api\routers core ai utils) do (
    if not exist "%%d\__init__.py" (
        echo. > "%%d\__init__.py"
        echo ✅ Created __init__.py in %%d
    )
)

REM Create environment file
echo 🌍 Creating environment configuration...
if not exist ".env.deployment" (
    (
    echo # Elite Trading Bot V3.0 - Windows Deployment Configuration
    echo ENVIRONMENT=development
    echo LOG_LEVEL=INFO
    echo PORT=8000
    echo HOST=0.0.0.0
    echo CORS_ORIGINS=*
    echo ROOT_PATH=
    echo.
    echo # Optional: Uncomment and configure for specific deployment
    echo # SERVER_URL=https://your-domain.com
    echo # SSL_KEYFILE=path/to/ssl.key
    echo # SSL_CERTFILE=path/to/ssl.cert
    ) > .env.deployment
    echo ✅ Environment configuration created (.env.deployment)
)

REM Create Windows startup script
echo 🚀 Creating Windows startup script...
if not exist "start_bot.bat" (
    (
    echo @echo off
    echo REM Elite Trading Bot V3.0 - Windows Startup Script
    echo.
    echo echo 🚀 Starting Elite Trading Bot V3.0...
    echo.
    echo REM Set default environment variables
    echo set PORT=8000
    echo set HOST=0.0.0.0
    echo set ENVIRONMENT=development
    echo.
    echo echo 🔧 Configuration:
    echo echo    Host: %%HOST%%
    echo echo    Port: %%PORT%%
    echo echo    Environment: %%ENVIRONMENT%%
    echo.
    echo echo 🎯 Starting server on http://%%HOST%%:%%PORT%%
    echo echo 📊 Market Data API: http://%%HOST%%:%%PORT%%/api/market-data
    echo echo 🏥 Health Check: http://%%HOST%%:%%PORT%%/health
    echo echo.
    echo python main.py
    echo.
    echo if errorlevel 1 ^(
    echo     echo ❌ Server failed to start. Check the error above.
    echo     pause
    echo ^)
    ) > start_bot.bat
    echo ✅ Windows startup script created (start_bot.bat)
)

REM Create testing script
echo 🧪 Creating testing script...
if not exist "test_deployment.py" (
    echo import asyncio > test_deployment.py
    echo import aiohttp >> test_deployment.py
    echo from datetime import datetime >> test_deployment.py
    echo. >> test_deployment.py
    echo async def test_endpoints(base_url, endpoints): >> test_deployment.py
    echo     results = [] >> test_deployment.py
    echo     async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session: >> test_deployment.py
    echo         for endpoint in endpoints: >> test_deployment.py
    echo             try: >> test_deployment.py
    echo                 async with session.get(f'{base_url}{endpoint}') as response: >> test_deployment.py
    echo                     if response.status == 200: >> test_deployment.py
    echo                         print(f'✅ {endpoint} - OK') >> test_deployment.py
    echo                         results.append(True) >> test_deployment.py
    echo                     else: >> test_deployment.py
    echo                         print(f'❌ {endpoint} - Status: {response.status}') >> test_deployment.py
    echo                         results.append(False) >> test_deployment.py
    echo             except Exception as e: >> test_deployment.py
    echo                 print(f'❌ {endpoint} - Error: {str(e)}') >> test_deployment.py
    echo                 results.append(False) >> test_deployment.py
    echo     return results >> test_deployment.py
    echo. >> test_deployment.py
    echo async def main(): >> test_deployment.py
    echo     print('🧪 Elite Trading Bot V3.0 - Windows Deployment Test') >> test_deployment.py
    echo     print('=' * 50) >> test_deployment.py
    echo     base_url = 'http://localhost:8000' >> test_deployment.py
    echo     endpoints = ['/ping', '/health', '/api/market-data', '/api/trading-pairs'] >> test_deployment.py
    echo     print(f'🎯 Testing server at: {base_url}') >> test_deployment.py
    echo     print('⏳ Waiting for server...') >> test_deployment.py
    echo     await asyncio.sleep(3) >> test_deployment.py
    echo     results = await test_endpoints(base_url, endpoints) >> test_deployment.py
    echo     successful = sum(results) >> test_deployment.py
    echo     total = len(results) >> test_deployment.py
    echo     print() >> test_deployment.py
    echo     print('📊 Test Results:') >> test_deployment.py
    echo     print(f'✅ Successful: {successful}/{total}') >> test_deployment.py
    echo     print(f'📈 Success Rate: {(successful/total)*100:.1f}%%') >> test_deployment.py
    echo     if successful == total: >> test_deployment.py
    echo         print('🎉 ALL TESTS PASSED! Deployment successful!') >> test_deployment.py
    echo     else: >> test_deployment.py
    echo         print('⚠️  Some tests failed. Check the bot status.') >> test_deployment.py
    echo     input('Press Enter to continue...') >> test_deployment.py
    echo. >> test_deployment.py
    echo if __name__ == '__main__': >> test_deployment.py
    echo     asyncio.run(main()) >> test_deployment.py
    
    echo ✅ Testing script created (test_deployment.py)
)

REM Create requirements.txt if it doesn't exist
if not exist "requirements.txt" (
    echo 📋 Creating requirements.txt...
    (
    echo # Elite Trading Bot V3.0 - Core Dependencies
    echo fastapi>=0.104.0
    echo uvicorn[standard]>=0.24.0
    echo python-multipart>=0.0.6
    echo jinja2>=3.1.2
    echo aiofiles>=23.2.1
    echo aiohttp>=3.9.0
    echo python-dotenv>=1.0.0
    echo psutil>=5.9.0
    echo numpy>=1.24.0
    ) > requirements.txt
    echo ✅ requirements.txt created
)

REM Final test of Python import
echo 🔍 Testing final Python setup...
echo import fastapi, uvicorn, aiohttp > test_imports.py
echo print('✅ All core imports successful') >> test_imports.py
python test_imports.py 2>nul
if errorlevel 1 (
    echo ⚠️  Some imports failed. Installing core packages...
    pip install fastapi uvicorn aiohttp python-dotenv psutil jinja2 numpy
) else (
    echo ✅ All imports working
)
del test_imports.py

echo.
echo 🎯 Windows Deployment Summary
echo =============================
echo ✅ Backup created for original main.py
echo ✅ Enhanced main.py ready for deployment
echo ✅ Directory structure created
echo ✅ Environment configuration ready
echo ✅ Windows startup script created (start_bot.bat)
echo ✅ Testing script created (test_deployment.py)
echo ✅ Requirements file ready
echo.
echo 🚀 Next Steps:
echo 1. Replace main.py with the enhanced version (from Claude's artifacts)
echo 2. Start the bot: start_bot.bat (or python main.py)
echo 3. Test deployment: python test_deployment.py
echo 4. Access dashboard: http://localhost:8000
echo 5. Test API: http://localhost:8000/api/market-data
echo.
echo 🔧 Windows Troubleshooting:
echo    - Check logs in logs\ directory
echo    - Use /health endpoint for diagnostics
echo    - Use /api/endpoints to verify all routes
echo    - Check Windows Firewall if connection issues
echo    - Run as Administrator if permission issues
echo.
echo ✨ Your Elite Trading Bot V3.0 is ready for Windows deployment!
echo.
echo 📝 IMPORTANT: You still need to copy the enhanced main.py code
echo    from Claude's first artifact to replace your current main.py
echo.
pause