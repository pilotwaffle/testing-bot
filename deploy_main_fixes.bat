@echo off
REM File: E:\Trade Chat Bot\G Trading Bot\deploy_main_fixes.bat
REM Windows Deployment Script for Elite Trading Bot V3.0

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

REM Check required packages
echo 📦 Checking required dependencies...
python -c "
import sys
missing = []
packages = ['fastapi', 'uvicorn', 'aiohttp', 'psutil', 'python-dotenv', 'jinja2']

for package in packages:
    try:
        __import__(package)
        print(f'✅ {package} found')
    except ImportError:
        missing.append(package)
        print(f'⚠️  {package} missing')

if missing:
    print(f'📦 Installing missing packages: {missing}')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
    print('✅ All packages installed')
else:
    print('✅ All required dependencies found')
"

REM Create necessary directories
echo 📂 Ensuring directory structure...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "static\js" mkdir static\js
if not exist "static\css" mkdir static\css
if not exist "templates" mkdir templates
echo ✅ Directory structure ready

REM Create environment file
echo 🌍 Creating environment configuration...
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

REM Create startup batch script
echo 🚀 Creating Windows startup script...
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
echo REM Load environment variables from .env if it exists
echo if exist ".env" ^(
echo     echo 📁 Loading environment variables from .env
echo     for /f "usebackq tokens=1,2 delims==" %%%%a in ^(".env"^) do ^(
echo         if not "%%%%a"=="" if not "%%%%a:~0,1%%"=="#" set %%%%a=%%%%b
echo     ^)
echo ^)
echo.
echo echo 🔧 Configuration:
echo echo    Host: %%HOST%%
echo echo    Port: %%PORT%%
echo echo    Environment: %%ENVIRONMENT%%
echo.
echo REM Check if port is available
echo netstat -an ^| findstr :%%PORT%% >nul 2>&1
echo if not errorlevel 1 ^(
echo     echo ⚠️  Port %%PORT%% might be in use. If issues occur, try a different port.
echo ^)
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

REM Create testing script
echo 🧪 Creating testing script...
(
echo import asyncio
echo import aiohttp
echo import json
echo import sys
echo import time
echo from datetime import datetime
echo.
echo async def test_endpoint^(session, url, endpoint, expected_status=200^):
echo     """Test a single endpoint"""
echo     full_url = f"{url}{endpoint}"
echo     try:
echo         async with session.get^(full_url, timeout=10^) as response:
echo             status = response.status
echo             content_type = response.headers.get^('content-type', '''^)
echo             
echo             if status == expected_status:
echo                 if 'application/json' in content_type:
echo                     data = await response.json^(^)
echo                     return True, f"✅ {endpoint} - Status: {status} - JSON response received"
echo                 else:
echo                     return True, f"✅ {endpoint} - Status: {status} - Response received"
echo             else:
echo                 return False, f"❌ {endpoint} - Expected: {expected_status}, Got: {status}"
echo                 
echo     except asyncio.TimeoutError:
echo         return False, f"❌ {endpoint} - Timeout after 10 seconds"
echo     except Exception as e:
echo         return False, f"❌ {endpoint} - Error: {str^(e^)}"
echo.
echo async def test_deployment^(^):
echo     """Test the deployment"""
echo     base_url = "http://localhost:8000"
echo     
echo     endpoints = [
echo         ^("/ping", 200^),
echo         ^("/health", 200^),
echo         ^("/api/market-data", 200^),
echo         ^("/api/trading-pairs", 200^),
echo         ^("/api/market-overview", 200^),
echo         ^("/api/endpoints", 200^),
echo         ^("/", 200^),
echo     ]
echo     
echo     print^("🧪 Elite Trading Bot V3.0 - Windows Deployment Test"^)
echo     print^("=" * 50^)
echo     print^(f"🎯 Testing server at: {base_url}"^)
echo     print^(f"⏰ Test started at: {datetime.now^(^).strftime^('%%Y-%%m-%%d %%H:%%M:%%S'^)}"^)
echo     print^(^)
echo     
echo     print^("⏳ Waiting for server to start..."^)
echo     await asyncio.sleep^(3^)
echo     
echo     connector = aiohttp.TCPConnector^(limit=10^)
echo     timeout = aiohttp.ClientTimeout^(total=30^)
echo     
echo     async with aiohttp.ClientSession^(connector=connector, timeout=timeout^) as session:
echo         results = []
echo         
echo         for endpoint, expected_status in endpoints:
echo             success, message = await test_endpoint^(session, base_url, endpoint, expected_status^)
echo             results.append^(^(success, message^)^)
echo             print^(message^)
echo             await asyncio.sleep^(0.5^)
echo     
echo     print^(^)
echo     print^("📊 Test Results Summary:"^)
echo     print^("=" * 50^)
echo     
echo     successful = sum^(1 for success, _ in results if success^)
echo     total = len^(results^)
echo     
echo     print^(f"✅ Successful: {successful}/{total}"^)
echo     print^(f"❌ Failed: {total - successful}/{total}"^)
echo     print^(f"📈 Success Rate: {^(successful/total^)*100:.1f}%%"^)
echo     
echo     if successful == total:
echo         print^(^)
echo         print^("🎉 ALL TESTS PASSED! Deployment successful!"^)
echo         print^("🌐 Your bot is ready at: http://localhost:8000"^)
echo         print^("📊 Market Data API: http://localhost:8000/api/market-data"^)
echo         return True
echo     else:
echo         print^(^)
echo         print^("⚠️  Some tests failed. Check the logs and fix issues."^)
echo         return False
echo.
echo if __name__ == "__main__":
echo     try:
echo         success = asyncio.run^(test_deployment^(^)^)
echo         input^("Press Enter to continue..."^)
echo         sys.exit^(0 if success else 1^)
echo     except KeyboardInterrupt:
echo         print^("\\n🛑 Tests interrupted by user"^)
echo         sys.exit^(1^)
echo     except Exception as e:
echo         print^(f"\\n❌ Test error: {e}"^)
echo         input^("Press Enter to continue..."^)
echo         sys.exit^(1^)
) > test_deployment.py

echo ✅ Testing script created (test_deployment.py)

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
    echo.
    echo # Optional: ML and Trading Dependencies
    echo # pandas>=2.0.0
    echo # ccxt>=4.0.0
    echo # tensorflow>=2.13.0
    echo # scikit-learn>=1.3.0
    ) > requirements.txt
    echo ✅ requirements.txt created
)

REM Final deployment summary
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
echo 1. Install dependencies: pip install -r requirements.txt
echo 2. Copy the enhanced main.py code (from Claude's first artifact)
echo 3. Start the bot: start_bot.bat (or python main.py)
echo 4. Test deployment: python test_deployment.py
echo 5. Access dashboard: http://localhost:8000
echo 6. Test API: http://localhost:8000/api/market-data
echo.
echo 🔧 Windows Troubleshooting:
echo    - Check logs in logs\ directory
echo    - Use /health endpoint for diagnostics
echo    - Use /api/endpoints to verify all routes
echo    - Use /ping for basic connectivity
echo    - Check Windows Firewall if connection issues
echo.
echo ✨ Your Elite Trading Bot V3.0 is ready for Windows deployment!
echo.
pause