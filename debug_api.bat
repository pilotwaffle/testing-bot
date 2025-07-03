@echo off
REM File: E:\Trade Chat Bot\G Trading Bot\debug_api.bat
REM Debug API Connection Issues for Elite Trading Bot V3.0

title Elite Trading Bot V3.0 - API Debug
color 0D

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║            Elite Trading Bot V3.0 - API Debugger            ║
echo ║                                                              ║
echo ║  Diagnosing "Failed to fetch" errors                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🔍 Step 1: Checking if bot is running
echo ====================================

REM Check if bot is running on port 8000
netstat -an | findstr :8000 >nul 2>&1
if errorlevel 1 (
    echo ❌ ISSUE FOUND: Bot is NOT running on port 8000
    echo.
    echo 💡 SOLUTION: Start the bot first
    echo 1. Open a new command prompt
    echo 2. Navigate to: %CD%
    echo 3. Run: python main.py
    echo.
    echo Would you like me to try starting it now? (Y/N)
    set /p start_choice=""
    if /i "%start_choice%"=="Y" (
        echo.
        echo 🚀 Starting bot in background...
        start "Elite Trading Bot" python main.py
        echo.
        echo ⏳ Waiting 10 seconds for bot to start...
        timeout /t 10 >nul
        echo.
        echo 🔍 Checking if bot started...
        netstat -an | findstr :8000 >nul 2>&1
        if not errorlevel 1 (
            echo ✅ Bot is now running!
        ) else (
            echo ❌ Bot failed to start. Check the bot window for errors.
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo Please start the bot first, then run this debug script again.
        pause
        exit /b 1
    )
) else (
    echo ✅ Bot is running on port 8000
)

echo.
echo 🔍 Step 2: Testing basic connectivity
echo ===================================

REM Test basic ping endpoint
echo 📡 Testing /ping endpoint...
python -c "
import requests
import sys
try:
    response = requests.get('http://localhost:8000/ping', timeout=5)
    if response.status_code == 200:
        print('✅ /ping - OK')
        print(f'   Response: {response.text}')
    else:
        print(f'❌ /ping - Status: {response.status_code}')
        print(f'   Response: {response.text}')
except requests.exceptions.ConnectionError:
    print('❌ /ping - Connection failed (bot not responding)')
    sys.exit(1)
except requests.exceptions.Timeout:
    print('❌ /ping - Timeout (bot too slow)')
    sys.exit(1)
except Exception as e:
    print(f'❌ /ping - Error: {e}')
    sys.exit(1)
" 2>nul
if errorlevel 1 (
    echo.
    echo ❌ ISSUE FOUND: Basic connectivity failed
    echo.
    echo 💡 SOLUTIONS:
    echo 1. Check if bot started without errors
    echo 2. Look for error messages in the bot console
    echo 3. Check Windows Firewall settings
    echo 4. Try running as Administrator
    pause
    exit /b 1
)

echo.
echo 🔍 Step 3: Testing market data API endpoint
echo ==========================================

echo 📊 Testing /api/market-data endpoint...
python -c "
import requests
import json
try:
    response = requests.get('http://localhost:8000/api/market-data', timeout=10)
    print(f'📈 Status Code: {response.status_code}')
    print(f'📋 Content-Type: {response.headers.get(\"content-type\", \"unknown\")}')
    
    if response.status_code == 200:
        try:
            data = response.json()
            print('✅ /api/market-data - OK (JSON response)')
            print(f'   Success: {data.get(\"success\", \"unknown\")}')
            print(f'   Currency: {data.get(\"currency\", \"unknown\")}')
            symbols = data.get('symbols', {})
            print(f'   Symbols count: {len(symbols)}')
            if symbols:
                first_symbol = list(symbols.keys())[0]
                print(f'   Sample symbol: {first_symbol}')
        except json.JSONDecodeError:
            print('⚠️  /api/market-data - Response is not JSON')
            print(f'   Raw response: {response.text[:200]}...')
    elif response.status_code == 404:
        print('❌ /api/market-data - 404 NOT FOUND')
        print('   The endpoint is not registered in main.py')
    elif response.status_code == 500:
        print('❌ /api/market-data - 500 INTERNAL ERROR')
        print(f'   Error response: {response.text[:200]}...')
    else:
        print(f'❌ /api/market-data - Unexpected status: {response.status_code}')
        print(f'   Response: {response.text[:200]}...')
        
except requests.exceptions.ConnectionError:
    print('❌ /api/market-data - Connection failed')
    print('   The bot is running but not responding to requests')
except requests.exceptions.Timeout:
    print('❌ /api/market-data - Timeout (>10 seconds)')
    print('   The endpoint exists but is very slow')
except Exception as e:
    print(f'❌ /api/market-data - Unexpected error: {e}')
" 2>nul

echo.
echo 🔍 Step 4: Checking main.py for API endpoint
echo ==========================================

echo 📄 Checking if market data endpoint exists in main.py...
findstr /C:"@app.get(\"/api/market-data\")" main.py >nul
if errorlevel 1 (
    echo ❌ ISSUE FOUND: /api/market-data endpoint not found in main.py
    echo.
    echo 💡 SOLUTION: The enhanced main.py code wasn't applied correctly
    echo.
    echo To fix this:
    echo 1. Make sure you copied the enhanced main.py code from Claude
    echo 2. Look for this line in main.py:
    echo    @app.get("/api/market-data")
    echo 3. If missing, you need to apply the enhanced main.py code
    echo.
    echo Would you like me to check what's missing? (Y/N)
    set /p check_choice=""
    if /i "%check_choice%"=="Y" (
        echo.
        echo 🔍 Checking main.py content...
        echo.
        findstr /C:"EnhancedMarketDataManager" main.py >nul
        if errorlevel 1 (
            echo ❌ EnhancedMarketDataManager class is missing
        ) else (
            echo ✅ EnhancedMarketDataManager class found
        )
        
        findstr /C:"market_manager" main.py >nul
        if errorlevel 1 (
            echo ❌ market_manager variable is missing
        ) else (
            echo ✅ market_manager variable found
        )
        
        findstr /C:"get_enhanced_market_data" main.py >nul
        if errorlevel 1 (
            echo ❌ get_enhanced_market_data function is missing
        ) else (
            echo ✅ get_enhanced_market_data function found
        )
    )
) else (
    echo ✅ Market data endpoint found in main.py
)

echo.
echo 🔍 Step 5: Testing endpoint registration
echo ======================================

echo 📋 Getting list of all registered endpoints...
python -c "
import requests
try:
    # Try to get endpoints list
    response = requests.get('http://localhost:8000/api/endpoints', timeout=5)
    if response.status_code == 200:
        data = response.json()
        api_routes = data.get('api_routes', [])
        print('✅ Endpoint listing available')
        print('📋 Registered API endpoints:')
        for route in api_routes:
            path = route.get('path', 'unknown')
            methods = route.get('methods', [])
            print(f'   • {path} {methods}')
        
        # Check if market-data is in the list
        market_data_found = any('/api/market-data' in route.get('path', '') for route in api_routes)
        if market_data_found:
            print('✅ /api/market-data is properly registered')
        else:
            print('❌ /api/market-data is NOT registered')
    else:
        print('⚠️  Cannot get endpoint list (endpoint may not exist)')
        
except Exception as e:
    print(f'⚠️  Cannot check endpoints: {e}')
" 2>nul

echo.
echo 🔍 Step 6: CORS and browser compatibility check
echo ==============================================

echo 🌐 Testing CORS headers...
python -c "
import requests
try:
    response = requests.get('http://localhost:8000/api/market-data', timeout=5)
    cors_headers = {
        'Access-Control-Allow-Origin': response.headers.get('access-control-allow-origin'),
        'Access-Control-Allow-Methods': response.headers.get('access-control-allow-methods'),
        'Access-Control-Allow-Headers': response.headers.get('access-control-allow-headers')
    }
    
    print('🔍 CORS Headers:')
    for header, value in cors_headers.items():
        if value:
            print(f'   ✅ {header}: {value}')
        else:
            print(f'   ❌ {header}: Not set')
    
    if cors_headers['Access-Control-Allow-Origin']:
        print('✅ CORS is configured')
    else:
        print('❌ CORS headers missing - this can cause frontend fetch errors')
        
except Exception as e:
    print(f'❌ Could not check CORS: {e}')
" 2>nul

echo.
echo 🎯 Diagnosis Summary and Solutions
echo ================================

echo.
echo Based on the tests above, here are the most likely solutions:
echo.
echo 🔧 If the bot is not running:
echo    → Start the bot: python main.py
echo.
echo 🔧 If 404 errors on /api/market-data:
echo    → Apply the enhanced main.py code from Claude
echo    → Make sure EnhancedMarketDataManager is included
echo.
echo 🔧 If CORS errors:
echo    → Enhanced main.py includes CORS fixes
echo    → Clear browser cache (Ctrl+F5)
echo.
echo 🔧 If connection errors:
echo    → Check Windows Firewall
echo    → Run as Administrator
echo    → Try different port: set PORT=8001
echo.
echo 🔧 If timeouts/slow responses:
echo    → Check system resources
echo    → Close other Python processes
echo    → Restart the bot
echo.
echo 🌐 Quick test URLs:
echo    • Basic connectivity: http://localhost:8000/ping
echo    • Health check: http://localhost:8000/health  
echo    • Market data: http://localhost:8000/api/market-data
echo    • Dashboard: http://localhost:8000
echo.

echo Would you like me to create a simple test HTML file to verify the fix? (Y/N)
set /p html_choice=""
if /i "%html_choice%"=="Y" (
    echo.
    echo 📝 Creating test HTML file...
    (
    echo ^<!DOCTYPE html^>
    echo ^<html^>
    echo ^<head^>
    echo     ^<title^>API Test^</title^>
    echo ^</head^>
    echo ^<body^>
    echo     ^<h1^>Elite Trading Bot API Test^</h1^>
    echo     ^<button onclick="testAPI()"^>Test Market Data API^</button^>
    echo     ^<div id="result"^>^</div^>
    echo     ^<script^>
    echo     async function testAPI() {
    echo         const resultDiv = document.getElementById('result');
    echo         try {
    echo             const response = await fetch('/api/market-data');
    echo             if (response.ok) {
    echo                 const data = await response.json();
    echo                 resultDiv.innerHTML = '^<pre^>' + JSON.stringify(data, null, 2) + '^</pre^>';
    echo             } else {
    echo                 resultDiv.innerHTML = 'Error: ' + response.status;
    echo             }
    echo         } catch (error) {
    echo             resultDiv.innerHTML = 'Fetch Error: ' + error.message;
    echo         }
    echo     }
    echo     ^</script^>
    echo ^</body^>
    echo ^</html^>
    ) > test_api.html
    echo ✅ Created test_api.html
    echo.
    echo 🌐 Open this file in your browser while the bot is running:
    echo    file://%CD%\test_api.html
    echo.
    echo Or access it through the bot:
    echo    http://localhost:8000/static/test_api.html
    echo    (if you move test_api.html to the static folder)
)

echo.
pause