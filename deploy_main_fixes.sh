#!/bin/bash
# File: E:\Trade Chat Bot\G Trading Bot\deploy_main_fixes.sh
# Deploy Main.py Fixes Script for Elite Trading Bot V3.0

echo "🔧 Elite Trading Bot V3.0 - Main.py Deployment Fixes"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Not in the correct directory. Please run this from the G Trading Bot folder."
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Create backup of current main.py
echo "💾 Creating backup of current main.py..."
cp main.py "main.py.backup_$(date +%Y%m%d_%H%M%S)"
echo "✅ Backup created: main.py.backup_$(date +%Y%m%d_%H%M%S)"

# Check Python installation and dependencies
echo "🐍 Checking Python environment..."
python --version || {
    echo "❌ Python not found. Please install Python 3.8+ and try again."
    exit 1
}

# Check required packages
echo "📦 Checking required dependencies..."
python -c "
import sys
missing = []
try:
    import fastapi
    print('✅ FastAPI found')
except ImportError:
    missing.append('fastapi')

try:
    import uvicorn
    print('✅ Uvicorn found')
except ImportError:
    missing.append('uvicorn')

try:
    import aiohttp
    print('✅ aiohttp found')
except ImportError:
    missing.append('aiohttp')

try:
    import psutil
    print('✅ psutil found')
except ImportError:
    missing.append('psutil')

if missing:
    print(f'⚠️  Missing packages: {missing}')
    print('📦 Installing missing packages...')
    import subprocess
    for package in missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', package])
else:
    print('✅ All required dependencies found')
"

# Create necessary directories
echo "📂 Ensuring directory structure..."
mkdir -p logs
mkdir -p data
mkdir -p models
mkdir -p static/js
mkdir -p static/css
mkdir -p templates
echo "✅ Directory structure ready"

# Check if enhanced main.py exists
if [ -f "main_enhanced.py" ]; then
    echo "🔄 Found enhanced main.py, backing up current and applying fixes..."
    cp main_enhanced.py main.py
    echo "✅ Enhanced main.py applied"
else
    echo "⚠️  Enhanced main.py not found. Please save the fixed version as main.py"
fi

# Set environment variables for better deployment
echo "🌍 Setting up environment variables..."
cat > .env.deployment << 'EOF'
# Elite Trading Bot V3.0 - Deployment Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=*
ROOT_PATH=

# Optional: Uncomment and configure for specific deployment
# SERVER_URL=https://your-domain.com
# SSL_KEYFILE=path/to/ssl.key
# SSL_CERTFILE=path/to/ssl.cert
EOF

echo "✅ Environment configuration created (.env.deployment)"

# Create a startup script
echo "🚀 Creating startup script..."
cat > start_bot.sh << 'EOF'
#!/bin/bash
# Elite Trading Bot V3.0 - Startup Script

echo "🚀 Starting Elite Trading Bot V3.0..."

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo "📁 Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default values if not provided
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}
export ENVIRONMENT=${ENVIRONMENT:-development}

echo "🔧 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT" 
echo "   Environment: $ENVIRONMENT"

# Check if port is available
if netstat -tuln | grep -q ":$PORT "; then
    echo "⚠️  Port $PORT is already in use. Trying port $((PORT + 1))..."
    export PORT=$((PORT + 1))
fi

# Start the bot
echo "🎯 Starting server on http://$HOST:$PORT"
python main.py
EOF

chmod +x start_bot.sh
echo "✅ Startup script created (start_bot.sh)"

# Create a testing script
echo "🧪 Creating testing script..."
cat > test_deployment.py << 'EOF'
#!/usr/bin/env python3
"""
Elite Trading Bot V3.0 - Deployment Test Script
Tests all critical endpoints to verify deployment success
"""

import asyncio
import aiohttp
import json
import sys
import time
from datetime import datetime

async def test_endpoint(session, url, endpoint, expected_status=200):
    """Test a single endpoint"""
    full_url = f"{url}{endpoint}"
    try:
        async with session.get(full_url, timeout=10) as response:
            status = response.status
            content_type = response.headers.get('content-type', '')
            
            if status == expected_status:
                if 'application/json' in content_type:
                    data = await response.json()
                    return True, f"✅ {endpoint} - Status: {status} - JSON response received"
                else:
                    return True, f"✅ {endpoint} - Status: {status} - Response received"
            else:
                return False, f"❌ {endpoint} - Expected: {expected_status}, Got: {status}"
                
    except asyncio.TimeoutError:
        return False, f"❌ {endpoint} - Timeout after 10 seconds"
    except Exception as e:
        return False, f"❌ {endpoint} - Error: {str(e)}"

async def test_deployment():
    """Test the deployment"""
    base_url = "http://localhost:8000"
    
    # Critical endpoints to test
    endpoints = [
        ("/ping", 200),
        ("/health", 200),
        ("/api/market-data", 200),
        ("/api/trading-pairs", 200),
        ("/api/market-overview", 200),
        ("/api/endpoints", 200),
        ("/", 200),  # Dashboard
    ]
    
    print("🧪 Elite Trading Bot V3.0 - Deployment Test")
    print("=" * 50)
    print(f"🎯 Testing server at: {base_url}")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    await asyncio.sleep(2)
    
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        results = []
        
        for endpoint, expected_status in endpoints:
            success, message = await test_endpoint(session, base_url, endpoint, expected_status)
            results.append((success, message))
            print(message)
            await asyncio.sleep(0.5)  # Small delay between tests
    
    print()
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    successful = sum(1 for success, _ in results if success)
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    print(f"📈 Success Rate: {(successful/total)*100:.1f}%")
    
    if successful == total:
        print()
        print("🎉 ALL TESTS PASSED! Deployment successful!")
        print("🌐 Your bot is ready at: http://localhost:8000")
        print("📊 Market Data API: http://localhost:8000/api/market-data")
        return True
    else:
        print()
        print("⚠️  Some tests failed. Check the logs and fix issues.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_deployment())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        sys.exit(1)
EOF

echo "✅ Testing script created (test_deployment.py)"

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "📋 Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Elite Trading Bot V3.0 - Core Dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
jinja2>=3.1.2
aiofiles>=23.2.1
aiohttp>=3.9.0
python-dotenv>=1.0.0
psutil>=5.9.0
numpy>=1.24.0

# Optional: ML and Trading Dependencies
# pandas>=2.0.0
# ccxt>=4.0.0
# tensorflow>=2.13.0
# scikit-learn>=1.3.0
EOF
    echo "✅ requirements.txt created"
fi

# Final deployment summary
echo ""
echo "🎯 Deployment Summary"
echo "==================="
echo "✅ Backup created for original main.py"
echo "✅ Enhanced main.py ready for deployment"
echo "✅ Directory structure created"
echo "✅ Environment configuration ready"
echo "✅ Startup script created (start_bot.sh)"
echo "✅ Testing script created (test_deployment.py)"
echo "✅ Requirements file ready"
echo ""
echo "🚀 Next Steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Start the bot: ./start_bot.sh (or python main.py)"
echo "3. Test deployment: python test_deployment.py"
echo "4. Access dashboard: http://localhost:8000"
echo "5. Test API: http://localhost:8000/api/market-data"
echo ""
echo "🔧 Troubleshooting:"
echo "   - Check logs in logs/ directory"
echo "   - Use /health endpoint for diagnostics"
echo "   - Use /api/endpoints to verify all routes"
echo "   - Use /ping for basic connectivity"
echo ""
echo "✨ Your Elite Trading Bot V3.0 is ready for deployment!"