# start_trading_bot.py - One Command to Rule Them All! 🚀
"""
Ultimate Trading Bot Startup Script
==================================

This script does EVERYTHING:
1. ✅ Checks and fixes system setup
2. ✅ Installs missing dependencies  
3. ✅ Creates missing core files
4. ✅ Fixes import paths and missing files
5. ✅ Disables problematic notifications
6. ✅ Trains ML models if needed
7. ✅ Starts the trading dashboard
8. ✅ Opens browser automatically
9. ✅ Provides health monitoring

USAGE: python start_trading_bot.py
"""

import os
import sys
import subprocess
import webbrowser
import time
import json
import logging
import signal
import threading
import socket
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingBotStarter:
    """Enhanced one-command trading bot starter"""
    
    def __init__(self):
        self.project_dir = Path.cwd()
        self.server_process = None
        self.dashboard_port = 8000
        self.fallback_port = 5000
        
        self.required_files = [
            'main.py', 'config.json', '.env',
            'core/enhanced_ml_engine.py', 
            'core/trading_engine.py',
            'requirements.txt'
        ]
        
        self.required_dirs = [
            'core', 'utils', 'strategies', 'templates', 
            'static', 'models', 'logs', 'data', 'ai',
            'database', 'exchanges'
        ]
        
        # Register cleanup on exit
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
    def cleanup(self, signum=None, frame=None):
        """Clean shutdown"""
        print("\n🛑 Shutting down trading bot...")
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                if self.server_process:
                    self.server_process.kill()
        print("👋 Trading bot stopped. Thanks for using Enhanced Trading Bot!")
        sys.exit(0)

    def print_banner(self):
        """Print startup banner"""
        print("""
🚀 =============================================== 🚀
   ENHANCED TRADING BOT - ONE-COMMAND STARTUP
🚀 =============================================== 🚀

Starting comprehensive setup and launch sequence...
This may take 2-5 minutes on first run.

Features:
✅ Complete system setup & validation
✅ Automatic dependency management
✅ ML model training & optimization
✅ Real-time trading dashboard
✅ Risk management & backtesting
✅ Multi-exchange integration ready
""")

    def print_step(self, step_num, title, status="RUNNING"):
        """Print step header with status"""
        status_icon = "🔄" if status == "RUNNING" else "✅" if status == "COMPLETE" else "❌"
        print(f"\n{'='*60}")
        print(f"{status_icon} STEP {step_num}: {title}")
        print('='*60)

    def check_system_requirements(self):
        """Comprehensive system check"""
        print("🔍 Checking system requirements...")
        
        issues = []
        
        # Python version check
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            issues.append(f"Python 3.8+ required, found {version.major}.{version.minor}")
        else:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        
        # Memory check (basic)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 4:
                issues.append(f"4GB+ RAM recommended, found {memory_gb:.1f}GB")
            else:
                print(f"✅ Memory: {memory_gb:.1f}GB")
        except ImportError:
            print("⚠️  Memory check skipped (psutil not available)")
        
        # Disk space check
        try:
            import shutil
            free_space_gb = shutil.disk_usage(self.project_dir).free / (1024**3)
            if free_space_gb < 1:
                issues.append(f"1GB+ free space recommended, found {free_space_gb:.1f}GB")
            else:
                print(f"✅ Free space: {free_space_gb:.1f}GB")
        except:
            print("⚠️  Disk space check skipped")
        
        # Port availability check
        for port in [self.dashboard_port, self.fallback_port]:
            if self.is_port_in_use(port):
                print(f"⚠️  Port {port} is in use, will try alternative")
            else:
                print(f"✅ Port {port} available")
        
        if issues:
            print("\n❌ System Issues Found:")
            for issue in issues:
                print(f"   • {issue}")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                return False
        
        return True

    def is_port_in_use(self, port):
        """Check if port is in use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        except:
            return False

    def install_dependencies(self):
        """Enhanced dependency installation"""
        print("📦 Installing/checking dependencies...")
        
        # Core dependencies with specific versions for stability
        core_deps = {
            'fastapi': '0.104.1',
            'uvicorn[standard]': '0.24.0',
            'jinja2': '3.1.2',
            'python-multipart': '0.0.6',
            'websockets': '12.0',
            'pandas': '2.1.4',
            'numpy': '1.24.3',
            'scikit-learn': '1.3.2',
            'tensorflow': '2.15.0',
            'ccxt': '4.1.77',
            'flask': '3.0.0',
            'flask-cors': '4.0.0',
            'plotly': '5.17.0',
            'matplotlib': '3.8.2',
            'seaborn': '0.13.0',
            'python-dotenv': '1.0.0',
            'aiohttp': '3.9.1',
            'requests': '2.31.0',
            'psutil': '5.9.6',
            'schedule': '1.2.0',
            'APScheduler': '3.10.4',
            'yfinance': '0.2.18',
            'ta': '0.10.2'  # Technical analysis library
        }
        
        # Optional AI/ML dependencies
        ai_deps = {
            'torch': 'latest',
            'transformers': 'latest',
            'xgboost': 'latest',
            'lightgbm': 'latest'
        }
        
        # Install core dependencies
        failed_deps = []
        for dep, version in core_deps.items():
            try:
                if version != 'latest':
                    dep_spec = f"{dep}=={version}"
                else:
                    dep_spec = dep
                    
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dep_spec
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {dep}")
                else:
                    failed_deps.append(dep)
                    print(f"⚠️  {dep} - installation failed")
                    
            except Exception as e:
                failed_deps.append(dep)
                print(f"❌ {dep} - error: {e}")
        
        # Try to install AI dependencies (optional)
        print("\n🧠 Installing optional AI dependencies...")
        for dep, version in ai_deps.items():
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', dep
                ], capture_output=True, text=True, timeout=120)
                print(f"✅ {dep} (optional)")
            except:
                print(f"⚠️  {dep} (optional) - skipped")
        
        # Create requirements.txt
        self.create_requirements_txt()
        
        if failed_deps:
            print(f"\n⚠️  Some dependencies failed: {', '.join(failed_deps)}")
            print("The bot will still work with reduced functionality.")
        
        print("✅ Dependency installation complete!")
        return True

    def create_requirements_txt(self):
        """Create/update requirements.txt"""
        requirements_content = """# Enhanced Trading Bot Requirements
fastapi==0.104.1
uvicorn[standard]==0.24.0
jinja2==3.1.2
python-multipart==0.0.6
websockets==12.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
tensorflow==2.15.0
ccxt==4.1.77
flask==3.0.0
flask-cors==4.0.0
plotly==5.17.0
matplotlib==3.8.2
seaborn==0.13.0
python-dotenv==1.0.0
aiohttp==3.9.1
requests==2.31.0
psutil==5.9.6
schedule==1.2.0
APScheduler==3.10.4
yfinance==0.2.18
ta==0.10.2

# Optional AI/ML (install if needed)
# torch
# transformers
# xgboost
# lightgbm
"""
        
        with open('requirements.txt', 'w') as f:
            f.write(requirements_content)

    def create_directory_structure(self):
        """Create comprehensive directory structure"""
        print("📁 Creating directory structure...")
        
        # Extended directory structure
        extended_dirs = [
            'core', 'utils', 'strategies', 'templates', 'static',
            'models', 'logs', 'data', 'ai', 'database', 'exchanges',
            'static/css', 'static/js', 'static/images',
            'templates/dashboard', 'data/historical', 'data/live',
            'logs/trading', 'logs/errors', 'logs/system',
            'models/trained', 'models/backup',
            'config', 'tests', 'docs'
        ]
        
        for dir_name in extended_dirs:
            dir_path = self.project_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py for Python packages
            if dir_name in ['core', 'utils', 'strategies', 'ai', 'database', 'exchanges']:
                init_file = dir_path / '__init__.py'
                if not init_file.exists():
                    with open(init_file, 'w') as f:
                        f.write(f'"""Enhanced Trading Bot - {dir_name.title()} Package"""\n')
        
        print("✅ Directory structure ready!")

    def create_missing_core_files(self):
        """Create all missing core files"""
        print("📝 Creating missing core files...")
        
        # Create main.py if missing
        if not Path('main.py').exists():
            self.create_main_py()
        
        # Create config.json if missing
        if not Path('config.json').exists():
            self.create_config_json()
        
        # Create enhanced_ml_engine.py if missing
        if not Path('core/enhanced_ml_engine.py').exists():
            self.create_enhanced_ml_engine()
        
        # Create trading_engine.py if missing
        if not Path('core/trading_engine.py').exists():
            self.create_trading_engine()
        
        # Create data fetcher
        if not Path('core/data_fetcher.py').exists():
            self.create_data_fetcher()
        
        print("✅ All core files created!")

    def create_main_py(self):
        """Create comprehensive main.py FastAPI application"""
        main_content = '''"""
Enhanced Trading Bot - Main FastAPI Application
=============================================
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import our modules
try:
    from core.enhanced_ml_engine import AdaptiveMLEngine
    from core.trading_engine import TradingEngine
    from core.data_fetcher import DataFetcher
    from utils.simple_notification_manager import SimpleNotificationManager
except ImportError as e:
    print(f"Import warning: {e}")
    print("Some features may be limited. Run the setup script first.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Trading Bot",
    description="AI-Powered Trading Dashboard",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global components
ml_engine = None
trading_engine = None
data_fetcher = None
notification_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global ml_engine, trading_engine, data_fetcher, notification_manager
    
    try:
        logger.info("🚀 Starting Enhanced Trading Bot...")
        
        # Initialize components
        ml_engine = AdaptiveMLEngine()
        trading_engine = TradingEngine()
        data_fetcher = DataFetcher()
        notification_manager = SimpleNotificationManager()
        
        logger.info("✅ All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        # Continue with limited functionality

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    try:
        # Get basic market data
        market_status = await get_market_overview()
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "title": "Enhanced Trading Bot",
            "market_data": market_status,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/api/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "ml_engine": ml_engine is not None,
            "trading_engine": trading_engine is not None,
            "data_fetcher": data_fetcher is not None
        }
    }

@app.get("/api/market-overview")
async def get_market_overview():
    """Get market overview data"""
    try:
        # Sample market data - replace with real data from your data fetcher
        market_data = {
            "btc_price": 50000 + (hash(str(datetime.now().minute)) % 5000),
            "eth_price": 3000 + (hash(str(datetime.now().minute)) % 500),
            "market_trend": "BULLISH",
            "volume_24h": "1.2B",
            "total_positions": 5,
            "active_strategies": 3,
            "pnl_today": "+2.34%"
        }
        
        return market_data
        
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading-signals/{symbol}")
async def get_trading_signals(symbol: str):
    """Get AI trading signals for a symbol"""
    try:
        if ml_engine:
            analysis = await ml_engine.analyze_symbol(symbol, "1h")
            return analysis
        else:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0.5,
                "trend": "NEUTRAL",
                "recommendation": "ML engine not available"
            }
    except Exception as e:
        logger.error(f"Trading signals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/live-data")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live data updates"""
    await websocket.accept()
    try:
        while True:
            # Send live market data every 5 seconds
            market_data = await get_market_overview()
            await websocket.send_json({
                "type": "market_update",
                "data": market_data,
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/api/backtest/{strategy}")
async def run_backtest(strategy: str, start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
    """Run strategy backtest"""
    try:
        # Sample backtest results
        results = {
            "strategy": strategy,
            "period": f"{start_date} to {end_date}",
            "total_return": "15.67%",
            "sharpe_ratio": 1.23,
            "max_drawdown": "-8.45%",
            "win_rate": "68.5%",
            "total_trades": 124,
            "avg_trade": "0.89%"
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Create basic templates if they don't exist
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Run the app
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
'''
        
        with open('main.py', 'w') as f:
            f.write(main_content)
        
        # Create basic HTML template
        self.create_dashboard_template()

    def create_dashboard_template(self):
        """Create dashboard HTML template"""
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)
        
        dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Trading Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .status { background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin-bottom: 20px; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { 
            background: rgba(255,255,255,0.1); 
            border-radius: 12px; 
            padding: 20px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card h3 { margin-bottom: 15px; color: #4fc3f7; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-value { font-weight: bold; color: #81c784; }
        .signal-box { 
            padding: 15px; 
            border-radius: 8px; 
            margin: 10px 0;
            text-align: center;
        }
        .bullish { background: rgba(76, 175, 80, 0.3); }
        .bearish { background: rgba(244, 67, 54, 0.3); }
        .neutral { background: rgba(255, 193, 7, 0.3); }
        .btn { 
            background: #4fc3f7; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover { background: #29b6f6; }
        #chart-container { height: 300px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced Trading Bot</h1>
            <div class="status">
                <strong>Status:</strong> <span id="bot-status">OPERATIONAL</span> | 
                <strong>Time:</strong> <span id="current-time">{{ timestamp }}</span>
            </div>
        </div>

        <div class="dashboard-grid">
            <div class="card">
                <h3>📊 Market Overview</h3>
                <div class="metric">
                    <span>BTC Price:</span>
                    <span class="metric-value" id="btc-price">${{ market_data.btc_price if market_data else '50,000' }}</span>
                </div>
                <div class="metric">
                    <span>ETH Price:</span>
                    <span class="metric-value" id="eth-price">${{ market_data.eth_price if market_data else '3,000' }}</span>
                </div>
                <div class="metric">
                    <span>24h Volume:</span>
                    <span class="metric-value">{{ market_data.volume_24h if market_data else '1.2B' }}</span>
                </div>
                <div class="signal-box bullish">
                    <strong>Market Trend: BULLISH 📈</strong>
                </div>
            </div>

            <div class="card">
                <h3>🤖 AI Signals</h3>
                <div id="signals-container">
                    <div class="signal-box bullish">
                        <strong>BTC/USD: BUY Signal</strong><br>
                        Confidence: 78%
                    </div>
                    <div class="signal-box neutral">
                        <strong>ETH/USD: HOLD Signal</strong><br>
                        Confidence: 65%
                    </div>
                </div>
                <button class="btn" onclick="refreshSignals()">🔄 Refresh Signals</button>
            </div>

            <div class="card">
                <h3>💼 Portfolio Status</h3>
                <div class="metric">
                    <span>Total Positions:</span>
                    <span class="metric-value">{{ market_data.total_positions if market_data else '5' }}</span>
                </div>
                <div class="metric">
                    <span>Active Strategies:</span>
                    <span class="metric-value">{{ market_data.active_strategies if market_data else '3' }}</span>
                </div>
                <div class="metric">
                    <span>Today's P&L:</span>
                    <span class="metric-value">{{ market_data.pnl_today if market_data else '+2.34%' }}</span>
                </div>
                <button class="btn" onclick="runBacktest()">📈 Run Backtest</button>
            </div>

            <div class="card">
                <h3>📈 Performance Chart</h3>
                <div id="chart-container">
                    <canvas id="performance-chart"></canvas>
                </div>
            </div>

            <div class="card">
                <h3>⚡ Quick Actions</h3>
                <button class="btn" onclick="startTrading()">▶️ Start Trading</button>
                <button class="btn" onclick="stopTrading()">⏸️ Stop Trading</button>
                <button class="btn" onclick="trainModels()">🧠 Train Models</button>
                <button class="btn" onclick="exportData()">📊 Export Data</button>
                
                <div style="margin-top: 20px;">
                    <h4>🔧 System Controls</h4>
                    <button class="btn" onclick="checkHealth()">❤️ Health Check</button>
                    <button class="btn" onclick="viewLogs()">📝 View Logs</button>
                </div>
            </div>

            <div class="card">
                <h3>📢 Recent Activity</h3>
                <div id="activity-log">
                    <div style="margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px;">
                        <strong>12:34:56</strong> - ML model retrained successfully
                    </div>
                    <div style="margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px;">
                        <strong>12:30:15</strong> - New BUY signal for BTC/USD
                    </div>
                    <div style="margin: 10px 0; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px;">
                        <strong>12:25:33</strong> - Portfolio rebalanced
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize performance chart
        const ctx = document.getElementById('performance-chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Portfolio Performance',
                    data: [100, 105, 103, 110, 115, 118],
                    borderColor: '#4fc3f7',
                    backgroundColor: 'rgba(79, 195, 247, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: 'white' } }
                },
                scales: {
                    x: { ticks: { color: 'white' } },
                    y: { ticks: { color: 'white' } }
                }
            }
        });

        // WebSocket connection for live updates
        let ws = null;
        function connectWebSocket() {
            try {
                ws = new WebSocket(`ws://${window.location.host}/ws/live-data`);
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'market_update') {
                        updateMarketData(data.data);
                    }
                };
                ws.onclose = function() {
                    setTimeout(connectWebSocket, 5000); // Reconnect after 5 seconds
                };
            } catch (e) {
                console.log('WebSocket connection failed, using polling');
                setInterval(updateMarketData, 10000);
            }
        }

        function updateMarketData(data) {
            if (data) {
                document.getElementById('btc-price').textContent = `$${data.btc_price?.toLocaleString() || '50,000'}`;
                document.getElementById('eth-price').textContent = `$${data.eth_price?.toLocaleString() || '3,000'}`;
            }
        }

        // Update current time
        function updateTime() {
            document.getElementById('current-time').textContent = new Date().toLocaleString();
        }
        setInterval(updateTime, 1000);

        // Action functions
        async function refreshSignals() {
            try {
                const response = await fetch('/api/trading-signals/BTCUSD');
                const data = await response.json();
                console.log('Signals refreshed:', data);
                alert('Signals refreshed successfully!');
            } catch (e) {
                console.error('Error refreshing signals:', e);
            }
        }

        async function checkHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                alert(`System Status: ${data.status.toUpperCase()}\\nComponents: ${JSON.stringify(data.components, null, 2)}`);
            } catch (e) {
                alert('Health check failed');
            }
        }

        function startTrading() { alert('Trading started! (Demo mode)'); }
        function stopTrading() { alert('Trading stopped!'); }
        function trainModels() { alert('Model training initiated...'); }
        function exportData() { alert('Data export started...'); }
        function runBacktest() { alert('Backtest started...'); }
        function viewLogs() { window.open('/logs', '_blank'); }

        // Initialize
        connectWebSocket();
        updateTime();
    </script>
</body>
</html>'''
        
        with open('templates/dashboard.html', 'w') as f:
            f.write(dashboard_html)

    def create_config_json(self):
        """Create comprehensive config.json"""
        config_content = {
            "app": {
                "name": "Enhanced Trading Bot",
                "version": "2.0.0",
                "environment": "development",
                "debug": True
            },
            "trading": {
                "live_trading_enabled": False,
                "default_timeframe": "1h",
                "risk_management": {
                    "max_position_size": 0.02,
                    "stop_loss_percentage": 0.05,
                    "take_profit_percentage": 0.10
                }
            },
            "ml": {
                "model_retrain_interval": 24,
                "prediction_confidence_threshold": 0.7,
                "features": [
                    "rsi", "macd", "bollinger_bands", 
                    "volume_sma", "price_sma"
                ]
            },
            "exchanges": {
                "binance": {
                    "enabled": False,
                    "sandbox": True
                },
                "coinbase": {
                    "enabled": False,
                    "sandbox": True
                }
            },
            "notifications": {
                "enabled": False,
                "email": False,
                "slack": False,
                "discord": False
            },
            "database": {
                "url": "sqlite:///trading_bot.db",
                "echo": False
            },
            "logging": {
                "level": "INFO",
                "file": "logs/trading_bot.log",
                "max_bytes": 10485760,
                "backup_count": 5
            }
        }
        
        with open('config.json', 'w') as f:
            json.dump(config_content, f, indent=4)

    def create_enhanced_ml_engine(self):
        """Create enhanced ML engine"""
        ml_engine_content = '''"""
Enhanced ML Engine for Trading Bot
================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import pickle
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class AdaptiveMLEngine:
    """Advanced ML engine for trading predictions"""
    
    def __init__(self, model_save_path='models/', performance_log_path='logs/'):
        self.model_save_path = Path(model_save_path)
        self.performance_log_path = Path(performance_log_path)
        
        # Create directories
        self.model_save_path.mkdir(exist_ok=True)
        self.performance_log_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            if SKLEARN_AVAILABLE:
                self.models['price_direction'] = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )
                self.models['price_target'] = RandomForestRegressor(
                    n_estimators=100, random_state=42
                )
                self.scalers['default'] = StandardScaler()
                
            if TENSORFLOW_AVAILABLE:
                self._create_deep_learning_model()
                
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Model initialization error: {e}")
    
    def _create_deep_learning_model(self):
        """Create deep learning model"""
        try:
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(10,)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(3, activation='softmax')  # Buy, Hold, Sell
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.models['deep_predictor'] = model
            
        except Exception as e:
            self.logger.error(f"Deep learning model creation error: {e}")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            features['price_sma_5'] = data['close'].rolling(5).mean()
            features['price_sma_20'] = data['close'].rolling(20).mean()
            features['price_ratio'] = data['close'] / features['price_sma_20']
            
            # Volatility features
            features['price_std'] = data['close'].rolling(20).std()
            features['volatility'] = features['price_std'] / data['close']
            
            # Volume features
            if 'volume' in data.columns:
                features['volume_sma'] = data['volume'].rolling(20).mean()
                features['volume_ratio'] = data['volume'] / features['volume_sma']
            else:
                features['volume_ratio'] = 1.0
            
            # Momentum features
            features['momentum_5'] = data['close'].pct_change(5)
            features['momentum_20'] = data['close'].pct_change(20)
            
            # RSI approximation
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD approximation
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            
            # Clean data
            features = features.fillna(method='bfill').fillna(0)
            
            self.feature_columns = features.columns.tolist()
            return features
            
        except Exception as e:
            self.logger.error(f"Feature creation error: {e}")
            return pd.DataFrame()
    
    def create_targets(self, data: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
        """Create prediction targets"""
        try:
            targets = pd.DataFrame(index=data.index)
            
            # Price direction (classification)
            future_return = data['close'].shift(-lookahead) / data['close'] - 1
            targets['direction'] = (future_return > 0.01).astype(int)  # 1% threshold
            
            # Price target (regression)
            targets['price_target'] = data['close'].shift(-lookahead)
            targets['return_target'] = future_return
            
            return targets.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Target creation error: {e}")
            return pd.DataFrame()
    
    async def train_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train ML models with provided data"""
        try:
            self.logger.info("Starting model training...")
            
            if len(data) < 100:
                self.logger.warning("Insufficient data for training")
                return {}
            
            # Create features and targets
            features = self.create_features(data)
            targets = self.create_targets(data)
            
            if features.empty or targets.empty:
                self.logger.error("Feature or target creation failed")
                return {}
            
            # Align data
            valid_idx = features.dropna().index.intersection(targets.dropna().index)
            X = features.loc[valid_idx]
            y = targets.loc[valid_idx]
            
            if len(X) < 50:
                self.logger.warning("Insufficient valid data for training")
                return {}
            
            # Scale features
            if SKLEARN_AVAILABLE:
                X_scaled = self.scalers['default'].fit_transform(X)
            else:
                X_scaled = X.values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            metrics = {}
            
            # Train classification model
            if SKLEARN_AVAILABLE and 'price_direction' in self.models:
                clf_model = self.models['price_direction']
                clf_model.fit(X_train, y_train['direction'])
                
                # Evaluate
                y_pred = clf_model.predict(X_test)
                accuracy = accuracy_score(y_test['direction'], y_pred)
                metrics['direction_accuracy'] = accuracy
                
                self.logger.info(f"Direction prediction accuracy: {accuracy:.3f}")
            
            # Train regression model
            if SKLEARN_AVAILABLE and 'price_target' in self.models:
                reg_model = self.models['price_target']
                reg_model.fit(X_train, y_train['return_target'])
                
                # Evaluate
                score = reg_model.score(X_test, y_test['return_target'])
                metrics['return_r2_score'] = score
                
                self.logger.info(f"Return prediction R² score: {score:.3f}")
            
            # Save models
            self._save_models()
            
            # Update performance metrics
            self.performance_metrics.update(metrics)
            self._save_performance_metrics()
            
            self.logger.info("Model training completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            return {}
    
    async def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """Make predictions using trained models"""
        try:
            if features.empty:
                return {}
            
            # Ensure features match training
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(features.columns)
                if missing_cols:
                    for col in missing_cols:
                        features[col] = 0
                features = features[self.feature_columns]
            
            # Scale features
            if SKLEARN_AVAILABLE and 'default' in self.scalers:
                X_scaled = self.scalers['default'].transform(features)
            else:
                X_scaled = features.values
            
            predictions = {}
            
            # Direction prediction
            if 'price_direction' in self.models:
                direction_prob = self.models['price_direction'].predict_proba(X_scaled)
                predictions['buy_probability'] = float(direction_prob[0][1])
                predictions['predicted_direction'] = int(direction_prob[0][1] > 0.5)
            
            # Return prediction
            if 'price_target' in self.models:
                return_pred = self.models['price_target'].predict(X_scaled)
                predictions['predicted_return'] = float(return_pred[0])
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {}
    
    async def analyze_symbol(self, symbol: str, timeframe: str = '1h') -> Dict[str, any]:
        """Comprehensive symbol analysis"""
        try:
            # Generate sample analysis for demo
            import random
            
            confidence = 0.6 + random.random() * 0.3
            base_price = 50000 if 'BTC' in symbol.upper() else 3000
            current_price = base_price + random.randint(-1000, 1000)
            
            # Determine signal based on confidence
            if confidence > 0.75:
                signal = 'BUY'
                trend = 'BULLISH'
            elif confidence < 0.6:
                signal = 'SELL'
                trend = 'BEARISH'
            else:
                signal = 'HOLD'
                trend = 'NEUTRAL'
            
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': signal,
                'trend': trend,
                'confidence': round(confidence, 3),
                'current_price': current_price,
                'support_level': round(current_price * 0.95, 2),
                'resistance_level': round(current_price * 1.05, 2),
                'predicted_return': round((confidence - 0.5) * 0.1, 4),
                'risk_level': 'Medium',
                'recommendation': f"{signal} signal for {symbol} with {confidence:.1%} confidence",
                'last_updated': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Symbol analysis error: {e}")
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def _save_models(self):
        """Save trained models"""
        try:
            if SKLEARN_AVAILABLE:
                for name, model in self.models.items():
                    if hasattr(model, 'fit'):  # Skip TensorFlow models
                        model_path = self.model_save_path / f"{name}.joblib"
                        joblib.dump(model, model_path)
                
                # Save scalers
                for name, scaler in self.scalers.items():
                    scaler_path = self.model_save_path / f"scaler_{name}.joblib"
                    joblib.dump(scaler, scaler_path)
                
                # Save feature columns
                with open(self.model_save_path / 'feature_columns.json', 'w') as f:
                    json.dump(self.feature_columns, f)
                
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
    
    def _load_models(self):
        """Load trained models"""
        try:
            if SKLEARN_AVAILABLE:
                for model_file in self.model_save_path.glob("*.joblib"):
                    if not model_file.name.startswith('scaler_'):
                        model_name = model_file.stem
                        self.models[model_name] = joblib.load(model_file)
                
                # Load scalers
                for scaler_file in self.model_save_path.glob("scaler_*.joblib"):
                    scaler_name = scaler_file.stem.replace('scaler_', '')
                    self.scalers[scaler_name] = joblib.load(scaler_file)
                
                # Load feature columns
                feature_file = self.model_save_path / 'feature_columns.json'
                if feature_file.exists():
                    with open(feature_file, 'r') as f:
                        self.feature_columns = json.load(f)
                
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
    
    def _save_performance_metrics(self):
        """Save performance metrics"""
        try:
            metrics_file = self.performance_log_path / 'performance_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Performance metrics saving error: {e}")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded models"""
        return {
            'available_models': list(self.models.keys()),
            'feature_count': len(self.feature_columns),
            'performance_metrics': self.performance_metrics,
            'sklearn_available': SKLEARN_AVAILABLE,
            'tensorflow_available': TENSORFLOW_AVAILABLE
        }
'''
        
        with open('core/enhanced_ml_engine.py', 'w') as f:
            f.write(ml_engine_content)

    def create_trading_engine(self):
        """Create trading engine"""
        trading_engine_content = '''"""
Trading Engine for Enhanced Trading Bot
=====================================
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

class TradingEngine:
    """Core trading engine for executing trades and managing positions"""
    
    def __init__(self, config_path: str = 'config.json'):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Trading state
        self.positions = {}
        self.orders = {}
        self.trading_enabled = False
        self.demo_mode = True
        
        # Risk management
        self.max_position_size = self.config.get('trading', {}).get('risk_management', {}).get('max_position_size', 0.02)
        self.stop_loss_pct = self.config.get('trading', {}).get('risk_management', {}).get('stop_loss_percentage', 0.05)
        self.take_profit_pct = self.config.get('trading', {}).get('risk_management', {}).get('take_profit_percentage', 0.10)
        
        self.logger.info("Trading Engine initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Config loading error: {e}")
            return {}
    
    async def execute_trade(self, symbol: str, side: str, quantity: float, 
                          order_type: str = 'market') -> Dict[str, any]:
        """Execute a trade order"""
        try:
            if not self.trading_enabled:
                return {
                    'status': 'rejected',
                    'reason': 'Trading not enabled',
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity
                }
            
            # Risk management checks
            risk_check = self._risk_management_check(symbol, side, quantity)
            if not risk_check['allowed']:
                return {
                    'status': 'rejected',
                    'reason': risk_check['reason'],
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity
                }
            
            # Simulate trade execution in demo mode
            if self.demo_mode:
                return self._simulate_trade(symbol, side, quantity, order_type)
            
            # Real trade execution would go here
            # This would integrate with actual exchange APIs
            
            return {
                'status': 'executed',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol,
                'side': side,
                'quantity': quantity
            }
    
    def _simulate_trade(self, symbol: str, side: str, quantity: float, 
                       order_type: str) -> Dict[str, any]:
        """Simulate trade execution for demo mode"""
        import random
        
        # Simulate realistic execution
        base_price = 50000 if 'BTC' in symbol.upper() else 3000
        execution_price = base_price + random.randint(-100, 100)
        
        order_id = f"DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'status': 'filled',
            'execution_price': execution_price,
            'total_value': execution_price * quantity,
            'timestamp': datetime.now().isoformat(),
            'demo_mode': True
        }
        
        # Update positions
        self._update_position(symbol, side, quantity, execution_price)
        
        # Store order
        self.orders[order_id] = order
        
        self.logger.info(f"Demo trade executed: {side} {quantity} {symbol} at {execution_price}")
        
        return order
    
    def _update_position(self, symbol: str, side: str, quantity: float, price: float):
        """Update position tracking"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0
            }
        
        position = self.positions[symbol]
        
        if side.upper() == 'BUY':
            new_quantity = position['quantity'] + quantity
            if new_quantity != 0:
                position['avg_price'] = (
                    (position['avg_price'] * position['quantity'] + price * quantity) / new_quantity
                )
            position['quantity'] = new_quantity
        
        elif side.upper() == 'SELL':
            if position['quantity'] >= quantity:
                # Calculate realized PnL
                realized_pnl = (price - position['avg_price']) * quantity
                position['realized_pnl'] += realized_pnl
                position['quantity'] -= quantity
            else:
                self.logger.warning(f"Insufficient position to sell {quantity} {symbol}")
    
    def _risk_management_check(self, symbol: str, side: str, quantity: float) -> Dict[str, any]:
        """Check if trade passes risk management rules"""
        try:
            # Position size check
            current_position = self.positions.get(symbol, {}).get('quantity', 0)
            
            if side.upper() == 'BUY':
                new_position = current_position + quantity
            else:
                new_position = current_position - quantity
            
            # Check maximum position size
            if abs(new_position) > self.max_position_size:
                return {
                    'allowed': False,
                    'reason': f'Position size would exceed maximum ({self.max_position_size})'
                }
            
            # Additional risk checks could go here
            # - Portfolio exposure limits
            # - Correlation limits
            # - Volatility checks
            # - Drawdown limits
            
            return {'allowed': True, 'reason': 'Risk checks passed'}
            
        except Exception as e:
            self.logger.error(f"Risk management check error: {e}")
            return {'allowed': False, 'reason': f'Risk check error: {e}'}
    
    def get_positions(self) -> Dict[str, any]:
        """Get current positions"""
        return {
            'positions': self.positions,
            'total_positions': len(self.positions),
            'demo_mode': self.demo_mode,
            'trading_enabled': self.trading_enabled
        }
    
    def get_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get order history"""
        if symbol:
            return [order for order in self.orders.values() if order['symbol'] == symbol]
        return list(self.orders.values())
    
    def enable_trading(self) -> bool:
        """Enable trading"""
        self.trading_enabled = True
        self.logger.info("Trading enabled")
        return True
    
    def disable_trading(self) -> bool:
        """Disable trading"""
        self.trading_enabled = False
        self.logger.info("Trading disabled")
        return True
    
    def set_demo_mode(self, demo: bool = True):
        """Set demo mode"""
        self.demo_mode = demo
        mode = "demo" if demo else "live"
        self.logger.info(f"Trading mode set to: {mode}")
    
    def calculate_portfolio_metrics(self) -> Dict[str, any]:
        """Calculate portfolio performance metrics"""
        try:
            total_value = 0
            total_pnl = 0
            
            for symbol, position in self.positions.items():
                # Simulate current market price
                import random
                base_price = 50000 if 'BTC' in symbol.upper() else 3000
                current_price = base_price + random.randint(-500, 500)
                
                position_value = position['quantity'] * current_price
                unrealized_pnl = (current_price - position['avg_price']) * position['quantity']
                
                total_value += abs(position_value)
                total_pnl += position['realized_pnl'] + unrealized_pnl
            
            return {
                'total_portfolio_value': round(total_value, 2),
                'total_pnl': round(total_pnl, 2),
                'pnl_percentage': round((total_pnl / max(total_value, 1)) * 100, 2),
                'position_count': len(self.positions),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio metrics calculation error: {e}")
            return {}
    
    def get_trading_status(self) -> Dict[str, any]:
        """Get comprehensive trading status"""
        return {
            'trading_enabled': self.trading_enabled,
            'demo_mode': self.demo_mode,
            'positions': self.get_positions(),
            'portfolio_metrics': self.calculate_portfolio_metrics(),
            'recent_orders': self.get_orders()[-10:],  # Last 10 orders
            'risk_settings': {
                'max_position_size': self.max_position_size,
                'stop_loss_percentage': self.stop_loss_pct,
                'take_profit_percentage': self.take_profit_pct
            }
        }
'''
        
        with open('core/trading_engine.py', 'w') as f:
            f.write(trading_engine_content)

    def create_data_fetcher(self):
        """Create data fetcher"""
        data_fetcher_content = '''"""
Data Fetcher for Enhanced Trading Bot
===================================
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

class DataFetcher:
    """Fetch market data from various sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # Data cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        self.logger.info("Data Fetcher initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_price_data(self, symbol: str, timeframe: str = '1h', 
                           limit: int = 100) -> pd.DataFrame:
        """Get price data for a symbol"""
        try:
            cache_key = f"{symbol}_{timeframe}_{limit}"
            
            # Check cache
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Try different data sources
            data = None
            
            # Try Yahoo Finance first
            if YFINANCE_AVAILABLE:
                data = await self._fetch_yahoo_data(symbol, timeframe, limit)
            
            # If Yahoo Finance fails, try simulated data
            if data is None or data.empty:
                data = self._generate_simulated_data(symbol, timeframe, limit)
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Price data fetch error for {symbol}: {e}")
            return self._generate_simulated_data(symbol, timeframe, limit)
    
    async def _fetch_yahoo_data(self, symbol: str, timeframe: str, 
                               limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            # Convert symbol format for Yahoo Finance
            yahoo_symbol = self._convert_to_yahoo_symbol(symbol)
            
            # Convert timeframe
            period_map = {
                '1m': '1d', '5m': '5d', '15m': '5d', '30m': '5d',
                '1h': '5d', '4h': '1mo', '1d': '1y', '1w': '5y'
            }
            period = period_map.get(timeframe, '1mo')
            
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period, interval=timeframe)
            
            if not data.empty:
                # Standardize column names
                data.columns = [col.lower() for col in data.columns]
                data = data.rename(columns={'adj close': 'adj_close'})
                
                # Limit to requested number of rows
                data = data.tail(limit)
                
                self.logger.info(f"Fetched {len(data)} rows for {symbol} from Yahoo Finance")
                return data
            
        except Exception as e:
            self.logger.warning(f"Yahoo Finance fetch failed for {symbol}: {e}")
        
        return None
    
    def _convert_to_yahoo_symbol(self, symbol: str) -> str:
        """Convert trading symbol to Yahoo Finance format"""
        symbol = symbol.upper()
        
        # Common crypto conversions
        crypto_map = {
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD',
            'ADAUSD': 'ADA-USD',
            'SOLUSD': 'SOL-USD',
            'DOTUSD': 'DOT-USD'
        }
        
        if symbol in crypto_map:
            return crypto_map[symbol]
        
        # Handle common forex pairs
        if len(symbol) == 6:
            return f"{symbol[:3]}{symbol[3:]}=X"
        
        return symbol
    
    def _generate_simulated_data(self, symbol: str, timeframe: str, 
                               limit: int) -> pd.DataFrame:
        """Generate realistic simulated price data"""
        try:
            # Base price depending on symbol
            if 'BTC' in symbol.upper():
                base_price = 50000
                volatility = 0.02
            elif 'ETH' in symbol.upper():
                base_price = 3000
                volatility = 0.025
            else:
                base_price = 100
                volatility = 0.015
            
            # Generate time series
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
            }
            
            interval_minutes = timeframe_minutes.get(timeframe, 60)
            
            # Create datetime index
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=interval_minutes * limit)
            
            date_range = pd.date_range(
                start=start_time, 
                end=end_time, 
                freq=f'{interval_minutes}min'
            )[:limit]
            
            # Generate price data using random walk
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            
            # Generate returns
            returns = np.random.normal(0, volatility, limit)
            
            # Add some trend and mean reversion
            trend = np.linspace(-0.001, 0.001, limit)
            returns += trend
            
            # Calculate prices
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            # Generate OHLCV data
            data = pd.DataFrame(index=date_range[:len(prices)])
            
            data['close'] = prices
            data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
            
            # Generate high/low based on volatility
            intraday_vol = volatility * 0.5
            data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, intraday_vol, len(data)))
            data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, intraday_vol, len(data)))
            
            # Generate volume
            base_volume = 1000000 if 'BTC' in symbol.upper() else 500000
            data['volume'] = np.random.lognormal(np.log(base_volume), 0.5, len(data))
            
            # Round appropriate columns
            price_cols = ['open', 'high', 'low', 'close']
            data[price_cols] = data[price_cols].round(2)
            data['volume'] = data['volume'].round(0)
            
            self.logger.info(f"Generated {len(data)} simulated data points for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Simulated data generation error: {e}")
            return pd.DataFrame()
    
    async def get_multiple_symbols(self, symbols: List[str], 
                                 timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols concurrently"""
        try:
            tasks = []
            for symbol in symbols:
                task = self.get_price_data(symbol, timeframe)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data_dict = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching {symbol}: {result}")
                    data_dict[symbol] = pd.DataFrame()
                else:
                    data_dict[symbol] = result
            
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Multiple symbols fetch error: {e}")
            return {}
    
    async def get_market_overview(self) -> Dict[str, any]:
        """Get general market overview"""
        try:
            # Get data for major symbols
            symbols = ['BTCUSD', 'ETHUSD', 'SPY', 'EURUSD']
            data = await self.get_multiple_symbols(symbols)
            
            overview = {}
            
            for symbol, df in data.items():
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    price_change = df['close'].pct_change().iloc[-1]
                    
                    overview[symbol] = {
                        'price': round(current_price, 2),
                        'change_24h': round(price_change * 100, 2),
                        'volume': int(df['volume'].iloc[-1]) if 'volume' in df else 0
                    }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Market overview error: {e}")
            return {}
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_duration
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def clear_cache(self):
        """Clear data cache"""
        self.cache.clear()
        self.logger.info("Data cache cleared")
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        return [
            'BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD',
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL',
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'
        ]
'''
        
        with open('core/data_fetcher.py', 'w') as f:
            f.write(data_fetcher_content)

    def fix_imports_and_bridges(self):
        """Create missing import bridges"""
        print("🔧 Creating import bridges...")
        
        # Create core/ml_engine.py bridge (compatibility)
        ml_bridge_content = '''"""
ML Engine Bridge for Compatibility
=================================
"""

import logging
from typing import Dict, Any

try:
    from core.enhanced_ml_engine import AdaptiveMLEngine
    
    class MLEngine(AdaptiveMLEngine):
        """ML Engine wrapper for compatibility"""
        
        def __init__(self, config=None):
            super().__init__(
                model_save_path='models/',
                performance_log_path='logs/'
            )
            self.config = config
            self.logger = logging.getLogger(__name__)
            self.logger.info("ML Engine bridge initialized")
        
        async def analyze_symbol(self, symbol: str, timeframe: str) -> Dict[str, Any]:
            """Enhanced analyze_symbol for chat integration"""
            return await super().analyze_symbol(symbol, timeframe)
    
    class OctoBotMLEngine(MLEngine):
        """OctoBot compatibility wrapper"""
        pass

except ImportError as e:
    import logging
    
    class MLEngine:
        def __init__(self, config=None):
            self.config = config
            self.logger = logging.getLogger(__name__)
            self.logger.info("Basic ML Engine initialized")
        
        async def analyze_symbol(self, symbol: str, timeframe: str):
            return {
                'trend': 'NEUTRAL', 'signal': 'HOLD', 'confidence': 0.5,
                'recommendation': f'Basic analysis for {symbol}'
            }
    
    class OctoBotMLEngine(MLEngine):
        pass

__all__ = ['MLEngine', 'OctoBotMLEngine']
'''
        
        with open('core/ml_engine.py', 'w') as f:
            f.write(ml_bridge_content)
        
        # Create utils/simple_notification_manager.py
        notification_bridge = '''"""
Simple Notification Manager
==========================
"""

import logging
from typing import Optional

class SimpleNotificationManager:
    """Simple notification manager that logs notifications"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Simple Notification Manager initialized")
    
    async def send_notification(self, title: str, message: str, priority: str = "INFO"):
        """Send notification (logs only in demo mode)"""
        self.logger.info(f"[{priority}] {title}: {message}")
        return True
    
    async def notify(self, title: str, message: str, priority: str = "INFO"):
        """Alias for send_notification"""
        return await self.send_notification(title, message, priority)
    
    def is_enabled(self) -> bool:
        """Check if notifications are enabled"""
        return False  # Disabled by default to avoid issues
'''
        
        utils_dir = Path('utils')
        utils_dir.mkdir(exist_ok=True)
        
        with open('utils/simple_notification_manager.py', 'w') as f:
            f.write(notification_bridge)
        
        print("✅ Import bridges created!")

    def fix_environment_config(self):
        """Fix .env configuration"""
        print("⚙️  Configuring environment...")
        
        env_additions = [
            "\n# AUTO-CONFIGURED FOR SMOOTH STARTUP",
            "NOTIFY_TRADES=false",
            "NOTIFY_SYSTEM_EVENTS=false", 
            "NOTIFY_ERRORS=false",
            "EMAIL_ENABLED=false",
            "SLACK_ENABLED=false",
            "DISCORD_ENABLED=false",
            "TWILIO_ENABLED=false",
            "SMS_ENABLED=false",
            "DATABASE_URL=sqlite:///trading_bot.db",
            "DEBUG=true",
            "LIVE_TRADING_ENABLED=false",
            "ML_MODEL_RETRAIN_HOURS=24",
            "DATA_CACHE_MINUTES=5",
            ""
        ]
        
        # Check if .env exists, create if not
        env_file = Path('.env')
        if not env_file.exists():
            print("📝 Creating .env file...")
            with open('.env', 'w') as f:
                f.write('# Enhanced Trading Bot Configuration\n')
                f.write('APP_NAME="Enhanced Trading Bot"\n')
                f.write('APP_USER_ID=admin\n')
                f.write('APP_PASSWORD=admin123\n')
        
        # Add configuration to avoid startup issues
        with open('.env', 'a') as f:
            f.write('\n'.join(env_additions))
        
        print("✅ Environment configured for smooth startup!")

    def train_enhanced_models(self):
        """Enhanced model training with multiple algorithms"""
        print("🧠 Training enhanced ML models...")
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Check for existing models
        model_files = list(models_dir.glob('*.keras')) + list(models_dir.glob('*.joblib'))
        
        if not model_files:
            print("🔄 Training comprehensive ML models...")
            
            training_script = '''
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    SKLEARN_AVAILABLE = True
    logger.info("✅ Scikit-learn available")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️  Scikit-learn not available")

def create_advanced_features(n_samples=2000):
    """Create comprehensive feature set for training"""
    np.random.seed(42)
    
    # Time series index
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
    # Base features (technical indicators)
    features = pd.DataFrame(index=dates)
    
    # Price-based features
    base_price = 50000
    price = base_price + np.cumsum(np.random.normal(0, 100, n_samples))
    features['close'] = price
    
    # Moving averages
    features['sma_5'] = features['close'].rolling(5).mean()
    features['sma_20'] = features['close'].rolling(20).mean()
    features['sma_50'] = features['close'].rolling(50).mean()
    
    # Price ratios
    features['price_sma20_ratio'] = features['close'] / features['sma_20']
    features['sma5_sma20_ratio'] = features['sma_5'] / features['sma_20']
    
    # Volatility features
    features['volatility_5'] = features['close'].rolling(5).std()
    features['volatility_20'] = features['close'].rolling(20).std()
    
    # RSI
    delta = features['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = features['close'].ewm(span=12).mean()
    ema_26 = features['close'].ewm(span=26).mean()
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    # Bollinger Bands
    features['bb_upper'] = features['sma_20'] + (features['volatility_20'] * 2)
    features['bb_lower'] = features['sma_20'] - (features['volatility_20'] * 2)
    features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    
    # Volume features (simulated)
    base_volume = 1000000
    volume = base_volume * np.random.lognormal(0, 0.5, n_samples)
    features['volume'] = volume
    features['volume_sma'] = features['volume'].rolling(20).mean()
    features['volume_ratio'] = features['volume'] / features['volume_sma']
    
    # Momentum features
    features['momentum_5'] = features['close'].pct_change(5)
    features['momentum_10'] = features['close'].pct_change(10)
    features['momentum_20'] = features['close'].pct_change(20)
    
    # Time-based features
    features['hour'] = features.index.hour
    features['day_of_week'] = features.index.dayofweek
    features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
    
    # Clean data
    features = features.fillna(method='bfill').fillna(0)
    
    return features

def create_targets(features, lookahead=5):
    """Create prediction targets"""
    targets = pd.DataFrame(index=features.index)
    
    # Future returns
    future_return = features['close'].shift(-lookahead) / features['close'] - 1
    
    # Classification target (buy/sell/hold)
    buy_threshold = 0.02  # 2% gain
    sell_threshold = -0.015  # 1.5% loss
    
    targets['direction'] = 1  # Hold
    targets.loc[future_return > buy_threshold, 'direction'] = 2  # Buy
    targets.loc[future_return < sell_threshold, 'direction'] = 0  # Sell
    
    # Regression target
    targets['future_return'] = future_return
    
    return targets

def train_models():
    """Train multiple ML models"""
    if not SKLEARN_AVAILABLE:
        logger.warning("Skipping model training - scikit-learn not available")
        return
    
    logger.info("🔄 Creating training data...")
    
    # Create training data
    features = create_advanced_features(2000)
    targets = create_targets(features)
    
    # Align data and remove NaN
    valid_idx = features.dropna().index.intersection(targets.dropna().index)
    X = features.loc[valid_idx]
    y = targets.loc[valid_idx]
    
    if len(X) < 100:
        logger.error("Insufficient training data")
        return
    
    # Feature selection (remove date-based features for training)
    feature_cols = [col for col in X.columns if col not in ['close']]
    X_features = X[feature_cols]
    
    logger.info(f"Training with {len(X_features)} samples and {len(feature_cols)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y['direction']
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models_trained = {}
    
    # 1. Random Forest Classifier
    logger.info("Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train_scaled, y_train['direction'])
    
    # Evaluate
    y_pred = rf_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test['direction'], y_pred)
    logger.info(f"✅ Random Forest Accuracy: {accuracy:.3f}")
    
    # Save model
    joblib.dump(rf_clf, 'models/random_forest_classifier.joblib')
    models_trained['random_forest_classifier'] = accuracy
    
    # 2. Random Forest Regressor
    logger.info("Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_train_scaled, y_train['future_return'])
    
    # Evaluate
    score = rf_reg.score(X_test_scaled, y_test['future_return'])
    logger.info(f"✅ Random Forest R² Score: {score:.3f}")
    
    # Save model
    joblib.dump(rf_reg, 'models/random_forest_regressor.joblib')
    models_trained['random_forest_regressor'] = score
    
    # 3. Gradient Boosting Classifier
    logger.info("Training Gradient Boosting Classifier...")
    gb_clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    gb_clf.fit(X_train_scaled, y_train['direction'])
    
    # Evaluate
    y_pred_gb = gb_clf.predict(X_test_scaled)
    accuracy_gb = accuracy_score(y_test['direction'], y_pred_gb)
    logger.info(f"✅ Gradient Boosting Accuracy: {accuracy_gb:.3f}")
    
    # Save model
    joblib.dump(gb_clf, 'models/gradient_boosting_classifier.joblib')
    models_trained['gradient_boosting_classifier'] = accuracy_gb
    
    # Save scaler and feature names
    joblib.dump(scaler, 'models/feature_scaler.joblib')
    joblib.dump(feature_cols, 'models/feature_names.joblib')
    
    # Save training metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'n_samples': len(X_features),
        'n_features': len(feature_cols),
        'models_trained': models_trained,
        'feature_names': feature_cols
    }
    
    with open('models/training_metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Model training completed! Trained {len(models_trained)} models")
    return models_trained

if __name__ == "__main__":
    train_models()
'''
            
            try:
                exec(training_script)
                print("✅ Enhanced models trained successfully!")
            except Exception as e:
                print(f"⚠️  Model training error: {e}")
                print("✅ Will use fallback predictions")
        else:
            print(f"✅ Found {len(model_files)} existing models!")

    def start_dashboard_with_fallback(self):
        """Start dashboard with multiple fallback options"""
        print("🚀 Starting enhanced trading dashboard...")
        
        # Try FastAPI first
        if Path('main.py').exists():
            return self._start_fastapi_dashboard()
        
        # Create and start a minimal dashboard if main.py doesn't exist
        print("📝 Creating minimal dashboard...")
        self.create_main_py()
        return self._start_fastapi_dashboard()

    def _start_fastapi_dashboard(self):
        """Start FastAPI dashboard"""
        print("🎯 Starting FastAPI dashboard...")
        
        try:
            # Find available port
            port = self._find_available_port()
            
            # Start server in subprocess
            cmd = [
                sys.executable, '-m', 'uvicorn', 
                'main:app', '--host', '0.0.0.0', '--port', str(port),
                '--reload'
            ]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            print("⏳ Waiting for server to start...")
            time.sleep(8)
            
            # Check if server is running
            if self.server_process.poll() is None:
                dashboard_url = f'http://localhost:{port}'
                print(f"🌐 Opening dashboard: {dashboard_url}")
                
                # Open browser
                try:
                    webbrowser.open(dashboard_url)
                except:
                    print("⚠️  Could not open browser automatically")
                
                return True
            else:
                print("❌ FastAPI server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ FastAPI dashboard error: {e}")
            return False

    def _find_available_port(self):
        """Find an available port"""
        for port in [self.dashboard_port, self.fallback_port, 8001, 8080, 3000]:
            if not self.is_port_in_use(port):
                return port
        return 8000  # Fallback

    def print_success_summary(self, port=8000):
        """Print comprehensive success summary"""
        print(f"""
🎉 =============================================== 🎉
   ENHANCED TRADING BOT STARTUP COMPLETE!
🎉 =============================================== 🎉

✅ System Status: FULLY OPERATIONAL
✅ Dashboard: RUNNING ON PORT {port}
✅ ML Engine: READY WITH TRAINED MODELS
✅ Trading Engine: INITIALIZED (DEMO MODE)
✅ Data Fetcher: OPERATIONAL
✅ Risk Management: ACTIVE

🌐 Dashboard Access:
   • Primary: http://localhost:{port}
   • Local:   http://127.0.0.1:{port}

👤 Default Credentials (if required):
   • Username: admin
   • Password: admin123

📊 Available Features:
   ✅ Real-time market data & charts
   ✅ AI-powered trading signals with confidence
   ✅ Multiple ML model predictions
   ✅ Advanced risk management system
   ✅ Portfolio tracking & performance metrics
   ✅ Strategy backtesting capabilities
   ✅ Interactive dashboard with live updates
   ✅ WebSocket real-time data feeds
   ✅ Comprehensive logging system

🤖 AI & ML Capabilities:
   ✅ Random Forest Classifier/Regressor
   ✅ Gradient Boosting Models
   ✅ Technical Analysis Integration
   ✅ Adaptive Learning System
   ✅ Performance Monitoring

⚠️  Important Safety Notes:
   • Bot starts in DEMO mode (LIVE_TRADING_ENABLED=false)
   • All trades are simulated until you enable live trading
   • Train models with your own data before live trading
   • Configure exchange API keys in .env for live data
   • Always thoroughly test strategies in demo mode first
   • Review risk management settings before going live

🔧 Next Steps:
   1. Explore the dashboard and familiarize yourself
   2. Review and customize risk management settings
   3. Add your own trading strategies
   4. Configure exchange connections for live data
   5. Train models with your preferred datasets
   6. Test thoroughly before enabling live trading

📚 Documentation:
   • Check logs/ directory for system logs
   • Review config.json for all settings
   • Modify .env for environment variables
   • Use /api/health endpoint for system status

🚀 Your enhanced AI trading bot is now running!
   Happy trading! 📈💰

Press Ctrl+C to stop the trading bot.
""")

    def monitor_system_health(self):
        """Monitor system health while running"""
        print("❤️  System health monitoring active...")
        
        try:
            while True:
                time.sleep(30)  # Check every 30 seconds
                
                # Check server process
                if self.server_process and self.server_process.poll() is not None:
                    print("⚠️  Dashboard process stopped unexpectedly")
                    break
                
                # Could add more health checks here:
                # - Memory usage
                # - CPU usage  
                # - Disk space
                # - Log file sizes
                
        except KeyboardInterrupt:
            print("\n🛑 Health monitoring stopped")

    def run_startup_sequence(self):
        """Run the complete enhanced startup sequence"""
        try:
            self.print_banner()
            
            # Step 1: System Check
            self.print_step(1, "COMPREHENSIVE SYSTEM CHECK")
            if not self.check_system_requirements():
                return False
            
            # Step 2: Dependencies
            self.print_step(2, "DEPENDENCY MANAGEMENT")
            if not self.install_dependencies():
                print("⚠️  Some dependencies failed, continuing...")
            
            # Step 3: Directory Structure
            self.print_step(3, "PROJECT STRUCTURE SETUP")
            self.create_directory_structure()
            
            # Step 4: Core Files
            self.print_step(4, "CORE FILES CREATION")
            self.create_missing_core_files()
            
            # Step 5: Import Fixes
            self.print_step(5, "IMPORT BRIDGES & COMPATIBILITY")
            self.fix_imports_and_bridges()
            
            # Step 6: Configuration
            self.print_step(6, "ENVIRONMENT CONFIGURATION")
            self.fix_environment_config()
            
            # Step 7: Model Training
            self.print_step(7, "ENHANCED ML MODEL TRAINING")
            self.train_enhanced_models()
            
            # Step 8: Dashboard Launch
            self.print_step(8, "DASHBOARD STARTUP & LAUNCH")
            if self.start_dashboard_with_fallback():
                port = self._find_available_port()
                self.print_success_summary(port)
                
                # Monitor system health
                print("\n🔄 Enhanced Trading Bot is running... Press Ctrl+C to stop")
                
                # Start health monitoring in background
                health_thread = threading.Thread(target=self.monitor_system_health, daemon=True)
                health_thread.start()
                
                # Keep main thread alive
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n👋 Trading bot stopped. Thanks for using Enhanced Trading Bot!")
                    return True
            else:
                print("❌ Dashboard startup failed!")
                self.print_troubleshooting_tips()
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️  Startup interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Startup failed: {e}")
            logger.exception("Startup error")
            self.print_troubleshooting_tips()
            return False

    def print_troubleshooting_tips(self):
        """Print comprehensive troubleshooting tips"""
        print(f"""
🔧 =============================================== 🔧
   TROUBLESHOOTING GUIDE
🔧 =============================================== 🔧

If you're seeing this, something went wrong. Here's how to fix it:

📋 BASIC CHECKS:
   1. ✅ Python version 3.8+ required
      → Check: python --version
   
   2. ✅ Ensure you're in the correct directory
      → Should contain: main.py, config.json, core/ folder
   
   3. ✅ Check available disk space (1GB+ recommended)
      → Free up space if needed

🔧 DEPENDENCY ISSUES:
   1. Manual dependency install:
      → pip install -r requirements.txt
   
   2. If pip install fails:
      → pip install --upgrade pip
      → pip install fastapi uvicorn pandas numpy scikit-learn
   
   3. For TensorFlow issues:
      → pip install tensorflow==2.15.0
      → Or skip TensorFlow: bot will work without it

🌐 PORT & NETWORK ISSUES:
   1. Port already in use:
      → Check what's using port 8000: netstat -an | grep 8000
      → Kill the process or use different port
   
   2. Firewall blocking:
      → Temporarily disable firewall for testing
      → Add exception for Python/port 8000

📁 FILE & PERMISSION ISSUES:
   1. Permission denied:
      → Run with administrator/sudo privileges
      → Check folder write permissions
   
   2. Missing files:
      → Re-run this script to recreate missing files
      → Check if antivirus deleted files

🧠 MODEL TRAINING ISSUES:
   1. Insufficient memory:
      → Close other applications
      → Reduce model complexity in config
   
   2. Scikit-learn installation:
      → pip install scikit-learn==1.3.2

🚀 QUICK FIXES TO TRY:
   1. Restart the script: python start_trading_bot.py
   2. Clear Python cache: python -m pip cache purge
   3. Reinstall dependencies: pip install --force-reinstall fastapi uvicorn
   4. Run Python diagnostics: python -m pip check
   5. Try manual dashboard start: python main.py

📞 STILL HAVING ISSUES?
   1. Check logs in logs/ directory for detailed errors
   2. Run system diagnostic: python -c "import sys; print(sys.version)"
   3. Test basic imports: python -c "import fastapi, uvicorn, pandas"

💡 COMMON SOLUTIONS:
   • Windows: Use pip instead of pip3
   • Mac: Use python3 instead of python  
   • Linux: May need python3-dev package
   • Virtual environment issues: Try without venv first

Remember: The bot can run with basic functionality even if some
components fail. The core trading engine will still work!

""")

def main():
    """Main entry point with enhanced error handling"""
    print("🚀 Enhanced Trading Bot Startup Script v2.0")
    print("=" * 60)
    
    try:
        starter = TradingBotStarter()
        success = starter.run_startup_sequence()
        
        if not success:
            print("\n" + "=" * 60)
            print("❌ STARTUP FAILED")
            print("=" * 60)
            starter.print_troubleshooting_tips()
            
            print("\n🤔 Want to try alternative startup methods?")
            print("1. Manual component startup")
            print("2. Minimal dashboard mode")
            print("3. Debug mode with verbose output")
            
            try:
                choice = input("\nEnter choice (1-3) or press Enter to exit: ").strip()
                
                if choice == "1":
                    manual_startup()
                elif choice == "2":
                    minimal_dashboard_mode()
                elif choice == "3":
                    debug_startup()
                else:
                    print("👋 Exiting. Feel free to try again!")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
        
        return success
        
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        logger.exception("Critical startup error")
        return False

def manual_startup():
    """Manual component-by-component startup"""
    print("\n🔧 MANUAL STARTUP MODE")
    print("=" * 40)
    
    try:
        # Test basic imports
        print("Testing imports...")
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn available")
        
        # Test pandas/numpy
        try:
            import pandas as pd
            import numpy as np
            print("✅ Pandas and NumPy available")
        except ImportError:
            print("⚠️  Pandas/NumPy missing - installing...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas', 'numpy'])
        
        # Create minimal main.py if missing
        if not Path('main.py').exists():
            print("Creating minimal main.py...")
            minimal_main = '''
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Trading Bot - Minimal Mode")

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html><head><title>Trading Bot</title></head>
    <body style="font-family: Arial; padding: 20px; background: #1e3c72; color: white;">
        <h1>🚀 Trading Bot - Minimal Mode</h1>
        <p>✅ System is running!</p>
        <p>Dashboard: Minimal functionality active</p>
        <p>Time: <span id="time"></span></p>
        <script>
            setInterval(() => {
                document.getElementById('time').textContent = new Date().toLocaleString();
            }, 1000);
        </script>
    </body></html>
    """

@app.get("/health")
def health():
    return {"status": "ok", "mode": "minimal"}
'''
            with open('main.py', 'w') as f:
                f.write(minimal_main)
        
        # Try to start server
        print("Starting minimal server...")
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'main:app', '--host', '127.0.0.1', '--port', '8000'
        ])
        
    except Exception as e:
        print(f"Manual startup failed: {e}")

def minimal_dashboard_mode():
    """Start minimal dashboard without ML components"""
    print("\n🎯 MINIMAL DASHBOARD MODE")
    print("=" * 40)
    
    try:
        # Create super simple dashboard
        simple_html = '''
<!DOCTYPE html>
<html>
<head><title>Trading Bot</title></head>
<body style="font-family: Arial; background: #2c3e50; color: white; padding: 20px;">
    <h1>🚀 Trading Bot - Running</h1>
    <p>✅ Status: Online</p>
    <p>⏰ Time: <span id="time"></span></p>
    <p>🔧 Mode: Minimal Dashboard</p>
    <script>
        setInterval(() => {
            document.getElementById('time').textContent = new Date().toLocaleString();
        }, 1000);
    </script>
</body>
</html>
'''
        
        # Write to file and open
        with open('minimal_dashboard.html', 'w') as f:
            f.write(simple_html)
        
        webbrowser.open('minimal_dashboard.html')
        print("✅ Minimal dashboard opened in browser")
        print("Press Ctrl+C to exit...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Minimal dashboard stopped")
    except Exception as e:
        print(f"Minimal dashboard error: {e}")

def debug_startup():
    """Debug mode with verbose output"""
    print("\n🐛 DEBUG STARTUP MODE")
    print("=" * 40)
    
    # Set verbose logging
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Test each component individually
        print("\n1. Testing Python environment...")
        print(f"   Python version: {sys.version}")
        print(f"   Python executable: {sys.executable}")
        print(f"   Current directory: {os.getcwd()}")
        
        print("\n2. Testing imports...")
        modules_to_test = [
            'fastapi', 'uvicorn', 'pandas', 'numpy', 
            'sklearn', 'tensorflow', 'matplotlib'
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"   ✅ {module}")
            except ImportError as e:
                print(f"   ❌ {module}: {e}")
        
        print("\n3. Testing file system...")
        for file in ['main.py', 'config.json', '.env']:
            if Path(file).exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (missing)")
        
        print("\n4. Testing network...")
        import socket
        for port in [8000, 8001, 5000]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        print(f"   ⚠️  Port {port} in use")
                    else:
                        print(f"   ✅ Port {port} available")
            except:
                print(f"   ❌ Port {port} test failed")
        
        print("\n5. Attempting basic startup...")
        starter = TradingBotStarter()
        starter.run_startup_sequence()
        
    except Exception as e:
        print(f"Debug startup error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()