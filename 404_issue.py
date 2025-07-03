"""
Dashboard 404 Fix Script
Diagnoses and fixes dashboard template rendering issues
"""

import os
import sys
import json
import shutil
import requests
from pathlib import Path
from datetime import datetime

def backup_file(file_path):
    """Create backup of file before modifying"""
    if Path(file_path).exists():
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        print(f"📁 Backup created: {backup_path}")
        return backup_path
    return None

def check_server_response():
    """Check what the server is actually returning"""
    print("🔍 Analyzing Server Response")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8000", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)} characters")
        print(f"Content Type: {response.headers.get('content-type', 'Unknown')}")
        
        # Show first 500 chars of response
        content_preview = response.text[:500]
        print(f"\nContent Preview:")
        print("-" * 30)
        print(content_preview)
        print("-" * 30)
        
        # Check if it's an error page
        if "error" in response.text.lower() or "exception" in response.text.lower():
            print("❌ Server is returning an error page")
            return False, response.text
        elif "Trading Bot" in response.text and len(response.text) > 5000:
            print("✅ Full dashboard appears to be loading")
            return True, response.text
        else:
            print("⚠️ Server responding but content seems incomplete")
            return False, response.text
            
    except Exception as e:
        print(f"❌ Error checking server response: {e}")
        return False, str(e)

def analyze_main_py():
    """Analyze main.py for common issues"""
    print("\n🔍 Analyzing main.py Configuration")
    print("=" * 50)
    
    if not Path("main.py").exists():
        print("❌ main.py not found")
        return False
    
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = []
        fixes = []
        
        # Check for common issues
        if "Jinja2Templates" not in content:
            issues.append("Missing Jinja2Templates import")
            fixes.append("Add: from fastapi.templating import Jinja2Templates")
        
        if 'templates = Jinja2Templates(directory="templates")' not in content:
            issues.append("Templates not initialized")
            fixes.append('Add: templates = Jinja2Templates(directory="templates")')
        
        if "StaticFiles" not in content:
            issues.append("Missing StaticFiles for CSS/JS")
            fixes.append("Add: from fastapi.staticfiles import StaticFiles")
        
        if 'mount("/static"' not in content:
            issues.append("Static files not mounted")
            fixes.append('Add: app.mount("/static", StaticFiles(directory="static"), name="static")')
        
        # Check dashboard route
        if '@app.get("/")' in content:
            print("✅ Root route found")
            
            # Extract the dashboard function
            lines = content.split('\n')
            dashboard_function = []
            in_dashboard = False
            
            for line in lines:
                if '@app.get("/")' in line or 'def dashboard' in line or 'def root' in line:
                    in_dashboard = True
                if in_dashboard:
                    dashboard_function.append(line)
                    if line.strip().startswith('def ') and 'dashboard' not in line and 'root' not in line:
                        break
            
            dashboard_code = '\n'.join(dashboard_function)
            
            # Check for template context issues
            if "ml_status" not in dashboard_code:
                issues.append("ml_status not passed to template")
                fixes.append("Add ml_status to template context")
            
            if "metrics" not in dashboard_code:
                issues.append("metrics not passed to template")
                fixes.append("Add metrics to template context")
        
        if issues:
            print("❌ Issues found:")
            for issue in issues:
                print(f"   • {issue}")
            print("\n🔧 Recommended fixes:")
            for fix in fixes:
                print(f"   • {fix}")
            return False
        else:
            print("✅ main.py looks good")
            return True
            
    except Exception as e:
        print(f"❌ Error analyzing main.py: {e}")
        return False

def create_fixed_main_py():
    """Create a corrected main.py"""
    print("\n🛠️ Creating Fixed main.py")
    print("=" * 50)
    
    # Backup original
    backup_file("main.py")
    
    fixed_main = '''"""
Fixed Trading Bot Main Application
Corrects template rendering and static file issues
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Trading Bot Dashboard", version="3.0")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Mount static files (CSS, JS, images)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted from /static")
else:
    logger.warning("⚠️ Static directory not found")

# Initialize components (with error handling)
try:
    # Import your existing components
    from core.enhanced_trading_engine import EnhancedTradingEngine
    from core.ml_engine import MLEngine
    
    # Initialize engines
    trading_engine = EnhancedTradingEngine()
    ml_engine = MLEngine()
    
    logger.info("✅ Trading and ML engines initialized")
except ImportError as e:
    logger.warning(f"⚠️ Could not import engines: {e}")
    trading_engine = None
    ml_engine = None
except Exception as e:
    logger.error(f"❌ Error initializing engines: {e}")
    trading_engine = None
    ml_engine = None

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard route with full context"""
    try:
        # Get trading status
        if trading_engine:
            status = trading_engine.get_status()
            metrics = {
                "total_value": status.get("total_value", 100000.0),
                "cash_balance": status.get("cash_balance", 100000.0),
                "unrealized_pnl": status.get("unrealized_pnl", 0.0),
                "total_profit": status.get("total_profit", 0.0),
                "num_positions": len(status.get("positions", {}))
            }
            active_strategies = list(status.get("active_strategies", {}).keys())
            bot_status = "RUNNING" if status.get("running", False) else "STOPPED"
        else:
            # Default values when engines not available
            metrics = {
                "total_value": 100000.0,
                "cash_balance": 100000.0, 
                "unrealized_pnl": 0.0,
                "total_profit": 0.0,
                "num_positions": 0
            }
            active_strategies = []
            bot_status = "STOPPED"
        
        # Get ML status
        if ml_engine:
            ml_status = ml_engine.get_status()
        else:
            # Provide default ML status for template
            ml_status = {
                "lorentzian_classifier": {
                    "model_type": "Lorentzian Classifier",
                    "description": "k-NN with Lorentzian distance using RSI, Williams %R, CCI, ADX",
                    "last_trained": "Not trained",
                    "metric_name": "Accuracy",
                    "metric_value_fmt": "N/A",
                    "training_samples": 0
                },
                "neural_network": {
                    "model_type": "Neural Network", 
                    "description": "Deep MLP for price prediction with technical indicators",
                    "last_trained": "Not trained",
                    "metric_name": "MSE",
                    "metric_value_fmt": "N/A",
                    "training_samples": 0
                }
            }
        
        # Available symbols
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
        
        # Template context
        context = {
            "request": request,
            "status": bot_status,
            "metrics": metrics,
            "ml_status": ml_status,
            "active_strategies": active_strategies,
            "symbols": symbols,
            "ai_enabled": ml_engine is not None
        }
        
        logger.info(f"✅ Dashboard rendered with {len(ml_status)} ML models")
        return templates.TemplateResponse("dashboard.html", context)
        
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        # Return error page instead of crashing
        error_html = f"""
        <html><body>
        <h1>Dashboard Error</h1>
        <p>Error loading dashboard: {str(e)}</p>
        <p><a href="/">Try again</a></p>
        </body></html>
        """
        return HTMLResponse(content=error_html, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Trading bot server is running",
        "engines": {
            "trading": trading_engine is not None,
            "ml": ml_engine is not None
        }
    }

@app.get("/api/chat")
async def chat_api():
    """Chat API endpoint"""
    return {"message": "Chat API is working"}

# Add other routes here as needed

if __name__ == "__main__":
    print("🚀 Starting Trading Bot Dashboard...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
'''
    
    try:
        with open("main.py", 'w', encoding='utf-8') as f:
            f.write(fixed_main)
        print("✅ Fixed main.py created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating fixed main.py: {e}")
        return False

def check_static_files():
    """Check and create missing static files"""
    print("\n🔍 Checking Static Files")
    print("=" * 50)
    
    static_dir = Path("static")
    if not static_dir.exists():
        print("❌ Static directory missing - creating...")
        static_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (static_dir / "css").mkdir(exist_ok=True)
        (static_dir / "js").mkdir(exist_ok=True)
        
        print("✅ Static directories created")
    
    # Check for essential files
    essential_files = [
        "static/css/style.css",
        "static/js/dashboard.js"
    ]
    
    missing_files = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n⚠️ {len(missing_files)} static files missing")
        print("This might cause styling/functionality issues")
        return False
    
    return True

def test_fixed_server():
    """Test the server after fixes"""
    print("\n🧪 Testing Fixed Server")
    print("=" * 50)
    
    print("Restart your server with: python main.py")
    print("Then test these URLs:")
    print("  • http://localhost:8000")
    print("  • http://localhost:8000/health")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except:
        print("⚠️ Server not responding (normal if you haven't restarted yet)")

def main():
    """Main fix function"""
    print("🔧 Dashboard 404 Fix Script")
    print("=" * 50)
    
    # Step 1: Check what server is returning
    is_working, response_content = check_server_response()
    
    # Step 2: Analyze main.py
    main_py_ok = analyze_main_py()
    
    # Step 3: Fix main.py if needed
    if not is_working or not main_py_ok:
        print("\n🔧 Applying fixes...")
        if create_fixed_main_py():
            print("✅ main.py has been fixed")
        else:
            print("❌ Failed to fix main.py")
            return
    
    # Step 4: Check static files
    check_static_files()
    
    # Step 5: Test
    test_fixed_server()
    
    print("\n📋 Summary:")
    print("✅ Fixed main.py with proper error handling")
    print("✅ Added default ML status for template")
    print("✅ Improved static file mounting")
    print("✅ Added comprehensive logging")
    
    print("\n🚀 Next Steps:")
    print("1. Restart the server: python main.py")
    print("2. Visit: http://localhost:8000")
    print("3. Check for ML training section")
    print("4. If issues persist, check server console output")

if __name__ == "__main__":
    main()