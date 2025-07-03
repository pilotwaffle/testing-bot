"""
404 Dashboard Error Diagnostic & Fix Script
Helps identify and resolve dashboard 404 issues
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

def check_server_status():
    """Check if the server is running"""
    print("🔍 Checking Server Status")
    print("=" * 50)
    
    urls_to_check = [
        "http://localhost:8000",
        "http://localhost:8000/",
        "http://localhost:8000/dashboard", 
        "http://localhost:8000/health",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/dashboard"
    ]
    
    for url in urls_to_check:
        try:
            response = requests.get(url, timeout=5)
            print(f"✅ {url} -> Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   📄 Content length: {len(response.text)} chars")
                return True
        except requests.exceptions.ConnectionError:
            print(f"❌ {url} -> Connection refused (server not running)")
        except requests.exceptions.Timeout:
            print(f"⏰ {url} -> Timeout")
        except Exception as e:
            print(f"❓ {url} -> Error: {e}")
    
    return False

def check_main_file():
    """Check main.py for dashboard routes"""
    print("\n🔍 Checking Main Application File")
    print("=" * 50)
    
    main_files = ["main.py", "app.py", "server.py", "run.py"]
    found_main = None
    
    for main_file in main_files:
        if Path(main_file).exists():
            found_main = main_file
            print(f"✅ Found: {main_file}")
            
            try:
                with open(main_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common dashboard routes
                routes_found = []
                route_patterns = [
                    '@app.get("/")',
                    '@app.get("/dashboard")', 
                    'def dashboard',
                    'def index',
                    'def root',
                    'return templates.TemplateResponse',
                    'dashboard.html'
                ]
                
                for pattern in route_patterns:
                    if pattern in content:
                        routes_found.append(pattern)
                
                if routes_found:
                    print(f"   📍 Routes found: {', '.join(routes_found)}")
                else:
                    print(f"   ❌ No dashboard routes found in {main_file}")
                
                # Check for FastAPI app initialization
                if "FastAPI" in content:
                    print(f"   ✅ FastAPI app detected")
                elif "Flask" in content:
                    print(f"   ✅ Flask app detected")
                else:
                    print(f"   ❓ App framework unclear")
                    
            except Exception as e:
                print(f"   ❌ Error reading {main_file}: {e}")
            
            break
    
    if not found_main:
        print("❌ No main application file found!")
        print("   Looking for: main.py, app.py, server.py, run.py")
    
    return found_main

def check_templates():
    """Check if template files exist"""
    print("\n🔍 Checking Template Files")
    print("=" * 50)
    
    template_dirs = ["templates", "static/templates", "app/templates"]
    dashboard_files = ["dashboard.html", "index.html"]
    
    found_templates = []
    
    for template_dir in template_dirs:
        if Path(template_dir).exists():
            print(f"✅ Template directory found: {template_dir}")
            
            for dashboard_file in dashboard_files:
                template_path = Path(template_dir) / dashboard_file
                if template_path.exists():
                    found_templates.append(str(template_path))
                    print(f"   ✅ Template found: {template_path}")
                    
                    # Check template content
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if "ml_status" in content:
                                print(f"      📊 Contains ml_status variable")
                            if "Trading Bot" in content:
                                print(f"      🤖 Trading bot template confirmed")
                    except Exception as e:
                        print(f"      ❌ Error reading template: {e}")
        else:
            print(f"❌ Template directory missing: {template_dir}")
    
    return found_templates

def provide_solutions(server_running, main_file, templates_found):
    """Provide solutions based on diagnostic results"""
    print("\n💡 Solutions")
    print("=" * 50)
    
    if not server_running:
        print("🔧 Server is not running - START THE SERVER:")
        print("   Method 1: Use safe startup script")
        print("   → start_safe.bat  (Windows)")
        print("   → ./start_safe.sh  (Linux/Mac)")
        print()
        print("   Method 2: Manual startup")
        print("   → python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        print()
        print("   Method 3: If using different file")
        if main_file and main_file != "main.py":
            app_name = main_file.replace('.py', '')
            print(f"   → python -m uvicorn {app_name}:app --host 0.0.0.0 --port 8000 --reload")
        print()
        
    elif not main_file:
        print("🔧 No main application file found:")
        print("   → Create main.py with FastAPI app")
        print("   → Ensure it has dashboard routes")
        
    elif not templates_found:
        print("🔧 No dashboard template found:")
        print("   → Create templates/dashboard.html")
        print("   → Ensure it's in the correct directory")
        
    else:
        print("🔧 Server running but dashboard not accessible:")
        print("   → Check route configuration in main.py")
        print("   → Verify template path in route handler")
        print("   → Check for syntax errors in main.py")

def create_minimal_dashboard_route():
    """Create a minimal dashboard route for testing"""
    print("\n🛠️ Creating Minimal Dashboard Route")
    print("=" * 50)
    
    minimal_route = '''
# Add this to your main.py file

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def dashboard(request: Request):
    """Main dashboard route"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "status": "RUNNING",
        "ml_status": {},  # Empty for now
        "metrics": {
            "total_value": 100000.00,
            "cash_balance": 100000.00,
            "unrealized_pnl": 0.00,
            "total_profit": 0.00,
            "num_positions": 0
        },
        "active_strategies": [],
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "ai_enabled": False
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Server is running"}
'''
    
    print("Copy and paste this minimal route into your main.py:")
    print(minimal_route)

def main():
    """Main diagnostic function"""
    print("🔍 Dashboard 404 Error Diagnostic")
    print("=" * 50)
    
    # Run checks
    server_running = check_server_status()
    main_file = check_main_file()
    templates_found = check_templates()
    
    # Provide solutions
    provide_solutions(server_running, main_file, templates_found)
    
    # Show minimal route if needed
    if not server_running or not templates_found:
        create_minimal_dashboard_route()
    
    print("\n📋 Next Steps:")
    print("1. Fix the issues identified above")
    print("2. Start/restart the server")
    print("3. Try accessing http://localhost:8000")
    print("4. Check browser console for any remaining errors")

if __name__ == "__main__":
    main()