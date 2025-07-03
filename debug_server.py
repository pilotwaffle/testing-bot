#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\debug_server.py
Location: E:\Trade Chat Bot\G Trading Bot\debug_server.py

🔍 Debug Server Issues and Test Correct Paths
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path

def check_files_exist():
    """Check if all required files exist"""
    print("📁 Checking required files...")
    
    files_to_check = [
        "main.py",
        "static/style.css",
        "templates/dashboard.html"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✅ {file_path} - EXISTS")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_server_running():
    """Check if server is running"""
    print("\n🔍 Checking if server is running...")
    
    try:
        response = requests.get("http://localhost:8000/ping", timeout=3)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"⚠️ Server responding but with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running")
        return False
    except requests.exceptions.Timeout:
        print("⚠️ Server is running but very slow")
        return True
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False

def test_correct_css_path():
    """Test the correct CSS path"""
    print("\n🎨 Testing correct CSS path...")
    
    try:
        # Test the correct path (without /css/)
        response = requests.get("http://localhost:8000/static/style.css", timeout=5)
        if response.status_code == 200:
            print("✅ CSS file accessible at /static/style.css")
            print(f"   Content length: {len(response.text)} characters")
            return True
        else:
            print(f"❌ CSS file not accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing CSS: {e}")
        return False

def kill_existing_server():
    """Kill any existing server processes"""
    print("\n🛑 Checking for existing server processes...")
    
    try:
        import psutil
        killed_any = False
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('main.py' in arg for arg in cmdline):
                    print(f"🔫 Killing process {proc.info['pid']}: {' '.join(cmdline)}")
                    proc.kill()
                    killed_any = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if killed_any:
            print("⏳ Waiting 2 seconds for processes to terminate...")
            time.sleep(2)
        else:
            print("✅ No existing server processes found")
            
    except ImportError:
        print("⚠️ psutil not available, cannot check for existing processes")
        print("💡 If server is running, manually stop it with Ctrl+C")

def start_server():
    """Start the server"""
    print("\n🚀 Starting server...")
    
    try:
        # Start server in background
        if os.name == 'nt':  # Windows
            subprocess.Popen([sys.executable, "main.py"], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux/Mac
            subprocess.Popen([sys.executable, "main.py"])
        
        print("⏳ Waiting 5 seconds for server to start...")
        time.sleep(5)
        
        # Check if it started
        if check_server_running():
            print("✅ Server started successfully!")
            return True
        else:
            print("❌ Server failed to start or taking too long")
            return False
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive endpoint test with correct paths"""
    print("\n🧪 Running comprehensive test...")
    
    endpoints = [
        ("http://localhost:8000/health", "Health Check"),
        ("http://localhost:8000/api/market-data", "Market Data"),
        ("http://localhost:8000/api/strategies/available", "Available Strategies"),
        ("http://localhost:8000/api/strategies/active", "Active Strategies"),
        ("http://localhost:8000/api/performance", "Performance Metrics"),
        ("http://localhost:8000/ping", "Ping Test"),
        ("http://localhost:8000/static/style.css", "CSS File (Correct Path)"),  # Fixed path
        ("http://localhost:8000/", "Dashboard Page")
    ]
    
    results = []
    
    for url, name in endpoints:
        try:
            print(f"🧪 Testing {name}...")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {name} - SUCCESS")
                
                # Special handling for different content types
                if 'css' in url:
                    print(f"   📝 CSS file size: {len(response.text)} chars")
                elif url.endswith('/'):
                    print(f"   📄 HTML page loaded")
                else:
                    try:
                        data = response.json()
                        if isinstance(data, dict):
                            if 'data' in data and isinstance(data['data'], list):
                                print(f"   📊 Data items: {len(data['data'])}")
                            elif 'strategies' in data:
                                print(f"   📊 Strategies: {len(data['strategies'])}")
                            elif 'active_strategies' in data:
                                print(f"   📊 Active strategies: {len(data['active_strategies'])}")
                            else:
                                print(f"   📊 Response keys: {list(data.keys())[:3]}...")
                    except:
                        print(f"   📊 Response received")
                
                results.append((name, True))
            else:
                print(f"❌ {name} - Status: {response.status_code}")
                results.append((name, False))
                
        except requests.exceptions.Timeout:
            print(f"❌ {name} - Timeout")
            results.append((name, False))
        except requests.exceptions.ConnectionError:
            print(f"❌ {name} - Connection Error")
            results.append((name, False))
        except Exception as e:
            print(f"❌ {name} - Error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n🎯 Results: {passed}/{total} endpoints working")
    
    if passed == total:
        print("🎉 PERFECT! All endpoints working!")
    elif passed >= total - 1:
        print("🎯 Almost there! Just one more fix needed.")
    else:
        print("⚠️ Multiple issues detected. Check server logs.")
    
    return passed, total

def main():
    """Main diagnostic and fix function"""
    print("🔍 Elite Trading Bot V3.0 - Server Diagnostic")
    print("="*60)
    
    # Step 1: Check files exist
    if not check_files_exist():
        print("\n❌ Missing required files! Please run the fix scripts first.")
        return
    
    # Step 2: Kill existing server
    kill_existing_server()
    
    # Step 3: Start fresh server
    if not start_server():
        print("\n❌ Failed to start server. Please start manually:")
        print("   python main.py")
        return
    
    # Step 4: Run comprehensive test
    passed, total = run_comprehensive_test()
    
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS")
    print("="*60)
    
    if passed == total:
        print("🎉 SUCCESS! Your dashboard is fully working!")
        print("🌐 Open: http://localhost:8000")
        print("🎨 You should see beautiful styling and no console errors!")
    elif passed >= 6:
        print("🎯 Almost perfect! Just minor issues remaining.")
        print("🌐 Dashboard should be functional at: http://localhost:8000")
    else:
        print("⚠️ Multiple issues detected.")
        print("💡 Try manually restarting with: python main.py")
    
    print("\n📋 NEXT STEPS:")
    print("1. Open http://localhost:8000 in your browser")
    print("2. Press F12 → Console tab to check for errors")
    print("3. You should see a beautiful gradient dashboard!")

if __name__ == "__main__":
    main()