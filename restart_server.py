#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\restart_server.py
Location: E:\Trade Chat Bot\G Trading Bot\restart_server.py

🔄 Server restart utility
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

def kill_existing_server():
    """Kill any existing server processes"""
    print("🔍 Checking for existing server processes...")
    
    try:
        # Kill any Python processes running main.py
        if os.name == 'nt':  # Windows
            os.system('taskkill /f /im python.exe 2>nul')
        else:  # Unix/Linux/Mac
            os.system('pkill -f "python.*main.py"')
        
        time.sleep(2)
        print("✅ Existing processes terminated")
    except Exception as e:
        print(f"⚠️ Could not kill existing processes: {e}")

def start_server():
    """Start the server"""
    print("🚀 Starting server...")
    
    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, 'main.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(5)
        
        # Check if server is responding
        for attempt in range(5):
            try:
                response = requests.get('http://localhost:8000/health', timeout=2)
                if response.status_code == 200:
                    print("✅ Server started successfully!")
                    return True
            except:
                print(f"⏳ Waiting for server... (attempt {attempt + 1}/5)")
                time.sleep(2)
        
        print("❌ Server failed to start properly")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def main():
    print("🔄 Elite Trading Bot V3.0 - Server Restart Utility")
    print("=" * 50)
    
    kill_existing_server()
    
    if start_server():
        print("\n🎉 Server restart complete!")
        print("🌐 Dashboard: http://localhost:8000")
        print("🧪 Run tests: python enhanced_dashboard_tester.py")
    else:
        print("\n❌ Server restart failed")
        print("💡 Try manually: python main.py")

if __name__ == "__main__":
    main()
