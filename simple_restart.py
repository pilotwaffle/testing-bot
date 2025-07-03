"""
File: simple_restart.py
Location: E:\Trade Chat Bot\G Trading Bot\simple_restart.py

Simple Server Restart Script
Clean restart of the trading bot server
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_port_free(port=8000):
    """Check if port is actually free by trying to connect"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        print(f"❌ Port {port} still has a server responding")
        return False
    except requests.exceptions.ConnectionError:
        print(f"✅ Port {port} is free - no server responding")
        return True
    except Exception as e:
        print(f"✅ Port {port} appears free (connection error: {e})")
        return True

def start_server(port=8000):
    """Start the server on specified port"""
    print(f"🚀 Starting Trading Bot Server on Port {port}")
    print("=" * 50)
    
    if not Path("main.py").exists():
        print("❌ main.py not found in current directory")
        return False
    
    # Build the command
    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]
    
    print(f"🚀 Command: {' '.join(cmd)}")
    print("⏳ Starting server...")
    print("🔄 Press Ctrl+C to stop the server when needed")
    print()
    
    try:
        # Start the server (blocking call)
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main function"""
    print("🔧 Simple Server Restart")
    print("=" * 50)
    
    # Check if ports are free
    port_8000_free = check_port_free(8000)
    port_8001_free = check_port_free(8001)
    
    if port_8000_free:
        print("\n✅ Port 8000 is available - using default port")
        start_port = 8000
    elif port_8001_free:
        print("\n⚠️ Port 8000 busy, using port 8001")
        start_port = 8001
    else:
        print("\n❌ Both ports 8000 and 8001 appear busy")
        print("🔧 Try manually stopping servers or use a different port")
        start_port = 8002
    
    print(f"\n🌐 Dashboard will be available at: http://localhost:{start_port}")
    print("📊 Expected features:")
    print("   ✅ Full dashboard with ML Training section")
    print("   ✅ Lorentzian Classifier, Neural Network options")
    print("   ✅ Portfolio Performance metrics")
    print("   ✅ Chat interface")
    print()
    
    input("Press Enter to start the server (or Ctrl+C to cancel)...")
    start_server(start_port)

if __name__ == "__main__":
    main()