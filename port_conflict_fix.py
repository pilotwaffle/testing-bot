"""
File: port_conflict_fix.py
Location: E:\Trade Chat Bot\G Trading Bot\port_conflict_fix.py

Port Conflict Fix Script
Stops existing server and starts the fixed version
"""

import os
import sys
import time
import subprocess
import psutil
import requests
from pathlib import Path

def find_processes_on_port(port=8000):
    """Find processes using the specified port"""
    print(f"🔍 Checking for processes on port {port}")
    print("=" * 50)
    
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if process has network connections
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    })
                    print(f"📍 Found process: PID {proc.info['pid']} - {proc.info['name']}")
                    if 'python' in proc.info['name'].lower():
                        print(f"   Command: {' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else 'N/A'}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return processes

def kill_processes_on_port(port=8000):
    """Kill processes using the specified port"""
    print(f"\n🔪 Stopping processes on port {port}")
    print("=" * 50)
    
    processes = find_processes_on_port(port)
    
    if not processes:
        print(f"✅ No processes found on port {port}")
        return True
    
    killed_count = 0
    for proc_info in processes:
        try:
            pid = proc_info['pid']
            proc = psutil.Process(pid)
            proc_name = proc_info['name']
            
            print(f"🔪 Killing process: PID {pid} ({proc_name})")
            
            # Try graceful termination first
            proc.terminate()
            try:
                proc.wait(timeout=3)
                print(f"   ✅ Process {pid} terminated gracefully")
                killed_count += 1
            except psutil.TimeoutExpired:
                # Force kill if graceful termination fails
                print(f"   ⚡ Force killing process {pid}")
                proc.kill()
                proc.wait()
                print(f"   ✅ Process {pid} force killed")
                killed_count += 1
                
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"   ❌ Could not kill process {pid}: {e}")
    
    print(f"\n📊 Killed {killed_count} out of {len(processes)} processes")
    
    # Verify port is free
    time.sleep(1)
    remaining = find_processes_on_port(port)
    if remaining:
        print(f"⚠️ {len(remaining)} processes still running on port {port}")
        return False
    else:
        print(f"✅ Port {port} is now free")
        return True

def check_port_availability(port=8000):
    """Check if port is available"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        print(f"❌ Port {port} is still in use (server responding)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"✅ Port {port} is available")
        return True
    except Exception as e:
        print(f"🤔 Port {port} status unclear: {e}")
        return True

def start_fixed_server(port=8000):
    """Start the fixed server"""
    print(f"\n🚀 Starting Fixed Server on Port {port}")
    print("=" * 50)
    
    if not Path("main.py").exists():
        print("❌ main.py not found in current directory")
        return False
    
    try:
        # Start server with the fixed main.py
        print(f"🚀 Launching server: python -m uvicorn main:app --host 0.0.0.0 --port {port} --reload")
        
        # Use subprocess to start server in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app", 
            "--host", "0.0.0.0", 
            "--port", str(port),
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Wait a moment for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        # Check if server is running
        if check_server_health(port):
            print(f"✅ Server started successfully on port {port}")
            print(f"🌐 Dashboard available at: http://localhost:{port}")
            return True
        else:
            print("❌ Server failed to start properly")
            return False
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def check_server_health(port=8000):
    """Check if server is healthy"""
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server health check passed (attempt {attempt + 1})")
                return True
        except requests.exceptions.ConnectionError:
            if attempt < max_attempts - 1:
                print(f"⏳ Server not ready yet (attempt {attempt + 1}/{max_attempts})")
                time.sleep(1)
        except Exception as e:
            print(f"❓ Health check error: {e}")
    
    print(f"❌ Server did not respond after {max_attempts} attempts")
    return False

def test_dashboard(port=8000):
    """Test the dashboard functionality"""
    print(f"\n🧪 Testing Dashboard Functionality")
    print("=" * 50)
    
    try:
        # Test main dashboard
        response = requests.get(f"http://localhost:{port}", timeout=10)
        print(f"📊 Dashboard response: {response.status_code}")
        print(f"📄 Content length: {len(response.text)} characters")
        
        if response.status_code == 200:
            content = response.text
            
            # Check for key elements
            checks = [
                ("Trading Bot", "Dashboard title"),
                ("ml_status", "ML status variable"),
                ("Portfolio Performance", "Portfolio section"),
                ("ML Training", "ML training section"),
                ("chat", "Chat functionality")
            ]
            
            results = []
            for search_term, description in checks:
                if search_term.lower() in content.lower():
                    results.append(f"✅ {description} found")
                else:
                    results.append(f"❌ {description} missing")
            
            for result in results:
                print(f"   {result}")
            
            # Check if content is substantially longer (indicating full dashboard)
            if len(content) > 5000:
                print("✅ Dashboard appears to be fully loaded")
                return True
            else:
                print("⚠️ Dashboard content seems incomplete")
                return False
                
        else:
            print(f"❌ Dashboard returned error code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
        return False

def manual_instructions():
    """Provide manual instructions"""
    print(f"\n📖 Manual Instructions")
    print("=" * 50)
    print("If automatic fixes don't work, try these manual steps:")
    print()
    print("1. **Kill existing server manually:**")
    print("   • Press Ctrl+C in any terminal running the bot")
    print("   • Or use Task Manager to end python.exe processes")
    print()
    print("2. **Start server manually:**")
    print("   • cd to your bot directory")
    print("   • python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("3. **Try different port:**")
    print("   • python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload")
    print("   • Then visit http://localhost:8001")
    print()
    print("4. **Check the dashboard:**")
    print("   • Look for 'ML Training' section")
    print("   • Should show Lorentzian Classifier, Neural Network, etc.")

def main():
    """Main function"""
    print("🔧 Port Conflict Fix Script")
    print("=" * 50)
    
    port = 8000
    
    # Step 1: Find and kill existing processes
    print("Step 1: Stopping existing server...")
    if kill_processes_on_port(port):
        print("✅ Port cleared successfully")
    else:
        print("⚠️ Some processes might still be running")
    
    # Step 2: Verify port is free
    time.sleep(2)
    if not check_port_availability(port):
        print("❌ Port still in use. Trying alternative solutions...")
        port = 8001
        print(f"🔄 Switching to port {port}")
    
    # Step 3: Start the fixed server
    print(f"\nStep 2: Starting fixed server on port {port}...")
    
    # Don't start automatically, just give instructions
    print("📋 To start the server manually:")
    print(f"   python -m uvicorn main:app --host 0.0.0.0 --port {port} --reload")
    print()
    print("🌐 Then visit:")
    print(f"   http://localhost:{port}")
    print()
    print("📊 You should now see:")
    print("   ✅ Full dashboard with all sections")
    print("   ✅ ML Training section with model options")
    print("   ✅ No more 404 errors")
    
    # Provide manual instructions
    manual_instructions()

if __name__ == "__main__":
    main()