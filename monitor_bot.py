#!/usr/bin/env python3
"""
File: monitor_bot.py
Location: E:\Trade Chat Bot\G Trading Bot\monitor_bot.py
Description: Elite Trading Bot V3.0 - Health Monitor
Purpose: Monitor bot health and restart if necessary
"""

import requests
import time
import subprocess
import sys
from pathlib import Path

def check_bot_health():
    """Check if bot is responding"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def check_bot_running():
    """Check if bot process is running"""
    try:
        response = requests.get("http://localhost:8000", timeout=5)
        return True
    except:
        return False

def restart_bot():
    """Restart the bot"""
    print("🔄 Restarting bot...")
    bot_path = Path(__file__).parent
    startup_script = bot_path / "start_safe.bat" if sys.platform == "win32" else bot_path / "start_safe.sh"
    
    if startup_script.exists():
        subprocess.Popen([str(startup_script)], shell=True)
    else:
        print("❌ Startup script not found")

def main():
    """Main monitoring loop"""
    print("🔍 Elite Trading Bot V3.0 - Health Monitor Starting...")
    
    consecutive_failures = 0
    max_failures = 3
    
    while True:
        try:
            if check_bot_running():
                if check_bot_health():
                    print(f"✅ Bot healthy at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    print(f"⚠️ Bot health check failed ({consecutive_failures}/{max_failures})")
            else:
                consecutive_failures += 1
                print(f"❌ Bot not responding ({consecutive_failures}/{max_failures})")
            
            if consecutive_failures >= max_failures:
                print("🚨 Bot appears to be stuck - restart required")
                # Could add restart logic here if desired
                consecutive_failures = 0
            
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
