# File: launch_bot.py
# Location: E:\Trade Chat Bot\G Trading Bot\launch_bot.py
# Purpose: Complete launcher and diagnostic tool for crypto trading bot
# Usage: python launch_bot.py

#!/usr/bin/env python3
"""
Complete Trading Bot Launcher
Handles diagnostics, fixes, and safe startup in one tool
"""

import sys
import os
import subprocess
import time
from pathlib import Path

class TradingBotLauncher:
    def __init__(self):
        self.scripts_created = []
        
    def check_environment(self):
        """Check if environment is ready"""
        print("🔍 Checking Environment...")
        
        required_files = ['main.py']
        missing_files = []
        
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
                print(f"❌ Missing: {file}")
            else:
                print(f"✅ Found: {file}")
        
        if missing_files:
            print(f"\n⚠️ Missing required files: {missing_files}")
            return False
        
        print("✅ Environment check passed")
        return True
    
    def run_diagnostics(self):
        """Run startup diagnostics"""
        print("\n🔍 Running Startup Diagnostics...")
        
        # Check for diagnostic scripts
        diag_scripts = [
            'startup_diagnostic.py',
            'quick_diagnostic.py'
        ]
        
        for script in diag_scripts:
            if Path(script).exists():
                print(f"📊 Running: {script}")
                try:
                    result = subprocess.run([sys.executable, script], 
                                          input="10\n", text=True,
                                          capture_output=True, timeout=60)
                    
                    if result.returncode == 0:
                        print("✅ Diagnostics completed successfully")
                        
                        # Check for hanging indicators
                        if "TIMEOUT" in result.stdout or "HANGING" in result.stdout:
                            print("⚠️ Hanging issues detected!")
                            print("Run fixes before starting bot")
                            return False
                        else:
                            print("✅ No hanging issues detected")
                            return True
                    else:
                        print(f"❌ Diagnostic failed: {result.stderr}")
                        return False
                        
                except subprocess.TimeoutExpired:
                    print("⏰ Diagnostic timed out - this indicates hanging issues")
                    return False
                except Exception as e:
                    print(f"❌ Diagnostic error: {e}")
                    return False
        
        print("⚠️ No diagnostic scripts found - creating them...")
        return None  # Will trigger creation
    
    def apply_fixes(self):
        """Apply startup fixes"""
        print("\n🔧 Applying Startup Fixes...")
        
        if Path('fix_startup_issues.py').exists():
            print("📋 Running fix_startup_issues.py...")
            try:
                result = subprocess.run([sys.executable, 'fix_startup_issues.py'],
                                      input="1\n", text=True,
                                      capture_output=True, timeout=120)
                
                if result.returncode == 0:
                    print("✅ Fixes applied successfully")
                    return True
                else:
                    print(f"❌ Fixes failed: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Fix error: {e}")
                return False
        else:
            print("⚠️ fix_startup_issues.py not found")
            return False
    
    def start_bot_safe(self):
        """Start bot with safe startup wrapper"""
        print("\n🚀 Starting Trading Bot (Safe Mode)...")
        
        if Path('start_bot_safe.py').exists():
            print("🛡️ Using safe startup wrapper...")
            try:
                # Run in interactive mode
                subprocess.run([sys.executable, 'start_bot_safe.py'])
                return True
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped by user")
                return True
            except Exception as e:
                print(f"❌ Safe startup failed: {e}")
                return False
        else:
            print("⚠️ Safe startup wrapper not found, using direct startup...")
            return self.start_bot_direct()
    
    def start_bot_direct(self):
        """Start bot directly with uvicorn"""
        print("\n🚀 Starting Trading Bot (Direct Mode)...")
        
        try:
            cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
            print(f"🔧 Command: {' '.join(cmd)}")
            
            process = subprocess.Popen(cmd)
            
            print("✅ Bot started!")
            print("🌐 Dashboard: http://localhost:8000")
            print("Press Ctrl+C to stop...")
            
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                process.terminate()
                process.wait()
            
            return True
            
        except Exception as e:
            print(f"❌ Direct startup failed: {e}")
            return False
    
    def show_menu(self):
        """Show main menu"""
        print("\n" + "=" * 60)
        print("🤖 CRYPTO TRADING BOT LAUNCHER")
        print("=" * 60)
        print("1. 🔍 Run Diagnostics Only")
        print("2. 🔧 Apply Fixes Only") 
        print("3. 🚀 Start Bot (Safe Mode)")
        print("4. ⚡ Start Bot (Direct Mode)")
        print("5. 🛠️ Full Setup (Diagnostics + Fixes + Start)")
        print("6. 📊 Create Missing Scripts")
        print("7. ❌ Exit")
        print("=" * 60)
        
        choice = input("Enter choice (1-7, default 5): ").strip() or "5"
        return choice
    
    def create_missing_scripts(self):
        """Create any missing diagnostic/fix scripts"""
        print("\n📋 Creating Missing Scripts...")
        
        # This would typically copy the scripts from the artifacts
        # For now, just inform the user
        scripts_to_create = [
            'startup_diagnostic.py',
            'fix_startup_issues.py', 
            'start_bot_safe.py'
        ]
        
        missing_scripts = []
        for script in scripts_to_create:
            if not Path(script).exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            print("⚠️ Missing scripts detected:")
            for script in missing_scripts:
                print(f"   - {script}")
            
            print("\n💡 Please save the provided scripts from the chat to your directory:")
            print("   1. Copy 'Complete Startup Diagnostic' as 'startup_diagnostic.py'")
            print("   2. Copy 'Fix Startup Issues' as 'fix_startup_issues.py'")
            print("   3. These scripts are provided in the previous chat messages")
            
            return False
        else:
            print("✅ All required scripts are present")
            return True
    
    def run_full_setup(self):
        """Run complete setup process"""
        print("\n🛠️ RUNNING FULL SETUP")
        print("=" * 50)
        
        # Step 1: Check environment
        if not self.check_environment():
            print("❌ Environment check failed")
            return False
        
        # Step 2: Create missing scripts
        if not self.create_missing_scripts():
            print("❌ Missing required scripts")
            return False
        
        # Step 3: Run diagnostics
        diag_result = self.run_diagnostics()
        if diag_result is False:
            print("🔧 Diagnostics detected issues - applying fixes...")
            
            # Step 4: Apply fixes
            if not self.apply_fixes():
                print("❌ Fixes failed")
                return False
            
            print("✅ Fixes applied - re-running diagnostics...")
            diag_result = self.run_diagnostics()
        
        if diag_result is None:
            print("⚠️ Could not run diagnostics")
        
        # Step 5: Start bot
        print("\n🚀 Starting bot...")
        return self.start_bot_safe()
    
    def run(self):
        """Main launcher loop"""
        print("🤖 Welcome to Crypto Trading Bot Launcher!")
        
        while True:
            choice = self.show_menu()
            
            if choice == "1":
                self.run_diagnostics()
            elif choice == "2":
                self.apply_fixes()
            elif choice == "3":
                self.start_bot_safe()
                break
            elif choice == "4":
                self.start_bot_direct()
                break
            elif choice == "5":
                self.run_full_setup()
                break
            elif choice == "6":
                self.create_missing_scripts()
            elif choice == "7":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice, please try again")
            
            input("\nPress Enter to continue...")

def main():
    """Main function"""
    launcher = TradingBotLauncher()
    launcher.run()

if __name__ == "__main__":
    main()