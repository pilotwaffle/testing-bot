# File: start_bot_safe.py
# Location: E:\Trade Chat Bot\G Trading Bot\start_bot_safe.py
# Purpose: Safe startup wrapper with timeout protection and comprehensive monitoring
# Usage: python start_bot_safe.py

#!/usr/bin/env python3
"""
Safe Startup Wrapper for Crypto Trading Bot
Provides timeout protection, detailed monitoring, and error handling
"""

import sys
import time
import subprocess
import threading
import signal
import os
from pathlib import Path
from datetime import datetime

class SafeStartup:
    def __init__(self, timeout_seconds=90):
        self.timeout_seconds = timeout_seconds
        self.process = None
        self.startup_success = False
        self.output_lines = []
        
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("🔍 Checking Prerequisites...")
        
        required_files = [
            'main.py',
            'enhanced_trading_engine.py'
        ]
        
        recommended_files = [
            'config.json',
            'core/enhanced_ml_engine.py',
            'api/routers/chat_routes.py'
        ]
        
        missing_required = []
        missing_recommended = []
        
        # Check required files
        for file in required_files:
            if not Path(file).exists():
                missing_required.append(file)
                print(f"❌ Missing Required: {file}")
            else:
                print(f"✅ Found Required: {file}")
        
        # Check recommended files
        for file in recommended_files:
            if not Path(file).exists():
                missing_recommended.append(file)
                print(f"⚠️ Missing Recommended: {file}")
            else:
                print(f"✅ Found Recommended: {file}")
        
        if missing_required:
            print(f"\n❌ CRITICAL: Missing required files: {missing_required}")
            print("🔧 Run 'python fix_startup_issues.py' to create missing files")
            return False
        
        if missing_recommended:
            print(f"\n⚠️ Missing recommended files: {missing_recommended}")
            print("🔧 Consider running 'python fix_startup_issues.py' for optimal setup")
        
        print("✅ Prerequisites check passed")
        return True
    
    def check_python_environment(self):
        """Check Python environment and packages"""
        print("\n🐍 Checking Python Environment...")
        
        # Check Python version
        python_version = sys.version_info
        print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("⚠️ Python 3.8+ recommended for best compatibility")
        
        # Check critical packages
        critical_packages = ['fastapi', 'uvicorn', 'pandas', 'numpy']
        missing_packages = []
        
        for package in critical_packages:
            try:
                __import__(package)
                print(f"✅ Package available: {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ Package missing: {package}")
        
        if missing_packages:
            print(f"\n❌ Missing critical packages: {missing_packages}")
            print("🔧 Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ Python environment check passed")
        return True
    
    def start_with_monitoring(self):
        """Start bot with comprehensive monitoring"""
        print("🚀 Starting Crypto Trading Bot with Safe Monitoring")
        print("=" * 70)
        
        # Pre-flight checks
        if not self.check_prerequisites():
            return False
        
        if not self.check_python_environment():
            return False
        
        # Startup command
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        print(f"⏰ Timeout: {self.timeout_seconds} seconds")
        print(f"📍 Working Directory: {Path.cwd()}")
        print(f"🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)
        
        try:
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"📊 Process started with PID: {self.process.pid}")
            
            # Monitor with timeout
            return self._monitor_startup()
            
        except Exception as e:
            print(f"❌ Failed to start process: {e}")
            return False
    
    def _monitor_startup(self):
        """Monitor startup process with detailed feedback"""
        start_time = time.time()
        
        # Success indicators (bot started successfully)
        startup_success_indicators = [
            'uvicorn running on',
            'application startup complete',
            'started server process',
            'enhanced trading engine started',
            'dashboard available at',
            'application started successfully'
        ]
        
        # Progress indicators (things are working)
        progress_indicators = [
            'loading',
            'initializing',
            'starting',
            'connecting',
            'setting up',
            'configuring'
        ]
        
        # Error indicators (potential problems)
        error_indicators = [
            'error',
            'exception',
            'failed',
            'traceback',
            'timeout',
            'cannot import',
            'module not found',
            'connection refused'
        ]
        
        # Warning indicators (non-critical issues)
        warning_indicators = [
            'warning',
            'deprecated',
            'fallback',
            'unavailable'
        ]
        
        last_activity_time = start_time
        progress_count = 0
        error_count = 0
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            time_since_activity = current_time - last_activity_time
            
            # Check timeout
            if elapsed > self.timeout_seconds:
                print(f"\n⏰ STARTUP TIMEOUT after {elapsed:.1f}s")
                print("The bot is taking too long to start - this usually indicates:")
                print("   • Component initialization hanging")
                print("   • Network connectivity issues")  
                print("   • Database connection problems")
                print("   • Heavy ML model loading")
                self._terminate_process()
                self._show_diagnostic_suggestions()
                return False
            
            # Check for process death
            if self.process.poll() is not None:
                return_code = self.process.returncode
                print(f"\n❌ Process exited unexpectedly (exit code: {return_code})")
                if return_code != 0:
                    print("This indicates a startup error occurred.")
                self._show_recent_output()
                return False
            
            # Check for no activity (potential hang)
            if time_since_activity > 30:  # 30 seconds of no output
                print(f"\n⚠️ No output for {time_since_activity:.0f}s - process may be hanging...")
                print(f"   Elapsed time: {elapsed:.1f}s / {self.timeout_seconds}s")
            
            # Read output
            try:
                output = self.process.stdout.readline()
                if not output:
                    time.sleep(0.1)
                    continue
                    
                output = output.strip()
                if not output:
                    continue
                
                # Record output for diagnostics
                self.output_lines.append(f"[{elapsed:.1f}s] {output}")
                last_activity_time = current_time
                
                # Display output with color coding
                self._display_output_with_context(output, elapsed)
                
                # Check for startup success
                if any(indicator in output.lower() for indicator in startup_success_indicators):
                    self.startup_success = True
                    print(f"\n🎉 STARTUP SUCCESSFUL!")
                    print("=" * 50)
                    print(f"⏱️  Total startup time: {elapsed:.1f}s")
                    print(f"🌐 Dashboard: http://localhost:8000")
                    print(f"📚 API Documentation: http://localhost:8000/docs")
                    print(f"🔧 Admin Interface: http://localhost:8000/admin")
                    print("=" * 50)
                    print("🔄 Bot is running. Press Ctrl+C to stop.")
                    
                    # Keep running and handle shutdown gracefully
                    try:
                        self.process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 Received shutdown signal...")
                        self._terminate_process()
                        print("✅ Bot stopped cleanly")
                    
                    return True
                
                # Count progress indicators
                elif any(indicator in output.lower() for indicator in progress_indicators):
                    progress_count += 1
                
                # Count and highlight errors
                elif any(error in output.lower() for error in error_indicators):
                    if not any(warning in output.lower() for warning in warning_indicators):
                        error_count += 1
                        if error_count > 5:  # Too many errors
                            print(f"\n⚠️ Multiple errors detected ({error_count}). Startup may be failing.")
                
            except Exception as e:
                print(f"❌ Error reading process output: {e}")
                break
            
            time.sleep(0.1)
        
        return False
    
    def _display_output_with_context(self, output, elapsed):
        """Display output with helpful context and formatting"""
        timestamp = f"[{elapsed:6.1f}s]"
        
        # Color code different types of messages
        if any(word in output.lower() for word in ['error', 'exception', 'failed', 'traceback']):
            if 'warning' not in output.lower():
                print(f"{timestamp} ❌ {output}")
            else:
                print(f"{timestamp} ⚠️  {output}")
        elif any(word in output.lower() for word in ['success', 'complete', 'started', 'running']):
            print(f"{timestamp} ✅ {output}")
        elif any(word in output.lower() for word in ['loading', 'initializing', 'starting']):
            print(f"{timestamp} 🔄 {output}")
        elif any(word in output.lower() for word in ['warning', 'deprecated']):
            print(f"{timestamp} ⚠️  {output}")
        else:
            print(f"{timestamp} ℹ️  {output}")
    
    def _terminate_process(self):
        """Safely terminate the process"""
        if self.process:
            try:
                print("🛑 Terminating process...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                    print("✅ Process terminated cleanly")
                except subprocess.TimeoutExpired:
                    print("🔥 Force killing process...")
                    self.process.kill()
                    self.process.wait()
                    print("✅ Process killed")
            except Exception as e:
                print(f"❌ Error terminating process: {e}")
    
    def _show_recent_output(self):
        """Show recent output for debugging"""
        if self.output_lines:
            print("\n📄 Recent output (last 10 lines):")
            print("-" * 40)
            for line in self.output_lines[-10:]:
                print(line)
            print("-" * 40)
    
    def _show_diagnostic_suggestions(self):
        """Show diagnostic and troubleshooting suggestions"""
        print("\n🔧 TROUBLESHOOTING SUGGESTIONS:")
        print("=" * 50)
        print("1. 🔍 Run diagnostics: python startup_diagnostic.py")
        print("2. 🔧 Apply fixes: python fix_startup_issues.py")
        print("3. 📋 Check log files for detailed errors")
        print("4. ⏰ Try increasing timeout (current: {}s)".format(self.timeout_seconds))
        print("5. 🌐 Check network connectivity")
        print("6. 💾 Ensure database is accessible")
        print("7. 🧠 Disable ML features temporarily")
        
        if self.output_lines:
            print("\n📊 Startup Analysis:")
            total_lines = len(self.output_lines)
            error_lines = len([line for line in self.output_lines if 'error' in line.lower() and 'warning' not in line.lower()])
            warning_lines = len([line for line in self.output_lines if 'warning' in line.lower()])
            
            print(f"   • Total output lines: {total_lines}")
            print(f"   • Error messages: {error_lines}")
            print(f"   • Warning messages: {warning_lines}")
            
            if error_lines > 0:
                print("\n❌ Error messages found - check logs for details")
            if warning_lines > warning_lines:
                print("\n⚠️ Multiple warnings - may indicate configuration issues")

def main():
    """Main function"""
    print("🛡️ Safe Startup for Crypto Trading Bot")
    print("Provides timeout protection and detailed monitoring during startup\n")
    
    # Configuration
    print("⚙️ Startup Configuration:")
    try:
        timeout_input = input("Enter startup timeout in seconds (default 90): ") or "90"
        timeout = int(timeout_input)
        if timeout < 30:
            print("⚠️ Timeout too low, using minimum of 30 seconds")
            timeout = 30
        elif timeout > 300:
            print("⚠️ Timeout very high, using maximum of 300 seconds")
            timeout = 300
    except ValueError:
        timeout = 90
        print(f"❌ Invalid input, using default: {timeout}s")
    
    # Start bot
    print(f"\n🚀 Starting bot with {timeout}s timeout...\n")
    safe_startup = SafeStartup(timeout_seconds=timeout)
    success = safe_startup.start_with_monitoring()
    
    # Results
    if success:
        print("\n🎉 Bot startup completed successfully!")
        print("Your crypto trading bot is now running and accessible.")
    else:
        print("\n❌ Bot startup failed!")
        print("Check the troubleshooting suggestions above.")
        
        # Offer to run diagnostics
        run_diag = input("\nRun startup diagnostics? (y/n): ").lower()
        if run_diag == 'y':
            try:
                import subprocess
                subprocess.run([sys.executable, "startup_diagnostic.py"])
            except:
                print("❌ Could not run diagnostics automatically")
                print("🔧 Run manually: python startup_diagnostic.py")

if __name__ == "__main__":
    main()