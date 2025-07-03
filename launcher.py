#!/usr/bin/env python3
"""
launcher.py - Enhanced Trading Bot Python Launcher
Simple Python script to launch different components
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        return False

def check_venv():
    """Check if we're in a virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        print("💡 Consider activating with: venv\\Scripts\\activate.bat")
        return False

def train_models():
    """Train ML models"""
    check_venv()
    return run_command("python enhanced_model_trainer.py --full-train --verbose", "Training models")

def start_bot():
    """Start the trading bot"""
    check_venv()
    return run_command("python main_trading_bot.py", "Starting trading bot")

def start_dashboard():
    """Start the web dashboard"""
    check_venv()
    return run_command("python web_dashboard.py --host localhost --port 8050", "Starting dashboard")

def run_backtest():
    """Run backtest"""
    check_venv()
    start_date = input("Enter start date (YYYY-MM-DD) or press Enter for 2024-01-01: ").strip()
    end_date = input("Enter end date (YYYY-MM-DD) or press Enter for 2024-12-31: ").strip()
    
    if not start_date:
        start_date = "2024-01-01"
    if not end_date:
        end_date = "2024-12-31"
    
    command = f"python backtesting_engine.py --start-date {start_date} --end-date {end_date}"
    return run_command(command, f"Running backtest from {start_date} to {end_date}")

def show_menu():
    """Show main menu"""
    print("\n" + "="*50)
    print("🤖 Enhanced Trading Bot Launcher")
    print("="*50)
    print("1. Train Models")
    print("2. Start Trading Bot")
    print("3. Start Web Dashboard")
    print("4. Run Backtest")
    print("5. Check System Status")
    print("0. Exit")
    print("="*50)

def check_system():
    """Check system status"""
    print("\n🔍 System Status Check:")
    print("-" * 30)
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check virtual environment
    check_venv()
    
    # Check if main files exist
    files_to_check = [
        "enhanced_model_trainer.py",
        "main_trading_bot.py", 
        "web_dashboard.py",
        ".env"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
    
    # Check directories
    dirs_to_check = ["models", "data", "logs", "config"]
    for dir in dirs_to_check:
        if os.path.exists(dir):
            print(f"✅ {dir}/ directory found")
        else:
            print(f"❌ {dir}/ directory missing")

def main():
    """Main launcher function"""
    while True:
        show_menu()
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == "1":
            train_models()
        elif choice == "2":
            start_bot()
        elif choice == "3":
            start_dashboard()
        elif choice == "4":
            run_backtest()
        elif choice == "5":
            check_system()
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()