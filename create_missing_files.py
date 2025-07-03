#!/usr/bin/env python3
"""
create_missing_files.py - Create missing Enhanced Trading Bot files
Run this script to create all the missing startup scripts and config files
Windows-compatible version without emoji characters
"""

import os
import json

def create_startup_scripts():
    """Create Linux/Mac startup scripts"""
    
    # start_bot.sh
    start_bot_content = '''#!/bin/bash
echo "Starting Enhanced Trading Bot..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "WARNING: No trained models found. Training models first..."
    python enhanced_model_trainer.py --full-train
fi

# Start the bot
echo "Starting trading bot..."
python main_trading_bot.py
'''

    # start_dashboard.sh
    start_dashboard_content = '''#!/bin/bash
echo "Starting Web Dashboard..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Start dashboard
echo "Dashboard will be available at http://localhost:8050"
python web_dashboard.py --host localhost --port 8050
'''

    # train_models.sh
    train_models_content = '''#!/bin/bash
echo "Training ML Models..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Run training
python enhanced_model_trainer.py --full-train --verbose
'''

    # run_backtest.sh
    run_backtest_content = '''#!/bin/bash
echo "Running Backtest..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Default parameters
START_DATE=${1:-"2024-01-01"}
END_DATE=${2:-"2024-12-31"}

echo "Backtesting period: $START_DATE to $END_DATE"
python backtesting_engine.py --start-date "$START_DATE" --end-date "$END_DATE"
'''

    scripts = {
        'start_bot.sh': start_bot_content,
        'start_dashboard.sh': start_dashboard_content,
        'train_models.sh': train_models_content,
        'run_backtest.sh': run_backtest_content
    }
    
    for filename, content in scripts.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        try:
            os.chmod(filename, 0o755)  # Make executable
        except:
            pass  # Skip if chmod not available on Windows
        print(f"Created {filename}")

def create_windows_scripts():
    """Create Windows batch files"""
    
    # start_bot.bat
    start_bot_bat = '''@echo off
echo Starting Enhanced Trading Bot...

REM Activate virtual environment
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
) else (
    echo Error: Virtual environment not found
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models" mkdir models
dir /b models | findstr /r ".*" >nul
if errorlevel 1 (
    echo WARNING: No trained models found. Training models first...
    python enhanced_model_trainer.py --full-train
)

REM Start the bot
echo Starting trading bot...
python main_trading_bot.py
pause
'''

    # start_dashboard.bat
    start_dashboard_bat = '''@echo off
echo Starting Web Dashboard...

REM Activate virtual environment
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
) else (
    echo Error: Virtual environment not found
    pause
    exit /b 1
)

echo Dashboard will be available at http://localhost:8050
python web_dashboard.py --host localhost --port 8050
pause
'''

    # train_models.bat
    train_models_bat = '''@echo off
echo Training ML Models...

REM Activate virtual environment
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
) else (
    echo Error: Virtual environment not found
    pause
    exit /b 1
)

python enhanced_model_trainer.py --full-train --verbose
pause
'''

    bat_files = {
        'start_bot.bat': start_bot_bat,
        'start_dashboard.bat': start_dashboard_bat,
        'train_models.bat': train_models_bat
    }
    
    for filename, content in bat_files.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created {filename}")

def create_config_files():
    """Create JSON configuration files"""
    
    # Ensure config directory exists
    os.makedirs('config', exist_ok=True)
    
    # bot_config.json
    bot_config = {
        "exchange": {
            "name": "kraken",
            "api_key": "YOUR_API_KEY_HERE",
            "secret": "YOUR_SECRET_HERE",
            "sandbox": True,
            "rate_limit": 1200,
            "timeout": 30000
        },
        "paths": {
            "data_cache_dir": "data/cache/",
            "model_save_path": "models/",
            "performance_log_path": "logs/",
            "strategy_config_path": "config/strategy_config.json",
            "monitor_config_path": "config/monitor_config.json"
        },
        "trading": {
            "symbols": ["BTC/USD", "ETH/USD", "ADA/USD"],
            "timeframes": ["1h", "4h", "1d"],
            "live_trading_enabled": False,
            "signal_generation_interval_minutes": 15,
            "max_concurrent_positions": 3,
            "portfolio_allocation": {
                "BTC/USD": 0.4,
                "ETH/USD": 0.35,
                "ADA/USD": 0.25
            }
        },
        "risk_management": {
            "max_portfolio_risk": 0.10,
            "max_daily_loss": 0.05,
            "emergency_stop_loss": 0.15,
            "position_timeout_hours": 72
        },
        "notifications": {
            "email_enabled": False,
            "webhook_enabled": False,
            "log_all_signals": True
        },
        "data_collection": {
            "historical_data_points": 2000,
            "real_time_updates": True,
            "cache_validity_hours": 6
        }
    }
    
    # strategy_config.json
    strategy_config = {
        "risk_management": {
            "max_risk_per_trade": 0.02,
            "max_portfolio_risk": 0.10,
            "stop_loss_multiplier": 2.0,
            "take_profit_multiplier": 3.0,
            "trailing_stop_enabled": True
        },
        "signal_generation": {
            "min_confidence_threshold": 0.6,
            "ensemble_weight_ml": 0.6,
            "ensemble_weight_technical": 0.3,
            "ensemble_weight_sentiment": 0.1,
            "confirmation_required": True,
            "multi_timeframe_analysis": True
        },
        "position_sizing": {
            "base_position_size": 0.1,
            "kelly_criterion_enabled": True,
            "volatility_adjustment": True,
            "confidence_scaling": True,
            "max_position_size": 0.25,
            "min_position_size": 0.01
        },
        "timeframe_weights": {
            "1h": 0.2,
            "4h": 0.3,
            "1d": 0.5
        }
    }
    
    # monitor_config.json
    monitor_config = {
        "performance_metrics": {
            "tracking_enabled": True,
            "calculation_frequency_minutes": 5,
            "history_retention_days": 365,
            "benchmark_symbol": "BTC/USD"
        },
        "alerts": {
            "portfolio_loss_threshold": 0.05,
            "daily_loss_threshold": 0.03,
            "consecutive_losses_threshold": 5,
            "low_balance_threshold": 1000,
            "high_volatility_threshold": 0.10
        },
        "reporting": {
            "daily_summary_enabled": True,
            "weekly_report_enabled": True,
            "performance_charts_enabled": True,
            "trade_analysis_enabled": True
        },
        "system_health": {
            "memory_usage_threshold": 0.85,
            "cpu_usage_threshold": 0.80,
            "disk_space_threshold": 0.90,
            "api_latency_threshold_ms": 5000
        }
    }
    
    configs = {
        'config/bot_config.json': bot_config,
        'config/strategy_config.json': strategy_config,
        'config/monitor_config.json': monitor_config
    }
    
    for filename, config_data in configs.items():
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        print(f"Created {filename}")

def main():
    print("Creating missing Enhanced Trading Bot files...")
    print()
    
    try:
        create_startup_scripts()
        print("Linux/Mac startup scripts created successfully!")
    except Exception as e:
        print(f"Error creating startup scripts: {e}")
    
    try:
        create_windows_scripts()
        print("Windows batch files created successfully!")
    except Exception as e:
        print(f"Error creating Windows scripts: {e}")
    
    try:
        create_config_files()
        print("Configuration files created successfully!")
    except Exception as e:
        print(f"Error creating config files: {e}")
    
    print()
    print("All missing files created successfully!")
    print()
    print("Next steps:")
    print("1. Configure your .env file with API credentials")
    print("2. Run: train_models.bat to train models")
    print("3. Run: start_bot.bat to start the bot")
    print("4. Run: start_dashboard.bat to start the dashboard")
    print()
    print("NOTE: On Windows, use the .bat files")

if __name__ == "__main__":
    main()