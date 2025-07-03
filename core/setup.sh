#!/bin/bash
# setup.sh - Complete setup script for Enhanced Trading Bot

set -e  # Exit on any error

echo "🤖 Enhanced Trading Bot - Setup Script"
echo "======================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python 3.9+ is installed
check_python() {
    print_step "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.9"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_status "Python $PYTHON_VERSION found ✓"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_status "Python $PYTHON_VERSION found ✓"
            PYTHON_CMD="python"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_step "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
            $PYTHON_CMD -m venv venv
            print_status "Virtual environment recreated ✓"
        else
            print_status "Using existing virtual environment ✓"
        fi
    else
        $PYTHON_CMD -m venv venv
        print_status "Virtual environment created ✓"
    fi
}

# Activate virtual environment
activate_venv() {
    print_step "Activating virtual environment..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_status "Virtual environment activated ✓"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        print_status "Virtual environment activated ✓"
    else
        print_error "Could not find virtual environment activation script"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_step "Installing dependencies..."
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f "requirements.txt" ]; then
        print_status "Creating requirements.txt..."
        cat > requirements.txt << 'EOF'
# Core ML and Data Science
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
joblib>=1.1.0

# Trading and Financial Data
ccxt>=3.0.0
ccxt-pro>=3.0.0

# Web Framework and Visualization
flask>=2.0.0
plotly>=5.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Database and Async
aiohttp>=3.8.0
asyncio

# Utilities
python-dateutil>=2.8.0
pytz>=2021.3
requests>=2.25.0
pathlib2>=2.3.0

# System Monitoring
psutil>=5.8.0

# Optional: For advanced features
ta>=0.7.0

# Development dependencies (optional)
pytest>=6.0.0
black>=21.0.0
flake8>=3.9.0
EOF
    fi
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    pip install -r requirements.txt
    
    print_status "Dependencies installed ✓"
}

# Create directory structure
create_directories() {
    print_step "Creating directory structure..."
    
    mkdir -p config
    mkdir -p data/cache
    mkdir -p logs
    mkdir -p models
    mkdir -p backtest_results
    mkdir -p templates
    mkdir -p tests
    mkdir -p scripts
    mkdir -p strategies
    
    print_status "Directory structure created ✓"
}

# Create environment file
create_env_file() {
    print_step "Setting up environment file..."
    
    if [ ! -f ".env" ]; then
        # Check if user has an existing .env they want to migrate
        if [ -f ".env.old" ] || [ -f "tradesv3.sqlite" ]; then
            print_warning "Existing trading bot configuration detected."
            echo "Would you like to migrate from your existing configuration? (y/N): "
            read -r migrate_response
            
            if [[ $migrate_response =~ ^[Yy]$ ]]; then
                print_status "Running migration helper..."
                if command -v python3 &> /dev/null; then
                    python3 migration_helper.py migrate --source .env.old --target .env 2>/dev/null || true
                fi
            fi
        fi
        
        # If still no .env file, create from template
        if [ ! -f ".env" ]; then
            if [ -f ".env.template" ]; then
                cp .env.template .env
                print_status "Created .env from template ✓"
                print_warning "IMPORTANT: Please edit .env file with your API credentials"
            else
                print_warning ".env.template not found, creating comprehensive .env file"
                cat > .env << 'EOF'
################################################################################
# IMPORTANT: DO NOT COMMIT THIS FILE TO PUBLIC VERSION CONTROL               #
# This file contains sensitive API keys and personal credentials.             #
################################################################################

# =============================================================================
# DATABASE CONNECTION
# =============================================================================
DATABASE_URL=sqlite:///data/enhanced_trading_bot.db

# =============================================================================
# APP METADATA  
# =============================================================================
APP_NAME="Enhanced Trading Bot"
APP_USER_ID=admin
APP_PASSWORD=admin123

# =============================================================================
# EXCHANGE API CREDENTIALS
# =============================================================================

# Kraken Exchange API Credentials (PRIMARY)
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_SECRET=your_kraken_secret_here
KRAKEN_SANDBOX=true

# Alpaca API Credentials (for stock/crypto trading)
APCA_API_KEY_ID=your_alpaca_key_id_here
APCA_API_SECRET_KEY=your_alpaca_secret_here
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets
ALPACA_STREAM_DATA_URL=wss://stream.data.alpaca.markets/v1beta3/crypto/us

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================
LIVE_TRADING_ENABLED=false
DEFAULT_EXCHANGE=kraken
TRADING_SYMBOLS=BTC/USD,ETH/USD,ADA/USD
DEFAULT_TRAINING_SYMBOLS=BTC/USD,ETH/USD,ADA/USD
TRADING_TIMEFRAMES=1h,4h,1d
INITIAL_CAPITAL=10000
CCXT_AVAILABLE=true

# Sync and Update Intervals
ALPACA_SYNC_INTERVAL_MINUTES=5
BROADCAST_INTERVAL_SECONDS=15
ERROR_RETRY_INTERVAL_SECONDS=30

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
MAX_RISK_PER_TRADE=0.02
MAX_PORTFOLIO_RISK=0.10
EMERGENCY_STOP_LOSS=0.15
TRADE_NOTIFICATION_MIN_VALUE=10.0

# =============================================================================
# AI AND EXTERNAL API KEYS
# =============================================================================
GOOGLE_AI_API_KEY=your_gemini_api_key_here
GOOGLE_AI_ENABLED=true
GOOGLE_AI_MODEL=gemini-pro
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key_here

# Enhanced Chat Settings
CHAT_MEMORY_SIZE=25
CHAT_VOICE_ENABLED=true
CHAT_PROACTIVE_INSIGHTS=true

# =============================================================================
# NOTIFICATIONS CONFIGURATION
# =============================================================================

# Email Notifications
EMAIL_ENABLED=false
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_gmail_app_password_here
RECIPIENT_EMAIL=alerts@yourdomain.com

# Slack Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
SLACK_CHANNEL=#trading-bot
SLACK_USERNAME=TradingBot

# Discord Notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK

# SMS Notifications (Twilio)
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_PHONE_NUMBER=your_twilio_phone_number_here
RECIPIENT_PHONE_NUMBER=your_recipient_phone_number_here

# =============================================================================
# NOTIFICATION BEHAVIOR SETTINGS
# =============================================================================
NOTIFY_TRADES=true
NOTIFY_SYSTEM_EVENTS=true
NOTIFY_ERRORS=true
NOTIFY_PERFORMANCE=true
NOTIFY_STRATEGY_CHANGES=true

# Notification Timing and Filtering
PERFORMANCE_NOTIFICATION_INTERVAL=3600
ERROR_NOTIFICATION_COOLDOWN=300
MIN_NOTIFICATION_PRIORITY=MEDIUM
NOTIFICATION_HISTORY_MAX_LENGTH=100

# =============================================================================
# SECURITY
# =============================================================================
SECRET_KEY=change_this_secret_key_for_production_use
JWT_SECRET=change_this_jwt_secret_for_dashboard_auth

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================
LOG_LEVEL=INFO
DEBUG=false
DEVELOPMENT_MODE=true
FLASK_DEBUG=true

# Directory Paths
DEFAULT_MODEL_SAVE_PATH=models/
STRATEGIES_DIR=strategies

# =============================================================================
# MIGRATION NOTES
# =============================================================================
# This .env file was created by the Enhanced Trading Bot setup script.
# Please update the following before running:
# 1. Add your Kraken/Alpaca API credentials
# 2. Configure Google AI API key for chat features
# 3. Set up notification services (Slack, Discord, Email, SMS)
# 4. Change default passwords and secrets for production
# 5. Review risk management settings
# 6. Update CoinMarketCap API key if using external market data
EOF
                print_status "Created comprehensive .env file ✓"
            fi
        fi
    else
        print_warning ".env file already exists"
        
        # Offer to backup and update existing .env
        echo "Would you like to backup your existing .env and update it? (y/N): "
        read -r backup_response
        
        if [[ $backup_response =~ ^[Yy]$ ]]; then
            cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
            print_status "Backed up existing .env file ✓"
            
            # Run migration to update format
            if command -v python3 &> /dev/null; then
                python3 migration_helper.py migrate --source .env --target .env.updated 2>/dev/null || true
                if [ -f ".env.updated" ]; then
                    mv .env.updated .env
                    print_status "Updated .env file format ✓"
                fi
            fi
        fi
    fi
    
    # Validate .env file
    if [ -f ".env" ]; then
        print_status "Validating .env configuration..."
        
        # Check for placeholder values
        if grep -q "your_kraken_api_key_here" .env; then
            print_warning "⚠️  Kraken API key needs to be configured in .env"
        fi
        
        if grep -q "your_alpaca_key_id_here" .env; then
            print_warning "⚠️  Alpaca API credentials need to be configured in .env"
        fi
        
        if grep -q "your_gemini_api_key_here" .env; then
            print_warning "⚠️  Google AI API key needs to be configured for chat features"
        fi
        
        if grep -q "admin123" .env; then
            print_warning "⚠️  Default dashboard password detected - change for production"
        fi
        
        if grep -q "change_this" .env; then
            print_warning "⚠️  Default secret keys detected - change for production"
        fi
    fi
}

# Create configuration files
create_config_files() {
    print_step "Creating configuration files..."
    
    # Bot configuration
    if [ ! -f "config/bot_config.json" ]; then
        cat > config/bot_config.json << 'EOF'
{
  "exchange": {
    "name": "kraken",
    "api_key": "YOUR_API_KEY_HERE",
    "secret": "YOUR_SECRET_HERE",
    "sandbox": true,
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
    "live_trading_enabled": false,
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
    "email_enabled": false,
    "webhook_enabled": false,
    "log_all_signals": true
  },
  "data_collection": {
    "historical_data_points": 2000,
    "real_time_updates": true,
    "cache_validity_hours": 6
  }
}
EOF
        print_status "Created config/bot_config.json"
    fi
    
    # Strategy configuration
    if [ ! -f "config/strategy_config.json" ]; then
        cat > config/strategy_config.json << 'EOF'
{
  "risk_management": {
    "max_risk_per_trade": 0.02,
    "max_portfolio_risk": 0.10,
    "stop_loss_multiplier": 2.0,
    "take_profit_multiplier": 3.0,
    "trailing_stop_enabled": true
  },
  "signal_generation": {
    "min_confidence_threshold": 0.6,
    "ensemble_weight_ml": 0.6,
    "ensemble_weight_technical": 0.3,
    "ensemble_weight_sentiment": 0.1,
    "confirmation_required": true,
    "multi_timeframe_analysis": true
  },
  "position_sizing": {
    "base_position_size": 0.1,
    "kelly_criterion_enabled": true,
    "volatility_adjustment": true,
    "confidence_scaling": true,
    "max_position_size": 0.25,
    "min_position_size": 0.01
  },
  "timeframe_weights": {
    "1h": 0.2,
    "4h": 0.3,
    "1d": 0.5
  }
}
EOF
        print_status "Created config/strategy_config.json"
    fi

    # Monitor configuration
    if [ ! -f "config/monitor_config.json" ]; then
        cat > config/monitor_config.json << 'EOF'
{
  "performance_metrics": {
    "tracking_enabled": true,
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
    "daily_summary_enabled": true,
    "weekly_report_enabled": true,
    "performance_charts_enabled": true,
    "trade_analysis_enabled": true
  },
  "system_health": {
    "memory_usage_threshold": 0.85,
    "cpu_usage_threshold": 0.80,
    "disk_space_threshold": 0.90,
    "api_latency_threshold_ms": 5000
  }
}
EOF
        print_status "Created config/monitor_config.json"
    fi
    
    print_status "Configuration files created ✓"
}

# Create startup scripts
create_startup_scripts() {
    print_step "Creating startup scripts..."
    
    # Main startup script
    cat > start_bot.sh << 'EOF'
#!/bin/bash
# Start Enhanced Trading Bot

echo "🤖 Starting Enhanced Trading Bot..."

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
    echo "⚠️  No trained models found. Training models first..."
    python enhanced_model_trainer.py --full-train
fi

# Start the bot
echo "🚀 Starting trading bot..."
python main_trading_bot.py
EOF

    # Dashboard startup script
    cat > start_dashboard.sh << 'EOF'
#!/bin/bash
# Start Web Dashboard

echo "📊 Starting Web Dashboard..."

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
echo "🌐 Dashboard will be available at http://localhost:8050"
python web_dashboard.py --host localhost --port 8050
EOF

    # Training script
    cat > train_models.sh << 'EOF'
#!/bin/bash
# Train ML Models

echo "🧠 Training ML Models..."

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
EOF

    # Backtest script
    cat > run_backtest.sh << 'EOF'
#!/bin/bash
# Run Backtest

echo "🧪 Running Backtest..."

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

echo "📅 Backtesting period: $START_DATE to $END_DATE"
python backtesting_engine.py --start-date "$START_DATE" --end-date "$END_DATE"
EOF

    # Make scripts executable
    chmod +x start_bot.sh start_dashboard.sh train_models.sh run_backtest.sh
    
    print_status "Startup scripts created ✓"
}

# Create Windows batch files
create_windows_scripts() {
    print_step "Creating Windows batch files..."
    
    # Windows startup script
    cat > start_bot.bat << 'EOF'
@echo off
echo 🤖 Starting Enhanced Trading Bot...

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models" mkdir models
dir /b models | findstr /r ".*" >nul
if errorlevel 1 (
    echo ⚠️ No trained models found. Training models first...
    python enhanced_model_trainer.py --full-train
)

REM Start the bot
echo 🚀 Starting trading bot...
python main_trading_bot.py
pause
EOF

    # Windows dashboard script
    cat > start_dashboard.bat << 'EOF'
@echo off
echo 📊 Starting Web Dashboard...

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found
    pause
    exit /b 1
)

echo 🌐 Dashboard will be available at http://localhost:8050
python web_dashboard.py --host localhost --port 8050
pause
EOF

    # Windows training script
    cat > train_models.bat << 'EOF'
@echo off
echo 🧠 Training ML Models...

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found
    pause
    exit /b 1
)

python enhanced_model_trainer.py --full-train --verbose
pause
EOF

    print_status "Windows batch files created ✓"
}

# Validate installation
validate_installation() {
    print_step "Validating installation..."
    
    # Check if main files exist
    local files=(
        "enhanced_ml_engine.py"
        "enhanced_data_fetcher.py" 
        "enhanced_trading_strategy.py"
        "enhanced_model_trainer.py"
        "performance_monitor.py"
        "main_trading_bot.py"
        "backtesting_engine.py"
        "web_dashboard.py"
    )
    
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            print_warning "Missing file: $file"
        fi
    done
    
    # Test imports
    if python -c "import numpy, pandas, sklearn, tensorflow" 2>/dev/null; then
        print_status "Core dependencies can be imported ✓"
    else
        print_error "Some core dependencies cannot be imported"
    fi
    
    if python -c "import ccxt" 2>/dev/null; then
        print_status "CCXT library available ✓"
    else
        print_warning "CCXT library not available - exchange connectivity may not work"
    fi
    
    print_status "Installation validation completed ✓"
}

# Display next steps
show_next_steps() {
    print_step "Installation Complete!"
    echo
    echo "🎉 Enhanced Trading Bot has been set up successfully!"
    echo
    echo "📝 Next Steps:"
    echo "1. 🔑 Configure your API credentials in the .env file:"
    echo "   - Edit .env with your Kraken API key and secret"
    echo "   - Add your Alpaca API credentials for stock/crypto trading"
    echo "   - Configure Google AI API key for enhanced chat features"
    echo "   - Set up notification services (Slack, Discord, Email)"
    echo "2. 🧠 Run initial model training:"
    echo "   ./train_models.sh (Linux/Mac) or train_models.bat (Windows)"
    echo "3. 🧪 Run a backtest to validate the strategy:"
    echo "   ./run_backtest.sh (Linux/Mac) or run_backtest.bat (Windows)"
    echo "4. 📊 Start paper trading:"
    echo "   ./start_bot.sh (Linux/Mac) or start_bot.bat (Windows)"
    echo "5. 🌐 Launch the web dashboard (in another terminal):"
    echo "   ./start_dashboard.sh (Linux/Mac) or start_dashboard.bat (Windows)"
    echo
    echo "📚 Documentation:"
    echo "- README.md - Complete system overview"
    echo "- Setup Guide - Detailed installation instructions"
    echo "- config/ - All configuration files"
    echo
    echo "⚠️  Important Security Notes:"
    echo "- Start with paper trading (LIVE_TRADING_ENABLED=false)"
    echo "- Change default passwords (APP_PASSWORD, SECRET_KEY, JWT_SECRET)"
    echo "- Never commit .env file to version control"
    echo "- Always validate your strategy with backtesting first"
    echo "- Never risk more than you can afford to lose"
    echo
    echo "🔧 API Keys to Configure:"
    echo "- Kraken API: https://www.kraken.com/u/security/api"
    echo "- Alpaca API: https://alpaca.markets/"
    echo "- Google AI: https://makersuite.google.com/"
    echo "- CoinMarketCap: https://coinmarketcap.com/api/"
    echo
    echo "🆘 Need help? Check the logs/ directory for detailed information"
    echo
}

# Main execution
main() {
    echo "Starting setup process..."
    echo
    
    check_python
    create_venv
    activate_venv
    install_dependencies
    create_directories
    create_env_file
    create_config_files
    create_startup_scripts
    create_windows_scripts
    validate_installation
    show_next_steps
    
    echo
    echo "✅ Setup completed successfully!"
}

# Check if we're being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi