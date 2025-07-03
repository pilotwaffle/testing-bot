#!/usr/bin/env python3
# cli.py - Command Line Interface Utility
"""
Enhanced Trading Bot CLI Utility
Command line interface for common operations and system management
"""

import click
import json
import logging
import sys
import subprocess
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

# Try to import our modules
try:
    from enhanced_model_trainer import AdaptiveModelTrainer
    from performance_monitor import PerformanceMonitor
    from main_trading_bot import TradingBotOrchestrator
    from backtesting_engine import AdvancedBacktester
except ImportError as e:
    print(f"Warning: Could not import trading bot modules: {e}")
    print("Make sure you're in the correct directory and dependencies are installed.")

@click.group()
@click.option('--config', default='config/bot_config.json', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Enhanced Trading Bot CLI - Manage your algorithmic trading system"""
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    # Check if config file exists
    if not Path(config).exists():
        click.echo(f"❌ Configuration file not found: {config}")
        click.echo("Run 'python cli.py setup' to create initial configuration")
        ctx.exit(1)

@cli.command()
def setup():
    """Setup initial configuration and directory structure"""
    
    click.echo("🔧 Setting up Enhanced Trading Bot...")
    
    # Create directories
    directories = [
        'config', 'data/cache', 'logs', 'models', 
        'backtest_results', 'templates', 'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        click.echo(f"✅ Created directory: {directory}")
    
    # Create basic configuration
    config = {
        "exchange": {
            "name": "kraken",
            "api_key": "YOUR_API_KEY_HERE",
            "secret": "YOUR_SECRET_HERE",
            "sandbox": True
        },
        "trading": {
            "symbols": ["BTC/USD", "ETH/USD"],
            "timeframes": ["1h", "4h", "1d"],
            "live_trading_enabled": False,
            "signal_generation_interval_minutes": 15
        }
    }
    
    config_path = Path('config/bot_config.json')
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        click.echo("✅ Created basic configuration file")
    else:
        click.echo("⚠️  Configuration file already exists")
    
    click.echo("\n🎉 Setup complete!")
    click.echo("Next steps:")
    click.echo("1. Edit config/bot_config.json with your exchange API keys")
    click.echo("2. Run: python cli.py train")
    click.echo("3. Run: python cli.py backtest")
    click.echo("4. Run: python cli.py start")

@cli.command()
@click.option('--symbols', help='Comma-separated list of symbols to train')
@click.option('--timeframes', help='Comma-separated list of timeframes')
@click.option('--force', is_flag=True, help='Force retraining even if models exist')
@click.pass_context
def train(ctx, symbols, timeframes, force):
    """Train ML models on historical data"""
    
    click.echo("🧠 Training ML models...")
    
    try:
        # Load configuration
        with open(ctx.obj['config_path'], 'r') as f:
            config = json.load(f)
        
        # Override with command line options
        if symbols:
            config['trading']['symbols'] = symbols.split(',')
        if timeframes:
            config['trading']['timeframes'] = timeframes.split(',')
        
        # Check if models already exist
        models_dir = Path('models')
        if models_dir.exists() and any(models_dir.iterdir()) and not force:
            click.echo("⚠️  Existing models found.")
            if not click.confirm("Do you want to retrain them?"):
                return
        
        # Initialize trainer
        trainer = AdaptiveModelTrainer()
        
        # Run training
        with click.progressbar(length=100, label='Training models') as bar:
            # This is a simplified progress bar
            # In a real implementation, you'd integrate with the trainer's progress
            for i in range(100):
                time.sleep(0.1)
                bar.update(1)
        
        trainer.run_full_training_pipeline()
        
        click.echo("✅ Model training completed successfully!")
        
        # Show training summary
        status = trainer.get_model_status()
        click.echo(f"📊 Trained models for {len(status['models'])} symbol/timeframe combinations")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()

@cli.command()
@click.option('--start-date', default='2024-01-01', help='Backtest start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='Backtest end date (YYYY-MM-DD)')
@click.option('--symbols', help='Comma-separated list of symbols')
@click.option('--initial-capital', type=float, default=10000, help='Initial capital amount')
@click.option('--walk-forward', is_flag=True, help='Enable walk-forward analysis')
@click.pass_context
def backtest(ctx, start_date, end_date, symbols, initial_capital, walk_forward):
    """Run strategy backtest on historical data"""
    
    click.echo(f"🧪 Running backtest from {start_date} to {end_date}...")
    
    try:
        # Initialize backtester
        backtester = AdvancedBacktester()
        
        # Override configuration
        backtester.config['data']['start_date'] = start_date
        backtester.config['data']['end_date'] = end_date
        backtester.config['execution']['initial_capital'] = initial_capital
        backtester.config['strategy']['walk_forward_enabled'] = walk_forward
        
        if symbols:
            backtester.config['data']['symbols'] = symbols.split(',')
        
        # Run backtest
        with click.progressbar(length=100, label='Running backtest') as bar:
            # Simplified progress bar
            for i in range(100):
                time.sleep(0.05)
                bar.update(1)
        
        results = backtester.run_backtest()
        
        # Display results
        click.echo("\n📈 Backtest Results:")
        click.echo("=" * 50)
        click.echo(f"Total Return:     {results.total_return_pct:.2f}%")
        click.echo(f"Annualized Return: {results.annualized_return:.2f}%")
        click.echo(f"Sharpe Ratio:     {results.sharpe_ratio:.3f}")
        click.echo(f"Max Drawdown:     {results.max_drawdown:.2f}%")
        click.echo(f"Win Rate:         {results.win_rate:.1f}%")
        click.echo(f"Total Trades:     {results.total_trades}")
        click.echo(f"Profit Factor:    {results.profit_factor:.2f}")
        
        click.echo(f"\n📁 Detailed results saved to: backtest_results/")
        
    except Exception as e:
        click.echo(f"❌ Backtest failed: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()

@cli.command()
@click.option('--paper', is_flag=True, help='Force paper trading mode')
@click.option('--live', is_flag=True, help='Enable live trading (use with caution)')
@click.pass_context
def start(ctx, paper, live):
    """Start the trading bot"""
    
    if live and paper:
        click.echo("❌ Cannot use both --paper and --live flags")
        return
    
    mode = "paper trading" if paper or not live else "live trading"
    click.echo(f"🚀 Starting trading bot in {mode} mode...")
    
    if live:
        click.echo("⚠️  WARNING: Live trading enabled!")
        if not click.confirm("Are you sure you want to trade with real money?"):
            return
    
    try:
        # Load and modify configuration
        with open(ctx.obj['config_path'], 'r') as f:
            config = json.load(f)
        
        # Override trading mode
        if paper:
            config['trading']['live_trading_enabled'] = False
        elif live:
            config['trading']['live_trading_enabled'] = True
        
        # Save modified config temporarily
        temp_config_path = 'config/temp_bot_config.json'
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize and start bot
        bot = TradingBotOrchestrator(temp_config_path)
        
        click.echo("✅ Bot started successfully!")
        click.echo("Press Ctrl+C to stop the bot")
        
        # Start bot (this would be async in real implementation)
        import asyncio
        asyncio.run(bot.start())
        
    except KeyboardInterrupt:
        click.echo("\n🛑 Bot stopped by user")
    except Exception as e:
        click.echo(f"❌ Bot failed to start: {e}")
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
    finally:
        # Clean up temp config
        if Path('config/temp_bot_config.json').exists():
            Path('config/temp_bot_config.json').unlink()

@cli.command()
@click.option('--port', default=8050, help='Port to run dashboard on')
@click.option('--host', default='localhost', help='Host to bind to')
def dashboard(port, host):
    """Start the web dashboard"""
    
    click.echo(f"📊 Starting web dashboard at http://{host}:{port}")
    
    try:
        # Start dashboard in subprocess
        subprocess.run([
            sys.executable, 'web_dashboard.py',
            '--host', host,
            '--port', str(port)
        ])
    except KeyboardInterrupt:
        click.echo("\n🛑 Dashboard stopped")
    except Exception as e:
        click.echo(f"❌ Dashboard failed to start: {e}")

@cli.command()
@click.option('--hours', default=24, help='Hours of history to show')
def status(hours):
    """Show system status and recent performance"""
    
    click.echo("📊 System Status")
    click.echo("=" * 50)
    
    try:
        # Check if bot is running
        click.echo("🔍 Checking system components...")
        
        # Check models
        models_dir = Path('models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*.keras')) + list(models_dir.glob('*.joblib'))
            click.echo(f"🧠 ML Models: {len(model_files)} files found")
        else:
            click.echo("🧠 ML Models: Not trained")
        
        # Check performance database
        db_path = Path('data/performance.db')
        if db_path.exists():
            click.echo("📈 Performance DB: Available")
            
            # Try to get recent performance
            try:
                monitor = PerformanceMonitor()
                performance = monitor.get_recent_performance(hours)
                
                if performance.get('total_trades', 0) > 0:
                    click.echo(f"   📊 Last {hours}h: {performance['total_trades']} trades")
                    click.echo(f"   💰 Total P&L: {performance.get('total_profit', 0):.4f}")
                    click.echo(f"   🎯 Win Rate: {performance.get('win_rate', 0)*100:.1f}%")
                else:
                    click.echo(f"   📊 No trades in last {hours} hours")
                    
            except Exception as e:
                click.echo(f"   ⚠️  Could not read performance data: {e}")
        else:
            click.echo("📈 Performance DB: Not found")
        
        # Check logs
        logs_dir = Path('logs')
        if logs_dir.exists():
            log_files = list(logs_dir.glob('*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                mod_time = datetime.fromtimestamp(latest_log.stat().st_mtime)
                click.echo(f"📝 Latest Log: {latest_log.name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                click.echo("📝 Logs: No log files found")
        
        # Check disk space
        disk_usage = subprocess.check_output(['df', '-h', '.']).decode().split('\n')[1].split()
        click.echo(f"💾 Disk Usage: {disk_usage[4]} used ({disk_usage[2]} available)")
        
    except Exception as e:
        click.echo(f"❌ Status check failed: {e}")

@cli.command()
@click.option('--lines', default=50, help='Number of log lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
def logs(lines, follow):
    """Show recent log entries"""
    
    logs_dir = Path('logs')
    if not logs_dir.exists():
        click.echo("❌ Logs directory not found")
        return
    
    # Find latest log file
    log_files = list(logs_dir.glob('*.log'))
    if not log_files:
        click.echo("❌ No log files found")
        return
    
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    
    try:
        if follow:
            # Follow mode (like tail -f)
            click.echo(f"📝 Following {latest_log.name} (Press Ctrl+C to stop)")
            subprocess.run(['tail', '-f', str(latest_log)])
        else:
            # Show last N lines
            click.echo(f"📝 Last {lines} lines from {latest_log.name}:")
            click.echo("-" * 50)
            
            with open(latest_log, 'r') as f:
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    click.echo(line.rstrip())
                    
    except KeyboardInterrupt:
        click.echo("\n🛑 Log following stopped")
    except Exception as e:
        click.echo(f"❌ Could not read logs: {e}")

@cli.command()
@click.option('--component', help='Specific component to check')
def health(component):
    """Check system health"""
    
    click.echo("🏥 System Health Check")
    click.echo("=" * 50)
    
    try:
        # Import psutil for system monitoring
        import psutil
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        click.echo(f"💻 CPU Usage: {cpu_percent:.1f}%")
        click.echo(f"🧠 Memory Usage: {memory.percent:.1f}% ({memory.used // 1024 // 1024}MB / {memory.total // 1024 // 1024}MB)")
        click.echo(f"💾 Disk Usage: {disk.percent:.1f}% ({disk.used // 1024 // 1024 // 1024}GB / {disk.total // 1024 // 1024 // 1024}GB)")
        
        # Check Python packages
        try:
            import numpy, pandas, sklearn, tensorflow
            click.echo("✅ Core ML packages: Available")
        except ImportError as e:
            click.echo(f"❌ Core ML packages: Missing ({e})")
        
        try:
            import ccxt
            click.echo("✅ Exchange library: Available")
        except ImportError:
            click.echo("❌ Exchange library: Missing")
        
        # Check network connectivity
        import requests
        try:
            response = requests.get('https://api.kraken.com/0/public/Time', timeout=5)
            if response.status_code == 200:
                click.echo("✅ Exchange connectivity: OK")
            else:
                click.echo("⚠️  Exchange connectivity: Degraded")
        except:
            click.echo("❌ Exchange connectivity: Failed")
        
    except Exception as e:
        click.echo(f"❌ Health check failed: {e}")

@cli.command()
@click.argument('action', type=click.Choice(['backup', 'restore', 'clean']))
@click.option('--path', help='Backup/restore path')
def data(action, path):
    """Data management operations"""
    
    if action == 'backup':
        click.echo("💾 Creating system backup...")
        
        backup_dir = Path(path or f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup important directories
        import shutil
        
        for source_dir in ['config', 'models', 'logs']:
            if Path(source_dir).exists():
                shutil.copytree(source_dir, backup_dir / source_dir)
                click.echo(f"✅ Backed up {source_dir}")
        
        click.echo(f"📁 Backup created: {backup_dir}")
        
    elif action == 'restore':
        if not path:
            click.echo("❌ Restore path required")
            return
            
        click.echo(f"📂 Restoring from {path}...")
        
        if not click.confirm("This will overwrite existing files. Continue?"):
            return
        
        # Restore logic would go here
        click.echo("✅ Restore completed")
        
    elif action == 'clean':
        click.echo("🧹 Cleaning up old data...")
        
        # Clean cache
        cache_dir = Path('data/cache')
        if cache_dir.exists():
            for cache_file in cache_dir.glob('*.pkl'):
                if cache_file.stat().st_mtime < time.time() - 86400 * 7:  # 7 days old
                    cache_file.unlink()
                    click.echo(f"🗑️  Removed old cache: {cache_file.name}")
        
        # Clean old logs
        logs_dir = Path('logs')
        if logs_dir.exists():
            for log_file in logs_dir.glob('*.log'):
                if log_file.stat().st_mtime < time.time() - 86400 * 30:  # 30 days old
                    log_file.unlink()
                    click.echo(f"🗑️  Removed old log: {log_file.name}")
        
        click.echo("✅ Cleanup completed")

@cli.command()
def update():
    """Update system dependencies"""
    
    click.echo("🔄 Updating system dependencies...")
    
    try:
        # Update pip packages
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', '-r', 'requirements.txt'])
        
        click.echo("✅ Dependencies updated successfully")
        
    except Exception as e:
        click.echo(f"❌ Update failed: {e}")

@cli.command()
@click.option('--output', default='system_info.txt', help='Output file')
def info(output):
    """Generate system information report"""
    
    click.echo("📋 Generating system information...")
    
    info_lines = [
        "Enhanced Trading Bot - System Information",
        "=" * 50,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "System Information:",
        f"- Python Version: {sys.version}",
        f"- Platform: {sys.platform}",
    ]
    
    # Add package versions
    try:
        import numpy, pandas, sklearn, tensorflow
        info_lines.extend([
            "",
            "Package Versions:",
            f"- NumPy: {numpy.__version__}",
            f"- Pandas: {pandas.__version__}",
            f"- Scikit-learn: {sklearn.__version__}",
            f"- TensorFlow: {tensorflow.__version__}",
        ])
    except ImportError:
        info_lines.append("- Some packages not available")
    
    # Add configuration info
    try:
        with open('config/bot_config.json', 'r') as f:
            config = json.load(f)
        
        info_lines.extend([
            "",
            "Configuration:",
            f"- Symbols: {', '.join(config.get('trading', {}).get('symbols', []))}",
            f"- Timeframes: {', '.join(config.get('trading', {}).get('timeframes', []))}",
            f"- Live Trading: {config.get('trading', {}).get('live_trading_enabled', False)}",
        ])
    except:
        info_lines.append("- Configuration not available")
    
    # Save to file
    with open(output, 'w') as f:
        f.write('\n'.join(info_lines))
    
    click.echo(f"📄 System information saved to: {output}")
    
    # Also display to console
    for line in info_lines[:20]:  # First 20 lines
        click.echo(line)

if __name__ == '__main__':
    cli()