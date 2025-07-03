#!/usr/bin/env python3
"""
migration_helper.py - Enhanced Trading Bot Migration Helper

This script helps migrate configurations from older trading bot setups
to the new Enhanced Trading Bot environment format.
"""

import os
import re
import json
import argparse
import shutil
from datetime import datetime
from pathlib import Path

class TradingBotMigrator:
    """Handle migration of trading bot configurations"""
    
    def __init__(self):
        self.backup_dir = "migration_backups"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_backup(self, source_file):
        """Create backup of source file"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            
        if os.path.exists(source_file):
            backup_name = f"{os.path.basename(source_file)}.backup.{self.timestamp}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            shutil.copy2(source_file, backup_path)
            print(f"✅ Backup created: {backup_path}")
            return backup_path
        return None
    
    def parse_env_file(self, file_path):
        """Parse existing .env file and extract key-value pairs"""
        env_vars = {}
        
        if not os.path.exists(file_path):
            print(f"❌ Source file not found: {file_path}")
            return env_vars
            
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # Parse key=value pairs
                if '=' in line:
                    # Handle lines with = in them properly
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                        
                    env_vars[key] = value
        
        print(f"✅ Parsed {len(env_vars)} environment variables from {file_path}")
        return env_vars
    
    def generate_enhanced_env(self, source_vars):
        """Generate enhanced .env file with migrated values"""
        
        # Enhanced .env template with placeholders
        enhanced_template = """################################################################################
# IMPORTANT: DO NOT COMMIT THIS FILE TO PUBLIC VERSION CONTROL               #
# This file contains sensitive API keys and personal credentials.             #
# Migrated from existing configuration on {timestamp}                         #
################################################################################

# =============================================================================
# DATABASE CONNECTION
# =============================================================================
DATABASE_URL={database_url}

# =============================================================================
# APP METADATA  
# =============================================================================
APP_NAME="{app_name}"
APP_USER_ID={app_user_id}
APP_PASSWORD={app_password}

# =============================================================================
# EXCHANGE API CREDENTIALS
# =============================================================================

# Kraken Exchange API Credentials (PRIMARY)
KRAKEN_API_KEY={kraken_api_key}
KRAKEN_SECRET={kraken_secret}
KRAKEN_SANDBOX={kraken_sandbox}

# Alpaca API Credentials (for stock/crypto trading)
APCA_API_KEY_ID={alpaca_key_id}
APCA_API_SECRET_KEY={alpaca_secret_key}
ALPACA_PAPER_BASE_URL={alpaca_paper_url}
ALPACA_STREAM_DATA_URL={alpaca_stream_url}

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================
LIVE_TRADING_ENABLED={live_trading_enabled}
DEFAULT_EXCHANGE={default_exchange}
TRADING_SYMBOLS={trading_symbols}
DEFAULT_TRAINING_SYMBOLS={default_training_symbols}
TRADING_TIMEFRAMES=1h,4h,1d
INITIAL_CAPITAL=10000
CCXT_AVAILABLE={ccxt_available}

# Sync and Update Intervals
ALPACA_SYNC_INTERVAL_MINUTES={alpaca_sync_interval}
BROADCAST_INTERVAL_SECONDS={broadcast_interval}
ERROR_RETRY_INTERVAL_SECONDS={error_retry_interval}

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
MAX_RISK_PER_TRADE=0.02
MAX_PORTFOLIO_RISK=0.10
EMERGENCY_STOP_LOSS=0.15
TRADE_NOTIFICATION_MIN_VALUE={trade_notification_min_value}

# =============================================================================
# AI AND EXTERNAL API KEYS
# =============================================================================
GOOGLE_AI_API_KEY={google_ai_api_key}
GOOGLE_AI_ENABLED={google_ai_enabled}
GOOGLE_AI_MODEL={google_ai_model}
COINMARKETCAP_API_KEY={coinmarketcap_api_key}

# Enhanced Chat Settings
CHAT_MEMORY_SIZE={chat_memory_size}
CHAT_VOICE_ENABLED={chat_voice_enabled}
CHAT_PROACTIVE_INSIGHTS={chat_proactive_insights}

# =============================================================================
# NOTIFICATIONS CONFIGURATION
# =============================================================================

# Email Notifications
EMAIL_ENABLED=false
SMTP_SERVER={smtp_server}
SMTP_PORT={smtp_port}
SENDER_EMAIL={sender_email}
SENDER_PASSWORD={sender_password}
RECIPIENT_EMAIL={recipient_email}

# Slack Notifications
SLACK_WEBHOOK_URL={slack_webhook_url}
SLACK_CHANNEL={slack_channel}
SLACK_USERNAME={slack_username}

# Discord Notifications
DISCORD_WEBHOOK_URL={discord_webhook_url}

# SMS Notifications (Twilio)
TWILIO_ACCOUNT_SID={twilio_account_sid}
TWILIO_AUTH_TOKEN={twilio_auth_token}
TWILIO_PHONE_NUMBER={twilio_phone_number}
RECIPIENT_PHONE_NUMBER={recipient_phone_number}

# =============================================================================
# NOTIFICATION BEHAVIOR SETTINGS
# =============================================================================
NOTIFY_TRADES={notify_trades}
NOTIFY_SYSTEM_EVENTS={notify_system_events}
NOTIFY_ERRORS={notify_errors}
NOTIFY_PERFORMANCE={notify_performance}
NOTIFY_STRATEGY_CHANGES={notify_strategy_changes}

# Notification Timing and Filtering
PERFORMANCE_NOTIFICATION_INTERVAL={performance_notification_interval}
ERROR_NOTIFICATION_COOLDOWN={error_notification_cooldown}
MIN_NOTIFICATION_PRIORITY={min_notification_priority}
NOTIFICATION_HISTORY_MAX_LENGTH={notification_history_max_length}

# =============================================================================
# SECURITY
# =============================================================================
SECRET_KEY={secret_key}
JWT_SECRET={jwt_secret}

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================
LOG_LEVEL={log_level}
DEBUG={debug}
DEVELOPMENT_MODE=true
FLASK_DEBUG=true

# Directory Paths
DEFAULT_MODEL_SAVE_PATH={default_model_save_path}
STRATEGIES_DIR={strategies_dir}

# =============================================================================
# MIGRATION NOTES
# =============================================================================
# This .env file was migrated from an existing configuration.
# Original configuration backed up to: migration_backups/
# Migration completed on: {timestamp}
# 
# Please review and update:
# 1. Verify all API credentials are correct
# 2. Update any placeholder values
# 3. Review notification settings
# 4. Test configuration before going live
"""
        
        # Map old values to new format
        migration_mapping = {
            'timestamp': self.timestamp,
            'database_url': source_vars.get('DATABASE_URL', 'sqlite:///data/enhanced_trading_bot.db'),
            'app_name': source_vars.get('APP_NAME', 'Enhanced Trading Bot'),
            'app_user_id': source_vars.get('APP_USER_ID', 'admin'),
            'app_password': source_vars.get('APP_PASSWORD', 'admin123'),
            
            # Exchange credentials
            'kraken_api_key': source_vars.get('KRAKEN_API_KEY', 'your_kraken_api_key_here'),
            'kraken_secret': source_vars.get('KRAKEN_SECRET', 'your_kraken_secret_here'),
            'kraken_sandbox': source_vars.get('KRAKEN_SANDBOX', 'true'),
            'alpaca_key_id': source_vars.get('APCA_API_KEY_ID', 'your_alpaca_key_id_here'),
            'alpaca_secret_key': source_vars.get('APCA_API_SECRET_KEY', 'your_alpaca_secret_here'),
            'alpaca_paper_url': source_vars.get('ALPACA_PAPER_BASE_URL', 'https://paper-api.alpaca.markets'),
            'alpaca_stream_url': source_vars.get('ALPACA_STREAM_DATA_URL', 'wss://stream.data.alpaca.markets/v1beta3/crypto/us'),
            
            # Trading config
            'live_trading_enabled': source_vars.get('LIVE_TRADING_ENABLED', 'false'),
            'default_exchange': source_vars.get('DEFAULT_EXCHANGE', 'kraken'),
            'trading_symbols': source_vars.get('TRADING_SYMBOLS', 'BTC/USD,ETH/USD,ADA/USD'),
            'default_training_symbols': source_vars.get('DEFAULT_TRAINING_SYMBOLS', 'BTC/USD,ETH/USD,ADA/USD'),
            'ccxt_available': source_vars.get('CCXT_AVAILABLE', 'true'),
            
            # Intervals
            'alpaca_sync_interval': source_vars.get('ALPACA_SYNC_INTERVAL_MINUTES', '5'),
            'broadcast_interval': source_vars.get('BROADCAST_INTERVAL_SECONDS', '15'),
            'error_retry_interval': source_vars.get('ERROR_RETRY_INTERVAL_SECONDS', '30'),
            'trade_notification_min_value': source_vars.get('TRADE_NOTIFICATION_MIN_VALUE', '10.0'),
            
            # AI and external APIs
            'google_ai_api_key': source_vars.get('GOOGLE_AI_API_KEY', 'your_gemini_api_key_here'),
            'google_ai_enabled': source_vars.get('GOOGLE_AI_ENABLED', 'true'),
            'google_ai_model': source_vars.get('GOOGLE_AI_MODEL', 'gemini-pro'),
            'coinmarketcap_api_key': source_vars.get('COINMARKETCAP_API_KEY', 'your_coinmarketcap_api_key_here'),
            
            # Chat settings
            'chat_memory_size': source_vars.get('CHAT_MEMORY_SIZE', '25'),
            'chat_voice_enabled': source_vars.get('CHAT_VOICE_ENABLED', 'true'),
            'chat_proactive_insights': source_vars.get('CHAT_PROACTIVE_INSIGHTS', 'true'),
            
            # Email settings
            'smtp_server': source_vars.get('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': source_vars.get('SMTP_PORT', '587'),
            'sender_email': source_vars.get('SENDER_EMAIL', 'your_email@gmail.com'),
            'sender_password': source_vars.get('SENDER_PASSWORD', 'your_gmail_app_password_here'),
            'recipient_email': source_vars.get('RECIPIENT_EMAIL', 'alerts@yourdomain.com'),
            
            # Notification services
            'slack_webhook_url': source_vars.get('SLACK_WEBHOOK_URL', 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'),
            'slack_channel': source_vars.get('SLACK_CHANNEL', '#trading-bot'),
            'slack_username': source_vars.get('SLACK_USERNAME', 'TradingBot'),
            'discord_webhook_url': source_vars.get('DISCORD_WEBHOOK_URL', 'https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK'),
            
            # SMS/Twilio
            'twilio_account_sid': source_vars.get('TWILIO_ACCOUNT_SID', 'your_twilio_account_sid_here'),
            'twilio_auth_token': source_vars.get('TWILIO_AUTH_TOKEN', 'your_twilio_auth_token_here'),
            'twilio_phone_number': source_vars.get('TWILIO_PHONE_NUMBER', 'your_twilio_phone_number_here'),
            'recipient_phone_number': source_vars.get('RECIPIENT_PHONE_NUMBER', 'your_recipient_phone_number_here'),
            
            # Notification behavior
            'notify_trades': source_vars.get('NOTIFY_TRADES', 'true'),
            'notify_system_events': source_vars.get('NOTIFY_SYSTEM_EVENTS', 'true'),
            'notify_errors': source_vars.get('NOTIFY_ERRORS', 'true'),
            'notify_performance': source_vars.get('NOTIFY_PERFORMANCE', 'true'),
            'notify_strategy_changes': source_vars.get('NOTIFY_STRATEGY_CHANGES', 'true'),
            
            # Notification timing
            'performance_notification_interval': source_vars.get('PERFORMANCE_NOTIFICATION_INTERVAL', '3600'),
            'error_notification_cooldown': source_vars.get('ERROR_NOTIFICATION_COOLDOWN', '300'),
            'min_notification_priority': source_vars.get('MIN_NOTIFICATION_PRIORITY', 'MEDIUM'),
            'notification_history_max_length': source_vars.get('NOTIFICATION_HISTORY_MAX_LENGTH', '100'),
            
            # Security
            'secret_key': source_vars.get('SECRET_KEY', 'change_this_secret_key_for_production_use'),
            'jwt_secret': source_vars.get('JWT_SECRET', 'change_this_jwt_secret_for_dashboard_auth'),
            
            # System settings
            'log_level': source_vars.get('LOG_LEVEL', 'INFO'),
            'debug': source_vars.get('DEBUG', 'false'),
            'default_model_save_path': source_vars.get('DEFAULT_MODEL_SAVE_PATH', 'models/'),
            'strategies_dir': source_vars.get('STRATEGIES_DIR', 'strategies')
        }
        
        # Generate the enhanced .env content
        enhanced_content = enhanced_template.format(**migration_mapping)
        
        return enhanced_content
    
    def migrate_database(self, source_db):
        """Migrate database if exists"""
        if os.path.exists(source_db):
            target_db = "data/enhanced_trading_bot.db"
            os.makedirs("data", exist_ok=True)
            
            # Create backup
            backup_db = os.path.join(self.backup_dir, f"database.backup.{self.timestamp}.sqlite")
            shutil.copy2(source_db, backup_db)
            print(f"✅ Database backup created: {backup_db}")
            
            # Copy to new location
            shutil.copy2(source_db, target_db)
            print(f"✅ Database migrated to: {target_db}")
            
            return target_db
        return None
    
    def migrate_configuration(self, source_env, target_env):
        """Main migration function"""
        print(f"🔄 Starting migration from {source_env} to {target_env}")
        
        # Create backup
        self.create_backup(source_env)
        
        # Parse source environment
        source_vars = self.parse_env_file(source_env)
        
        if not source_vars:
            print("❌ No variables found in source file")
            return False
        
        # Generate enhanced configuration
        enhanced_content = self.generate_enhanced_env(source_vars)
        
        # Write new configuration
        with open(target_env, 'w') as file:
            file.write(enhanced_content)
        
        print(f"✅ Enhanced configuration written to: {target_env}")
        
        # Migrate database if exists
        if 'DATABASE_URL' in source_vars:
            db_path = source_vars['DATABASE_URL'].replace('sqlite:///', '')
            if os.path.exists(db_path):
                self.migrate_database(db_path)
        
        # Check for tradesv3.sqlite
        if os.path.exists('tradesv3.sqlite'):
            self.migrate_database('tradesv3.sqlite')
        
        # Generate migration report
        self.generate_migration_report(source_vars, target_env)
        
        return True
    
    def generate_migration_report(self, source_vars, target_file):
        """Generate a migration report"""
        report_file = f"migration_report_{self.timestamp}.txt"
        
        with open(report_file, 'w') as file:
            file.write("Enhanced Trading Bot Migration Report\n")
            file.write("=" * 50 + "\n\n")
            file.write(f"Migration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Target Configuration: {target_file}\n")
            file.write(f"Backup Directory: {self.backup_dir}\n\n")
            
            file.write("Variables Migrated:\n")
            file.write("-" * 20 + "\n")
            for key, value in sorted(source_vars.items()):
                # Mask sensitive values
                if any(sensitive in key.upper() for sensitive in ['KEY', 'SECRET', 'PASSWORD', 'TOKEN']):
                    masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "*" * len(value)
                    file.write(f"{key}: {masked_value}\n")
                else:
                    file.write(f"{key}: {value}\n")
            
            file.write("\n\nNext Steps:\n")
            file.write("-" * 11 + "\n")
            file.write("1. Review the migrated .env file\n")
            file.write("2. Update any placeholder values\n")
            file.write("3. Test the configuration\n")
            file.write("4. Run the setup script if needed\n")
            file.write("5. Verify all API credentials\n")
        
        print(f"📋 Migration report generated: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Trading Bot Migration Helper")
    parser.add_argument("command", choices=["migrate"], help="Migration command")
    parser.add_argument("--source", required=True, help="Source .env file path")
    parser.add_argument("--target", required=True, help="Target .env file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    migrator = TradingBotMigrator()
    
    if args.command == "migrate":
        success = migrator.migrate_configuration(args.source, args.target)
        if success:
            print("\n🎉 Migration completed successfully!")
            print("\n📝 Next steps:")
            print("1. Review the migrated .env file")
            print("2. Update any placeholder values")
            print("3. Run the setup script: ./setup.sh")
            print("4. Test your configuration")
        else:
            print("❌ Migration failed")
            return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())