import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def _str_to_bool(s: Optional[str]) -> bool:
    if s is None:
        return False
    return s.lower() in ('true', '1', 't', 'y', 'yes')

def _str_to_int(s: Optional[str], default: int) -> int:
    try:
        return int(s) if s else default
    except ValueError:
        logging.warning(f"Invalid integer value for {s}. Using default {default}.")
        return default

def _str_to_float(s: Optional[str], default: float) -> float:
    try:
        return float(s) if s else default
    except ValueError:
        logging.warning(f"Invalid float value for {s}. Using default {default}.")
        return default

# Enhanced configuration classes (FreqTrade-style)
@dataclass
class TradingConfig:
    """FreqTrade-style trading configuration"""
    max_open_trades: int = 3
    stake_amount: Union[float, str] = 100.0  # Can be "unlimited"
    stake_currency: str = "USDT"
    dry_run: bool = True
    dry_run_wallet: float = 10000.0
    trading_mode: str = "spot"  # spot, margin, futures
    margin_mode: str = "cross"  # cross, isolated
    timeframe: str = "1h"
    # Risk management
    max_risk_per_trade: float = 0.02  # 2%
    max_total_risk: float = 0.10  # 10%
    max_correlation_exposure: float = 0.50  # 50%
    max_daily_drawdown: float = 0.05  # 5%
    max_daily_trades: int = 10
    # Position sizing
    position_adjustment_enable: bool = False
    max_entry_position_adjustment: int = -1  # -1 for unlimited

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str = "binance"
    key: str = ""
    secret: str = ""
    password: str = ""  # For some exchanges
    sandbox: bool = True
    trading_fees: float = 0.001
    pair_whitelist: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    pair_blacklist: List[str] = field(default_factory=list)
    # CCXT specific options
    options: Dict[str, Any] = field(default_factory=dict)
    # Rate limiting
    rateLimit: Optional[int] = None
    enableRateLimit: bool = True
    enabled: bool = True  # Allow config files to set this
    api_key: Optional[str] = None  # For compatibility with config files
    api_secret: Optional[str] = None  # For compatibility

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str = "MLStrategy"
    timeframe: str = "1h"
    startup_candle_count: int = 200
    process_only_new_candles: bool = True
    use_exit_signal: bool = True
    exit_profit_only: bool = False
    ignore_roi_if_buy_signal: bool = False
    # Advanced strategy options
    disable_dataframe_checks: bool = False
    use_custom_stoploss: bool = False
    default_symbol: str = "BTC/USDT"  # For compatibility with some configs

@dataclass
class MLConfig:
    """ML-specific configuration"""
    enabled: bool = True
    model_save_path: str = "./models"
    retrain_interval: int = 24  # hours
    feature_engineering: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)
    # FreqAI-style settings
    startup_candles: int = 400
    data_kitchen_thread_count: int = 4
    write_metrics_to_disk: bool = True

@dataclass
class NotificationConfig:
    """Notification configuration"""
    enabled: bool = False
    # Telegram
    telegram_enabled: bool = False
    telegram_token: str = ""
    telegram_chat_id: str = ""
    # Slack
    slack_enabled: bool = False
    slack_webhook: str = ""
    # Discord
    discord_enabled: bool = False
    discord_webhook: str = ""
    # Email
    email_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipient_email: str = ""

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    enable_protections: bool = False
    backtest_cache: str = "day"
    export: str = "none"  # none, trades, signals
    breakdown: List[str] = field(default_factory=list)

@dataclass
class APIConfig:
    """API server configuration"""
    enabled: bool = True
    listen_ip_address: str = "127.0.0.1"
    listen_port: int = 8000
    verbosity: str = "error"
    enable_openapi: bool = True
    jwt_secret_key: str = "secret-key"
    username: str = "admin"
    password: str = "password"
    # CORS settings
    CORS_origins: List[str] = field(default_factory=lambda: ["*"])

# Original Settings class (maintained for backward compatibility)
class Settings:
    """Centralized configuration settings for the application (original)."""

    # --- APP METADATA ---
    APP_NAME: str = os.getenv("APP_NAME", "Industrial Trading Bot")

    # --- API KEYS & CREDENTIALS ---
    GOOGLE_AI_API_KEY: Optional[str] = os.getenv('GOOGLE_AI_API_KEY')
    GOOGLE_AI_MODEL: str = os.getenv('GOOGLE_AI_MODEL', 'gemini-pro')
    GOOGLE_AI_ENABLED: bool = _str_to_bool(os.getenv('GOOGLE_AI_ENABLED', 'True'))

    ALPACA_API_KEY: Optional[str] = os.getenv('APCA_API_KEY_ID')
    ALPACA_SECRET_KEY: Optional[str] = os.getenv('APCA_SECRET_KEY')
    ALPACA_BASE_URL: str = os.getenv('ALPACA_PAPER_BASE_URL', 'https://paper-api.alpaca.markets')
    ALPACA_STREAM_DATA_URL: str = os.getenv('ALPACA_STREAM_DATA_URL', 'wss://stream.data.alpaca.markets/v1beta3/crypto/us')

    COINMARKETCAP_API_KEY: Optional[str] = os.getenv('COINMARKETCAP_API_KEY')

    # --- GENERAL BOT SETTINGS ---
    DEFAULT_EXCHANGE: str = os.getenv('DEFAULT_EXCHANGE', 'kraken').lower()
    DEFAULT_MODEL_SAVE_PATH: str = os.getenv('DEFAULT_MODEL_SAVE_PATH', 'models/')

    CCXT_AVAILABLE: bool = _str_to_bool(os.getenv('CCXT_AVAILABLE', 'true'))

    BROADCAST_INTERVAL_SECONDS: int = _str_to_int(os.getenv('BROADCAST_INTERVAL_SECONDS'), 15)
    ERROR_RETRY_INTERVAL_SECONDS: int = _str_to_int(os.getenv('ERROR_RETRY_INTERVAL_SECONDS'), 30)

    STRATEGIES_DIR: str = os.getenv('STRATEGIES_DIR', "strategies")
    ALPACA_SYNC_INTERVAL_MINUTES: int = _str_to_int(os.getenv('ALPACA_SYNC_INTERVAL_MINUTES'), 5)

    DEFAULT_TRAINING_SYMBOLS: List[str] = os.getenv('DEFAULT_TRAINING_SYMBOLS', 'BTC/USD,ETH/USD,ADA/USD').split(',')
    DEFAULT_TRAINING_SYMBOLS = [s.strip() for s in DEFAULT_TRAINING_SYMBOLS]

    APP_USER_ID: str = os.getenv('APP_USER_ID', 'admin')
    APP_PASSWORD: str = os.getenv('APP_PASSWORD', 'admin123')

    DEBUG: bool = _str_to_bool(os.getenv('DEBUG', 'False'))
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO').upper()

    # --- NOTIFICATION SYSTEM CONFIGURATION ---
    SLACK_WEBHOOK_URL: Optional[str] = os.getenv('SLACK_WEBHOOK_URL')
    SLACK_CHANNEL: Optional[str] = os.getenv('SLACK_CHANNEL')
    SLACK_USERNAME: Optional[str] = os.getenv('SLACK_USERNAME')

    SMTP_SERVER: Optional[str] = os.getenv('SMTP_SERVER')
    SMTP_PORT: int = _str_to_int(os.getenv('SMTP_PORT'), 587)
    SENDER_EMAIL: Optional[str] = os.getenv('SENDER_EMAIL')
    SENDER_PASSWORD: Optional[str] = os.getenv('SENDER_PASSWORD')
    RECIPIENT_EMAIL: Optional[str] = os.getenv('RECIPIENT_EMAIL')

    DISCORD_WEBHOOK_URL: Optional[str] = os.getenv('DISCORD_WEBHOOK_URL')

    TWILIO_ACCOUNT_SID: Optional[str] = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN: Optional[str] = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER: Optional[str] = os.getenv('TWILIO_PHONE_NUMBER')
    RECIPIENT_PHONE_NUMBER: Optional[str] = os.getenv('RECIPIENT_PHONE_NUMBER')

    # --- NOTIFICATION SETTINGS ---
    NOTIFY_TRADES: bool = _str_to_bool(os.getenv('NOTIFY_TRADES', 'true'))
    NOTIFY_SYSTEM_EVENTS: bool = _str_to_bool(os.getenv('NOTIFY_SYSTEM_EVENTS', 'true'))
    NOTIFY_ERRORS: bool = _str_to_bool(os.getenv('NOTIFY_ERRORS', 'true'))
    NOTIFY_PERFORMANCE: bool = _str_to_bool(os.getenv('NOTIFY_PERFORMANCE', 'true'))
    NOTIFY_STRATEGY_CHANGES: bool = _str_to_bool(os.getenv('NOTIFY_STRATEGY_CHANGES', 'true'))

    PERFORMANCE_NOTIFICATION_INTERVAL: int = _str_to_int(os.getenv('PERFORMANCE_NOTIFICATION_INTERVAL'), 3600)
    ERROR_NOTIFICATION_COOLDOWN: int = _str_to_int(os.getenv('ERROR_NOTIFICATION_COOLDOWN'), 300)
    TRADE_NOTIFICATION_MIN_VALUE: float = _str_to_float(os.getenv('TRADE_NOTIFICATION_MIN_VALUE'), 10.0)

    MIN_NOTIFICATION_PRIORITY: str = os.getenv('MIN_NOTIFICATION_PRIORITY', 'MEDIUM').upper()

    NOTIFICATION_HISTORY_MAX_LENGTH: int = _str_to_int(os.getenv('NOTIFICATION_HISTORY_MAX_LENGTH'), 100)

    # Database URL
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///tradesv3.sqlite")

    @property
    def ALPACA_ENABLED(self) -> bool:
        return bool(self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY and self.ALPACA_BASE_URL)

# Enhanced Settings class (new FreqTrade-style configuration)
@dataclass
class EnhancedSettings:
    """Enhanced settings with FreqTrade-style organization"""
    # Core configuration sections
    trading: TradingConfig = field(default_factory=TradingConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    api_server: APIConfig = field(default_factory=APIConfig)
    # Paths
    user_data_dir: Path = field(default_factory=lambda: Path("./user_data"))
    log_level: str = "INFO"
    # Database
    db_url: str = os.getenv("DATABASE_URL", "sqlite:///tradesv3.sqlite")
    # Import configurations from other files
    add_config_files: List[str] = field(default_factory=list)
    # Original settings for backward compatibility
    original: Optional[Settings] = field(default_factory=Settings)

    @property
    def DATABASE_URL(self):
        return self.db_url

    def __getattr__(self, name):
        """Handle getattr calls for backward compatibility"""
        # Handle DATABASE_URL specifically
        if name == 'DATABASE_URL':
            return self.db_url
        
        # Check if it's in the original settings
        if hasattr(self, 'original') and self.original:
            if hasattr(self.original, name):
                return getattr(self.original, name)
        
        # Check nested attributes
        if '.' in name:
            parts = name.split('.')
            obj = self
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            return obj
        
        # Default behavior
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def get(self, key, default=None):
        """Dictionary-style get method"""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

class ConfigManager:
    """Enhanced configuration manager with FreqTrade features"""

    def __init__(self):
        self.config = EnhancedSettings()
        self._config_files: List[Path] = []

    def load_config(self, config_files: Union[str, List[str]] = None) -> EnhancedSettings:
        """
        Load configuration from files with validation
        Supports multiple config files that get merged
        """
        if config_files is None:
            # Try to find default config files
            default_files = ["config.json", "user_data/config.json"]
            config_files = [f for f in default_files if os.path.exists(f)]

        if isinstance(config_files, str):
            config_files = [config_files]

        if not config_files:
            logging.warning("No configuration files found, using defaults")
            self.config = self._create_default_config()
            return self.config

        merged_config = {}

        for config_file in config_files:
            config_path = Path(config_file)
            if not config_path.exists():
                logging.warning(f"Config file {config_path} does not exist")
                continue

            logging.info(f"Loading configuration from {config_path}")

            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                # Merge configurations (later files override earlier ones)
                merged_config = self._deep_merge(merged_config, file_config)
                self._config_files.append(config_path)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON in config file {config_path}: {e}")
                raise
            except Exception as e:
                logging.error(f"Error loading config file {config_path}: {e}")
                raise

        # Load additional config files if specified
        if 'add_config_files' in merged_config:
            for additional_file in merged_config['add_config_files']:
                try:
                    additional_config = self.load_config(additional_file)
                    merged_config = self._deep_merge(merged_config, self._settings_to_dict(additional_config))
                except Exception as e:
                    logging.error(f"Failed to load additional config file {additional_file}: {e}")

        # Apply environment variable overrides
        merged_config = self._apply_env_overrides(merged_config)

        # Convert to settings object
        self.config = self._dict_to_settings(merged_config)

        return self.config

    def _create_default_config(self) -> EnhancedSettings:
        """Create default configuration"""
        config = EnhancedSettings()
        # Apply environment variables to default config
        config.exchange.name = os.getenv('DEFAULT_EXCHANGE', 'kraken').lower()
        config.exchange.key = os.getenv('EXCHANGE_KEY', '')
        config.exchange.secret = os.getenv('EXCHANGE_SECRET', '')
        config.exchange.sandbox = _str_to_bool(os.getenv('EXCHANGE_SANDBOX', 'true'))

        config.trading.dry_run = _str_to_bool(os.getenv('DRY_RUN', 'true'))
        config.trading.dry_run_wallet = _str_to_float(os.getenv('DRY_RUN_WALLET'), 10000.0)
        config.trading.max_open_trades = _str_to_int(os.getenv('MAX_OPEN_TRADES'), 3)
        config.trading.stake_amount = _str_to_float(os.getenv('STAKE_AMOUNT'), 100.0)

        config.strategy.name = os.getenv('STRATEGY_NAME', 'MLStrategy')
        config.strategy.timeframe = os.getenv('STRATEGY_TIMEFRAME', '1h')

        config.ml.enabled = _str_to_bool(os.getenv('ML_ENABLED', 'true'))
        config.ml.model_save_path = os.getenv('MODEL_SAVE_PATH', './models')

        # Copy from original settings for backward compatibility
        original = Settings()
        config.original = original

        return config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides
        Format: FREQTRADE__SECTION__KEY=value or TRADING_BOT__SECTION__KEY=value
        """
        prefixes = ["FREQTRADE__", "TRADING_BOT__"]

        for env_key, env_value in os.environ.items():
            for prefix in prefixes:
                if env_key.startswith(prefix):
                    # Parse environment variable path
                    path_parts = env_key.replace(prefix, "").lower().split("__")
                    # Navigate to the correct config section
                    current_config = config
                    for part in path_parts[:-1]:
                        if part not in current_config:
                            current_config[part] = {}
                        current_config = current_config[part]
                    # Set the value (try to parse as JSON first for complex types)
                    try:
                        parsed_value = json.loads(env_value)
                    except json.JSONDecodeError:
                        # Try to convert to appropriate type
                        if env_value.lower() in ('true', 'false'):
                            parsed_value = env_value.lower() == 'true'
                        elif env_value.isdigit():
                            parsed_value = int(env_value)
                        elif env_value.replace('.', '').isdigit():
                            parsed_value = float(env_value)
                        else:
                            parsed_value = env_value
                    current_config[path_parts[-1]] = parsed_value
                    logging.info(f"Applied environment override: {env_key}")
                    break
        return config

    def _dict_to_settings(self, config_dict: Dict[str, Any]) -> EnhancedSettings:
        """Convert dictionary to settings object"""
        settings = EnhancedSettings()
        # Map configuration sections
        if 'trading' in config_dict:
            settings.trading = TradingConfig(**config_dict['trading'])
        if 'exchange' in config_dict:
            settings.exchange = ExchangeConfig(**config_dict['exchange'])
        if 'strategy' in config_dict:
            settings.strategy = StrategyConfig(**config_dict['strategy'])
        if 'ml' in config_dict:
            settings.ml = MLConfig(**config_dict['ml'])
        if 'notifications' in config_dict:
            settings.notifications = NotificationConfig(**config_dict['notifications'])
        if 'backtest' in config_dict:
            settings.backtest = BacktestConfig(**config_dict['backtest'])
        if 'api_server' in config_dict:
            api_config = config_dict['api_server']
            settings.api_server = APIConfig(**api_config)
        # Direct mappings
        if 'user_data_dir' in config_dict:
            settings.user_data_dir = Path(config_dict['user_data_dir'])
        if 'log_level' in config_dict:
            settings.log_level = config_dict['log_level']
        if 'db_url' in config_dict:
            settings.db_url = config_dict['db_url']
        elif 'DATABASE_URL' in config_dict:
            settings.db_url = config_dict['DATABASE_URL']
        return settings

    def _settings_to_dict(self, settings: EnhancedSettings) -> Dict[str, Any]:
        """Convert settings object back to dictionary"""
        return {
            'trading': settings.trading.__dict__,
            'exchange': settings.exchange.__dict__,
            'strategy': settings.strategy.__dict__,
            'ml': settings.ml.__dict__,
            'notifications': settings.notifications.__dict__,
            'backtest': settings.backtest.__dict__,
            'api_server': settings.api_server.__dict__,
            'user_data_dir': str(settings.user_data_dir),
            'log_level': settings.log_level,
            'db_url': settings.db_url
        }

    def save_config(self, config_path: str = "config.json"):
        """Save current configuration to file"""
        config_dict = self._settings_to_dict(self.config)
        os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        logging.info(f"Configuration saved to {config_path}")

    def validate_config(self) -> List[str]:
        """Validate current configuration and return any issues"""
        issues = []
        # Check required paths exist
        if not self.config.user_data_dir.exists():
            self.config.user_data_dir.mkdir(parents=True, exist_ok=True)
        if not Path(self.config.ml.model_save_path).exists():
            Path(self.config.ml.model_save_path).mkdir(parents=True, exist_ok=True)
        # Validate exchange configuration
        if not self.config.exchange.name:
            issues.append("Exchange name is required")
        if not self.config.trading.dry_run and not self.config.exchange.key:
            issues.append("Exchange API key required for live trading")
        # Validate strategy
        if not self.config.strategy.name:
            issues.append("Strategy name is required")
        # Validate trading parameters
        if self.config.trading.max_open_trades <= 0:
            issues.append("max_open_trades must be greater than 0")
        if isinstance(self.config.trading.stake_amount, (int, float)) and self.config.trading.stake_amount <= 0:
            issues.append("stake_amount must be greater than 0")
        # Validate database URL
        if not self.config.db_url:
            issues.append("DATABASE_URL (db_url) must be set")
        return issues

# Helper function for safe attribute access
def safe_getattr(obj, attr_path, default=None):
    """Safely get nested attributes"""
    try:
        if isinstance(obj, dict):
            # Handle dictionary access
            keys = attr_path.split('.')
            value = obj
            for key in keys:
                value = value.get(key, default)
                if value is default:
                    return default
            return value
        else:
            # Handle object attribute access
            attrs = attr_path.split('.')
            value = obj
            for attr in attrs:
                value = getattr(value, attr, default)
                if value is default:
                    return default
            return value
    except (AttributeError, KeyError, TypeError):
        return default

# Create global instances
settings = Settings()  # Original settings for backward compatibility
config_manager = ConfigManager()  # New enhanced config manager

# Temporary debugging (keep for now)
if settings.GOOGLE_AI_API_KEY:
    print(f"DEBUG: GOOGLE_AI_API_KEY loaded: '{settings.GOOGLE_AI_API_KEY[:10]}...' (Length: {len(settings.GOOGLE_AI_API_KEY)})")
else:
    print("DEBUG: GOOGLE_AI_API_KEY not found or empty")
print(f"DEBUG: GOOGLE_AI_ENABLED loaded: {settings.GOOGLE_AI_ENABLED}")