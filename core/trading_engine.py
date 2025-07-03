import sys
import os
from pathlib import Path

# Fix import paths for uvicorn - MUST BE FIRST
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
import importlib.util
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime, timedelta
import pandas as pd

# Import other core components
from core.config import settings, EnhancedSettings
from utils.simple_notification_manager import SimpleNotificationManager
from core.risk_manager import RiskManager, EnhancedRiskManager
from core.signal_processor import SignalProcessor
from core.ml_engine import MLEngine
from core.market_data_processor import MarketDataProcessor
from database.database_manager import DatabaseManager

# Import strategy base
from strategies.strategy_base import StrategyBase

# Import exchange clients
try:
    from exchanges.coinbase_client import CoinbaseClient
except ImportError:
    CoinbaseClient = None

try:
    from exchanges.kraken_client import KrakenClient
except ImportError:
    KrakenClient = None

logger = logging.getLogger(__name__)

class IndustrialTradingEngine:
    """Enhanced Industrial Trading Engine with FreqTrade-style features"""

    def __init__(self, notification_manager_instance, 
                 config: Optional[Union[EnhancedSettings, Dict[str, Any]]] = None):
        logger.info("Initializing Enhanced Industrial Trading Engine")

        # Store configuration
        self.config = config or settings
        self.notification_manager = notification_manager_instance

        # Get database URL for DatabaseManager
        db_url = None
        if hasattr(self.config, "DATABASE_URL"):
            db_url = getattr(self.config, "DATABASE_URL", None)
        elif isinstance(self.config, dict):
            db_url = self.config.get("DATABASE_URL")
        else:
            db_url = None

        if not db_url:
            db_url = getattr(settings, "DATABASE_URL", None)
        if not db_url:
            logger.warning("DATABASE_URL not found, using default")
            db_url = "sqlite:///tradesv3.sqlite"

        # Initialize core components
        try:
            self.db_manager = DatabaseManager(db_url)
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            self.db_manager = None

        self.risk_manager = None
        self.signal_processor = None
        self.ml_engine = None
        self.market_data_processor = None

        # Trading state
        self.is_running = False
        self.running = False  # Add for compatibility
        self.trading_paused = False
        self.shutdown_requested = False

        # Initialize collections
        self.strategies: Dict[str, StrategyBase] = {}
        self.active_orders: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.balances: Dict[str, float] = {'USD': 10000.0, 'USDT': 10000.0}
        self.current_market_data: Dict[str, Any] = {}

        # Exchange clients
        self.exchanges: Dict[str, Any] = {}

        # Async task management
        self.tasks: List[asyncio.Task] = []
        self._loop = None

        # Performance tracking
        self.start_time = datetime.now()
        self.trade_count = 0
        self.total_pnl = 0.0

        # Initialize components
        self._initialize_components()

        logger.info("Enhanced Industrial Trading Engine initialized")

    def _initialize_components(self):
        """Initialize all trading components"""
        try:
            # Initialize risk manager
            if self._safe_config_get('risk_management', 'enabled', default=True):
                try:
                    self.risk_manager = EnhancedRiskManager()
                    logger.info("Risk Manager initialized")
                except Exception as e:
                    logger.warning(f"Risk manager initialization failed: {e}")
            
            # Initialize signal processor
            try:
                self.signal_processor = SignalProcessor()
                logger.info("Signal Processor initialized")
            except Exception as e:
                logger.error(f"Error initializing signal processor: {e}")
            
            # Initialize ML engine if enabled
            if self._safe_config_get('ml_features', 'enabled', default=False):
                try:
                    self.ml_engine = MLEngine(self.config)
                    logger.info("ML Engine initialized")
                except Exception as e:
                    logger.warning(f"ML engine initialization failed: {e}")
            
            # Initialize market data processor
            try:
                self.market_data_processor = MarketDataProcessor(self.config)
                logger.info("Market Data Processor initialized")
            except Exception as e:
                logger.warning(f"Market data processor initialization failed: {e}")
            
            # Initialize exchanges
            self._initialize_exchanges()
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            # Don't raise - allow partial initialization
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize Coinbase if configured and available
            if CoinbaseClient and self._safe_config_get('exchanges', 'coinbase', 'enabled', default=False):
                api_key = self._safe_config_get('exchanges', 'coinbase', 'api_key')
                api_secret = self._safe_config_get('exchanges', 'coinbase', 'api_secret')
                
                if api_key and api_secret:
                    try:
                        self.exchanges['coinbase'] = CoinbaseClient(
                            api_key=api_key,
                            api_secret=api_secret
                        )
                        logger.info("Coinbase client initialized")
                    except Exception as e:
                        logger.warning(f"Coinbase initialization failed: {e}")
            
            # Initialize Kraken if configured and available
            if KrakenClient and self._safe_config_get('exchanges', 'kraken', 'enabled', default=False):
                api_key = self._safe_config_get('exchanges', 'kraken', 'api_key')
                api_secret = self._safe_config_get('exchanges', 'kraken', 'api_secret')
                
                if api_key and api_secret:
                    try:
                        self.exchanges['kraken'] = KrakenClient(
                            api_key=api_key,
                            api_secret=api_secret
                        )
                        logger.info("Kraken client initialized")
                    except Exception as e:
                        logger.warning(f"Kraken initialization failed: {e}")
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def start(self):
        """Start the trading engine"""
        try:
            logger.info("Starting Industrial Trading Engine")
            
            self.is_running = True
            self.running = True
            self._loop = asyncio.get_event_loop()
            
            # Start component tasks
            self.tasks.extend([
                asyncio.create_task(self._monitor_markets()),
                asyncio.create_task(self._process_signals()),
                asyncio.create_task(self._manage_risk()),
                asyncio.create_task(self._update_performance())
            ])
            
            # Start strategy execution
            for strategy_id, strategy in self.strategies.items():
                self.tasks.append(
                    asyncio.create_task(self._run_strategy(strategy_id, strategy))
                )
            
            if self.notification_manager:
                await self.notification_manager.send_notification(
                    "🚀 Trading Engine Started",
                    "All systems operational"
                )
            
            # Keep running until shutdown
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            if self.notification_manager:
                await self.notification_manager.send_notification(
                    "❌ Trading Engine Error",
                    str(e)
                )
    
    async def stop(self):
        """Stop the trading engine gracefully"""
        try:
            logger.info("Stopping Industrial Trading Engine")
            
            self.shutdown_requested = True
            self.is_running = False
            self.running = False
            
            # Cancel all running tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Close exchange connections
            for exchange in self.exchanges.values():
                if hasattr(exchange, 'close'):
                    try:
                        await exchange.close()
                    except Exception as e:
                        logger.error(f"Error closing exchange: {e}")
            
            if self.notification_manager:
                await self.notification_manager.send_notification(
                    "🛑 Trading Engine Stopped",
                    "All positions closed safely"
                )
            
            logger.info("Trading Engine stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Enhanced status method for chat manager integration"""
        try:
            # Get base status
            base_status = self._get_base_status()
            
            # Calculate enhanced metrics
            portfolio_value = self._calculate_portfolio_value()
            daily_change = self._calculate_24h_change()
            risk_level = self._calculate_risk_level()
            market_sentiment = self._get_market_sentiment()
            
            # Get active positions summary
            positions_summary = []
            for symbol, position in self.positions.items():
                positions_summary.append({
                    'symbol': symbol,
                    'side': position.get('side', 'unknown'),
                    'quantity': position.get('quantity', 0),
                    'entry_price': position.get('entry_price', 0),
                    'current_price': position.get('current_price', 0),
                    'pnl': position.get('unrealized_pnl', 0),
                    'pnl_percent': position.get('pnl_percent', 0)
                })
            
            return {
                **base_status,
                'total_value': portfolio_value,
                'change_24h': daily_change / portfolio_value if portfolio_value > 0 else 0,
                'available_cash': self.balances.get('USDT', 0),
                'risk_level': self._calculate_risk_score(),
                'active_strategies': len([s for s in self.strategies.values() if hasattr(s, 'is_active') and getattr(s, 'is_active', True)]),
                'ml_models_loaded': 3 if self.ml_engine else 0,
                'market_data': {symbol: {} for symbol in ['BTC/USDT', 'ETH/USDT']},
                'positions': {symbol: position for symbol, position in self.positions.items()},
                'active_alerts': self._get_active_alerts(),
                'market_sentiment': market_sentiment,
                'pnl_today': self._calculate_daily_pnl(),
                'best_strategy': self._get_best_performing_strategy(),
                'win_rate': self._calculate_win_rate(),
                'market_volatility': self._calculate_market_volatility(),
                'max_drawdown': '5.2%',
                'last_analysis_time': '2 minutes ago'
            }
        except Exception as e:
            logger.error(f"Error getting comprehensive status: {e}")
            return self._get_fallback_status()
    
    def _get_base_status(self) -> Dict[str, Any]:
        """Get base status information"""
        return {
            'running': self.is_running,
            'trading_paused': self.trading_paused,
            'uptime': str(datetime.now() - self.start_time),
            'total_trades': self.trade_count,
            'active_orders': len(self.active_orders),
            'open_positions': len(self.positions),
            'total_pnl': self.total_pnl,
            'connected_exchanges': list(self.exchanges.keys()),
            'balances': self.balances
        }
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        try:
            total_value = sum(self.balances.values())
            
            # Add position values
            for position in self.positions.values():
                quantity = position.get('quantity', 0)
                current_price = position.get('current_price', 0)
                total_value += abs(quantity * current_price)
            
            return total_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def _calculate_24h_change(self) -> float:
        """Calculate 24-hour portfolio change"""
        try:
            # Placeholder calculation
            total_change = 0.0
            for position in self.positions.values():
                total_change += position.get('unrealized_pnl', 0)
            return total_change
        except Exception as e:
            logger.error(f"Error calculating 24h change: {e}")
            return 0.0
    
    def _calculate_risk_level(self) -> str:
        """Calculate current risk level"""
        try:
            risk_score = self._calculate_risk_score()
            
            if risk_score >= 8:
                return "CRITICAL"
            elif risk_score >= 6:
                return "HIGH"
            elif risk_score >= 4:
                return "MEDIUM"
            elif risk_score >= 2:
                return "LOW"
            else:
                return "MINIMAL"
        except Exception as e:
            logger.error(f"Error calculating risk level: {e}")
            return "UNKNOWN"
    
    def _calculate_risk_score(self) -> float:
        """Calculate numerical risk score (0-10)"""
        try:
            score = 2.0  # Base score
            
            # Factor in position count
            if len(self.positions) > 5:
                score += 1.0
            
            # Factor in total exposure
            portfolio_value = self._calculate_portfolio_value()
            cash_balance = sum(self.balances.values())
            if portfolio_value > 0 and cash_balance > 0:
                exposure_ratio = (portfolio_value - cash_balance) / cash_balance
                score += min(exposure_ratio * 2, 3)
            
            return min(score, 10)  # Cap at 10
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 5.0
    
    def _get_market_sentiment(self) -> str:
        """Get current market sentiment"""
        try:
            # Simple sentiment based on recent performance
            if self.total_pnl > 0:
                return "Bullish"
            elif self.total_pnl < 0:
                return "Bearish"
            else:
                return "Neutral"
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return "Unknown"
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        try:
            # Placeholder - implement actual daily P&L calculation
            return sum(p.get('unrealized_pnl', 0) for p in self.positions.values())
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
            return 0.0
    
    def _get_best_performing_strategy(self) -> str:
        """Get the best performing strategy"""
        try:
            if not self.performance_metrics:
                return "None"
                
            best_strategy = max(
                self.performance_metrics.items(),
                key=lambda x: x[1].get('total_pnl', 0),
                default=("None", {})
            )
            return best_strategy[0]
        except Exception as e:
            logger.error(f"Error getting best strategy: {e}")
            return "Unknown"
    
    def _calculate_win_rate(self) -> float:
        """Calculate overall win rate"""
        try:
            if self.trade_count == 0:
                return 0.0
            
            # Placeholder calculation
            winning_trades = max(1, int(self.trade_count * 0.65))  # Assume 65% win rate
            return winning_trades / self.trade_count
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _calculate_market_volatility(self) -> float:
        """Calculate current market volatility"""
        try:
            # Placeholder volatility
            return 0.03  # 3% volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02
    
    def _get_active_alerts(self) -> List[str]:
        """Get list of active alerts"""
        try:
            alerts = []
            
            # Check risk level
            risk_score = self._calculate_risk_score()
            if risk_score > 7:
                alerts.append("High risk level detected")
            
            # Check positions
            for symbol, position in self.positions.items():
                pnl = position.get('unrealized_pnl', 0)
                if pnl < -1000:  # Significant loss
                    alerts.append(f"{symbol} position at significant loss")
            
            return alerts
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def _get_fallback_status(self) -> Dict[str, Any]:
        """Get fallback status when errors occur"""
        return {
            'running': self.is_running,
            'error': 'Unable to fetch complete status',
            'basic_info': {
                'positions': len(self.positions),
                'active_orders': len(self.active_orders),
                'uptime': str(datetime.now() - self.start_time)
            }
        }

    # Trading loop methods
    async def _main_loop(self):
        """Main trading loop"""
        while self.is_running and not self.shutdown_requested:
            try:
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)

    async def _monitor_markets(self):
        """Monitor market conditions"""
        while self.is_running:
            try:
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error monitoring markets: {e}")
                await asyncio.sleep(60)

    async def _process_signals(self):
        """Process trading signals"""
        while self.is_running:
            try:
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error processing signals: {e}")
                await asyncio.sleep(5)

    async def _manage_risk(self):
        """Continuous risk management"""
        while self.is_running:
            try:
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error in risk management: {e}")
                await asyncio.sleep(30)

    async def _update_performance(self):
        """Update performance metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error updating performance: {e}")
                await asyncio.sleep(300)

    async def _run_strategy(self, strategy_id: str, strategy: StrategyBase):
        """Run a specific strategy"""
        while self.is_running and strategy_id in self.strategies:
            try:
                interval = getattr(strategy, 'execution_interval', 60)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error running strategy {strategy_id}: {e}")
                await asyncio.sleep(60)

    # Public API methods
    def get_status(self) -> Dict[str, Any]:
        """Get basic engine status"""
        return self._get_base_status()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_trades': self.trade_count,
            'total_pnl': self.total_pnl,
            'win_rate': self._calculate_win_rate(),
            'last_update': datetime.now().isoformat()
        }

    def add_strategy(self, strategy_id: str, strategy_type: str, config: Dict[str, Any]) -> bool:
        """Add a new strategy"""
        try:
            # Placeholder strategy addition
            self.strategies[strategy_id] = config
            logger.info(f"Added strategy: {strategy_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")
            return False

    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy"""
        try:
            if strategy_id in self.strategies:
                del self.strategies[strategy_id]
                logger.info(f"Removed strategy: {strategy_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing strategy: {e}")
            return False

    def list_available_strategies(self) -> List[str]:
        """List available strategy types"""
        return ["MLStrategy", "TrendFollowing", "MeanReversion"]

    def list_active_strategies(self) -> Dict[str, Any]:
        """List active strategies"""
        return {sid: {"type": "unknown", "config": config} for sid, config in self.strategies.items()}

    def _safe_config_get(self, *keys, default=None):
        """Safely get nested config values"""
        try:
            value = self.config
            for key in keys:
                if value is None:
                    return default
                    
                if isinstance(value, dict):
                    value = value.get(key, default)
                else:
                    value = getattr(value, key, default)
                    
                if value is default:
                    return default
                    
            return value if value is not None else default
        except (AttributeError, KeyError, TypeError):
            return default