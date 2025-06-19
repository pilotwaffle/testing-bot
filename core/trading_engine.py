# core/trading_engine.py
import asyncio
import logging
import os
import importlib.util
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime, timedelta
import pandas as pd

# Import other core components
from core.config import settings, EnhancedSettings, ConfigManager
from core.data_fetcher import CryptoDataFetcher
from core.notification_manager import SimpleNotificationManager
from ml.ml_engine import OctoBotMLEngine
from strategies.strategy_base import StrategyBase, EnhancedStrategyBase, SignalType, Signal

# Attempt to import Alpaca-py SDK
ALPACA_PY_AVAILABLE = False
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, LimitOrderRequest, OrderSide, TimeInForce
    from alpaca.trading.enums import OrderStatus
    ALPACA_PY_AVAILABLE = True
    logging.getLogger(__name__).info("Alpaca-py SDK available for trading.")
except ImportError:
    logging.getLogger(__name__).warning("Alpaca-py SDK not installed. Alpaca trading features will be disabled.")
except Exception as e:
    ALPACA_PY_AVAILABLE = False
    logging.getLogger(__name__).error(f"Error during Alpaca-py import: {e}. Alpaca trading features will be disabled.", exc_info=False)

# Try to import enhanced components (graceful fallback if not available)
try:
    from core.database import DatabaseManager, Trade
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logging.getLogger(__name__).warning("Database components not available. Trade persistence disabled.")

try:
    from core.risk_manager import RiskManager, RiskLevel
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False
    logging.getLogger(__name__).warning("Risk management components not available. Advanced risk features disabled.")

try:
    from core.backtesting_engine import BacktestingEngine, BacktestResult
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    logging.getLogger(__name__).warning("Backtesting engine not available. Backtesting features disabled.")

logger = logging.getLogger(__name__)

class IndustrialTradingEngine:
    """Enhanced Industrial Trading Engine with FreqTrade-style features"""
    
    def __init__(self, notification_manager_instance: SimpleNotificationManager, 
                 config: Optional[Union[EnhancedSettings, Dict[str, Any]]] = None):
        logger.info("Initializing Enhanced Industrial Trading Engine...")

        self.notification_manager = notification_manager_instance
        
        # Handle both old and new config formats
        if config is None:
            # Create default config
            config_manager = ConfigManager()
            self.config = config_manager._create_default_config()
        elif isinstance(config, dict):
            # Legacy config format - wrap in enhanced settings
            self.config = self._convert_legacy_config(config)
        else:
            # Already enhanced settings
            self.config = config
        
        # Core state
        self.running: bool = False
        self.alpaca_api: Optional[TradingClient] = None
        
        # Trading state (enhanced)
        initial_balance = getattr(self.config.trading, 'dry_run_wallet', 10000.0)
        self.balances: Dict[str, float] = {"USDT": 0.0, "USD": initial_balance}
        self.positions: Dict[str, Any] = {}
        self.active_strategies: Dict[str, StrategyBase] = {}
        
        # Core components
        self.data_fetcher = CryptoDataFetcher()
        self.current_market_data: Dict[str, Any] = {}
        
        # Enhanced components (optional)
        self.db_manager: Optional[DatabaseManager] = None
        self.risk_manager: Optional[RiskManager] = None
        
        if DATABASE_AVAILABLE:
            try:
                self.db_manager = DatabaseManager(self.config.db_url)
                logger.info("Database manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                self.db_manager = None
        
        if RISK_MANAGER_AVAILABLE:
            try:
                risk_config = self.config.trading.__dict__ if hasattr(self.config, 'trading') else {}
                self.risk_manager = RiskManager(risk_config)
                logger.info("Risk manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize risk manager: {e}")
                self.risk_manager = None

        # Initialize ML Engine
        self.ml_engine = OctoBotMLEngine()
        if hasattr(self.ml_engine, '_engine_reference'):
            self.ml_engine._engine_reference = self

        self.last_alpaca_sync_time = datetime.min
        
        # Load available strategy classes from files
        self.available_strategy_classes: Dict[str, Type[StrategyBase]] = self._load_strategy_classes()

        # Initialize Alpaca client if API keys are provided and SDK available
        self._initialize_alpaca()
        
        # Load existing trades from database
        if self.db_manager:
            self._load_open_trades()

        # Load ML models on startup
        self.ml_engine.load_models()
        logger.info(f"Loaded {len(self.ml_engine.models)} ML models on startup.")
        logger.info("Enhanced Industrial Trading Engine initialized.")

    def _convert_legacy_config(self, legacy_config: Dict[str, Any]) -> EnhancedSettings:
        """Convert legacy config dict to enhanced settings"""
        enhanced = EnhancedSettings()
        
        # Map common legacy fields
        if 'symbol' in legacy_config:
            enhanced.strategy.name = legacy_config.get('strategy_name', 'MLStrategy')
            
        if 'timeframe' in legacy_config:
            enhanced.strategy.timeframe = legacy_config['timeframe']
            
        if 'stake_amount' in legacy_config:
            enhanced.trading.stake_amount = legacy_config['stake_amount']
            
        return enhanced

    def _initialize_alpaca(self):
        """Initialize Alpaca client"""
        # Use enhanced config if available, fallback to original settings
        if hasattr(self.config, 'original') and self.config.original:
            alpaca_enabled = self.config.original.ALPACA_ENABLED
            api_key = self.config.original.ALPACA_API_KEY
            secret_key = self.config.original.ALPACA_SECRET_KEY
            base_url = self.config.original.ALPACA_BASE_URL
        else:
            # Try to get from exchange config
            alpaca_enabled = bool(getattr(self.config.exchange, 'key', None) and 
                                getattr(self.config.exchange, 'secret', None))
            api_key = getattr(self.config.exchange, 'key', None)
            secret_key = getattr(self.config.exchange, 'secret', None)
            base_url = 'https://paper-api.alpaca.markets'

        if alpaca_enabled and ALPACA_PY_AVAILABLE and api_key and secret_key:
            try:
                paper_mode = "paper" in base_url.lower()
                self.alpaca_api = TradingClient(
                    api_key=api_key,
                    secret_key=secret_key,
                    paper=paper_mode
                )
                logger.info(f"Alpaca TradingClient initialized (Paper Mode: {paper_mode}).")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca TradingClient: {e}. Alpaca trading disabled.", exc_info=True)
                self.alpaca_api = None
        else:
            logger.warning("Alpaca API keys not set or Alpaca-py not available. Trading operations will be simulated.")

    def _load_open_trades(self):
        """Load open trades from database"""
        if not self.db_manager:
            return
            
        try:
            open_trades = self.db_manager.get_open_trades()
            for trade in open_trades:
                self.positions[trade.pair] = {
                    'id': trade.id,
                    'amount': trade.amount,
                    'entry_price': trade.open_rate,
                    'open_date': trade.open_date,
                    'strategy': trade.strategy,
                    'is_short': getattr(trade, 'is_short', False)
                }
            logger.info(f"Loaded {len(open_trades)} open trades from database")
        except Exception as e:
            logger.error(f"Failed to load open trades: {e}")

    # Helper method to safely notify asynchronously
    async def _safe_notify(self, title: str, message: str, priority: str = "INFO"):
        try:
            await self.notification_manager.notify(title, message, priority)
        except Exception as e:
            logger.error(f"Failed to send notification: {title} - {message} ({e})", exc_info=False)

    # --- Market Data Callback ---
    async def on_market_data_update(self, market_data_batch: Dict[str, Any]):
        """Callback method to receive real-time market data updates."""
        for symbol, data in market_data_batch.items():
            self.current_market_data[symbol] = data
        logger.debug(f"Engine received real-time market data update for: {', '.join(market_data_batch.keys())}")

    async def start(self):
        """Starts the main trading engine loop and real-time data feed."""
        if self.running:
            logger.info("Trading engine is already running.")
            return

        self.running = True
        logger.info("Starting enhanced trading engine main loop...")
        
        await self._safe_notify(
            "Bot Startup", "Enhanced trading engine has started.", "HIGH"
        )

        # Start main trading loop
        asyncio.create_task(self._main_trading_loop())
        
        # Start real-time data feed
        logger.info("Starting real-time market data feed...")
        await self.data_fetcher.start_real_time_feed(self.on_market_data_update)

    async def stop(self):
        """Stops the main trading engine loop and data feed."""
        if not self.running:
            logger.info("Trading engine is not running.")
            return

        self.running = False
        logger.info("Stopping enhanced trading engine main loop...")
        await self.data_fetcher.stop_feed()

        await self._safe_notify(
            "Bot Shutdown", "Enhanced trading engine has stopped.", "HIGH"
        )

    # --- Enhanced Trading Operations ---
    async def place_order_enhanced(self, symbol: str, side: str, qty: float, 
                                  strategy_name: str = "manual", entry_tag: str = None,
                                  order_type: str = "market", limit_price: Optional[float] = None) -> Dict[str, Any]:
        """Enhanced order placement with database tracking and risk management"""
        
        # Risk management check (if available)
        if self.risk_manager and side.lower() == "buy":
            current_positions = {'open_trades': list(self.positions.values())}
            account_status = {'drawdown': self._calculate_current_drawdown()}
            
            signal_confidence = 0.8  # This should come from the strategy
            can_enter, reason = self.risk_manager.should_enter_trade(
                symbol, signal_confidence, current_positions, account_status
            )
            if not can_enter:
                logger.warning(f"Trade blocked by risk management: {reason}")
                await self._safe_notify("Trade Blocked", f"{symbol} trade blocked: {reason}", "WARNING")
                return {"status": "blocked", "reason": reason}

            # Calculate position size using risk management
            current_balance = self.balances.get("USDT", 0) + self.balances.get("USD", 0)
            current_price = self.current_market_data.get(symbol, {}).get("price", limit_price or 0)
            
            if current_price > 0:
                # Get stop loss from strategy if available
                stop_loss = None
                for strat in self.active_strategies.values():
                    if hasattr(strat, 'stoploss') and getattr(strat, 'symbol', None) == symbol:
                        stop_loss = current_price * (1 + strat.stoploss)  # stoploss is negative
                        break
                
                position_sizing = self.risk_manager.calculate_position_size(
                    symbol, current_price, stop_loss, current_balance, current_positions
                )
                
                # Use risk-managed quantity
                qty = position_sizing.recommended_amount
                logger.info(f"Risk-adjusted position size: {qty:.6f} {symbol} ({position_sizing.reasoning})")

        # Execute the order using existing logic
        order_result = await self.place_order(symbol, side, qty, order_type, limit_price)
        
        # Record in database if successful and database is available
        if order_result.get("status") == "filled" and self.db_manager:
            try:
                await self._record_trade_in_database(order_result, symbol, side, qty, strategy_name, entry_tag)
            except Exception as e:
                logger.error(f"Failed to record trade in database: {e}")
        
        return order_result

    async def _record_trade_in_database(self, order_result: Dict[str, Any], symbol: str, 
                                      side: str, qty: float, strategy_name: str, entry_tag: str):
        """Record trade in database"""
        if side.lower() == "buy":
            # Create new trade record
            trade_data = {
                'exchange': 'simulated' if not self.alpaca_api else 'alpaca',
                'pair': symbol,
                'base_currency': symbol.split('/')[0],
                'quote_currency': symbol.split('/')[1] if '/' in symbol else 'USD',
                'is_open': True,
                'fee_open': 0.001,
                'open_rate': order_result.get('price'),
                'stake_amount': qty * order_result.get('price'),
                'amount': qty,
                'open_date': datetime.now(),
                'strategy': strategy_name,
                'enter_tag': entry_tag,
                'timeframe': getattr(self.active_strategies.get(strategy_name), 'timeframe', None),
                'is_short': False,
                'trading_mode': getattr(self.config.trading, 'trading_mode', 'spot')
            }
            
            trade = self.db_manager.add_trade(trade_data)
            
            # Update local positions tracking
            self.positions[symbol] = {
                'id': trade.id,
                'amount': trade.amount,
                'entry_price': trade.open_rate,
                'open_date': trade.open_date,
                'strategy': strategy_name,
                'is_short': False
            }
            
        elif side.lower() == "sell":
            # Close existing trade
            if symbol in self.positions and 'id' in self.positions[symbol]:
                trade_id = self.positions[symbol]['id']
                close_updates = {
                    'is_open': False,
                    'close_rate': order_result.get('price'),
                    'close_date': datetime.now(),
                    'close_profit': self._calculate_trade_profit(symbol, order_result.get('price')),
                    'close_profit_abs': self._calculate_absolute_profit(symbol, order_result.get('price')),
                    'exit_reason': 'manual_exit'
                }
                
                self.db_manager.update_trade(trade_id, close_updates)
                
                # Update risk manager stats
                if self.risk_manager:
                    self.risk_manager.update_daily_stats({
                        'profit': close_updates['close_profit_abs']
                    })
                
                # Remove from local positions
                del self.positions[symbol]

    async def place_order(self, symbol: str, side: str, qty: float, order_type: str = "market", 
                         limit_price: Optional[float] = None) -> Dict[str, Any]:
        """Original place_order method (maintained for backward compatibility)"""
        logger.info(f"Attempting to place {side.upper()} {qty:.4f} of {symbol} at {order_type} type.")

        if self.alpaca_api:
            try:
                order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
                request_params = {
                    "symbol": symbol,
                    "qty": str(qty),
                    "side": order_side,
                    "time_in_force": TimeInForce.GTC
                }
                if order_type.lower() == "limit":
                    if limit_price is None:
                        raise ValueError("Limit price is required for limit orders.")
                    request_params["limit_price"] = str(limit_price)
                    order_request = LimitOrderRequest(**request_params)
                else:
                    order_request = MarketOrderRequest(**request_params)

                order = await self.alpaca_api.submit_order(order_request)
                logger.info(f"Alpaca order submitted: {order.id} - {order.status}")
                await self._safe_notify(
                    "Trade Executed (Alpaca)",
                    f"Placed {order.side.value.upper()} {order.qty} {order.symbol} @ {order_type} "
                    f"(status: {order.status.value}).",
                    "INFO"
                )
                await self._sync_alpaca_account()
                return order.dict()

            except Exception as e:
                logger.error(f"Failed to place Alpaca order for {symbol} ({side} {qty}): {e}", exc_info=True)
                await self._safe_notify(
                    "Trade Failed (Alpaca)", f"Order for {symbol} failed: {e}", "ERROR"
                )
                return {"status": "failed", "error": str(e)}
        else:
            # Simulate trade
            return await self._simulate_trade(symbol, side, qty)

    async def _simulate_trade(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        """Simulate trade execution"""
        current_price = self.current_market_data.get(symbol, {}).get("price")
        if not current_price:
            logger.warning(f"Cannot simulate trade for {symbol}, no current market data.")
            await self._safe_notify("Simulated Trade Failed", f"No market data for {symbol} to simulate trade.", "WARNING")
            return {"status": "failed", "error": f"No market data for {symbol}"}

        cost = qty * current_price
        
        if side.lower() == "buy":
            if self.balances.get("USDT", 0) >= cost:
                self.balances["USDT"] -= cost
                self.balances["USDT"] = round(self.balances["USDT"], 8)
                
                if symbol not in self.positions:
                    self.positions[symbol] = {"amount": 0.0, "entry_price": 0.0}
                    
                self.positions[symbol]["amount"] += qty
                self.positions[symbol]["amount"] = round(self.positions[symbol]["amount"], 8)
                self.positions[symbol]["entry_price"] = current_price
                
                logger.info(f"SIMULATED BUY: Bought {qty} {symbol} @ {current_price:.2f}. Balance: {self.balances['USDT']:.2f}")
                await self._safe_notify(
                    "Trade Executed (Simulated)",
                    f"Simulated BUY of {qty:.4f} {symbol} @ {current_price:.2f}.",
                    "INFO"
                )
                return {"status": "filled", "side": "buy", "symbol": symbol, "qty": qty, "price": current_price}
            else:
                logger.warning(f"SIMULATED BUY FAILED: Insufficient USDT balance ({self.balances.get('USDT',0):.2f}) to buy {qty} {symbol} at {current_price:.2f}.")
                await self._safe_notify(
                    "Trade Failed (Simulated)", f"Simulated BUY failed: Insufficient balance for {symbol}.", "WARNING"
                )
                return {"status": "rejected", "error": "Insufficient funds"}
                
        elif side.lower() == "sell":
            if self.positions.get(symbol, {}).get("amount", 0.0) >= qty:
                self.balances["USDT"] += cost
                self.balances["USDT"] = round(self.balances["USDT"], 8)
                self.positions[symbol]["amount"] -= qty
                self.positions[symbol]["amount"] = round(self.positions[symbol]["amount"], 8)
                
                if self.positions[symbol]["amount"] <= 0.0:
                    del self.positions[symbol]
                    
                logger.info(f"SIMULATED SELL: Sold {qty} {symbol} @ {current_price:.2f}. Balance: {self.balances['USDT']:.2f}")
                await self._safe_notify(
                    "Trade Executed (Simulated)",
                    f"Simulated SELL of {qty:.4f} {symbol} @ {current_price:.2f}.",
                    "INFO"
                )
                return {"status": "filled", "side": "sell", "symbol": symbol, "qty": qty, "price": current_price}
            else:
                logger.warning(f"SIMULATED SELL FAILED: No {symbol} position or insufficient quantity.")
                await self._safe_notify(
                    "Trade Failed (Simulated)", f"Simulated SELL failed: No position for {symbol}.", "WARNING"
                )
                return {"status": "rejected", "error": "No position"}
                
        return {"status": "failed", "error": "Unknown error in place_order"}

    def _calculate_trade_profit(self, symbol: str, close_price: float) -> float:
        """Calculate trade profit percentage"""
        if symbol not in self.positions:
            return 0.0
        entry_price = self.positions[symbol]['entry_price']
        return (close_price - entry_price) / entry_price

    def _calculate_absolute_profit(self, symbol: str, close_price: float) -> float:
        """Calculate absolute profit in base currency"""
        if symbol not in self.positions:
            return 0.0
        position = self.positions[symbol]
        entry_price = position['entry_price']
        amount = position['amount']
        return (close_price - entry_price) * amount

    def _calculate_current_drawdown(self) -> float:
        """Calculate current account drawdown"""
        current_value = sum(self.balances.values())
        initial_value = getattr(self.config.trading, 'dry_run_wallet', 10000.0)
        if initial_value > 0:
            return max(0, (initial_value - current_value) / initial_value)
        return 0.0

    # --- Alpaca Sync Operations ---
    async def _sync_alpaca_account(self):
        """Synchronizes account balances and open positions with Alpaca."""
        if not self.alpaca_api:
            return

        now = datetime.now()
        sync_interval = getattr(self.config.original, 'ALPACA_SYNC_INTERVAL_MINUTES', 5) if hasattr(self.config, 'original') else 5
        if now - self.last_alpaca_sync_time < timedelta(minutes=sync_interval):
            return

        logger.info("Synchronizing Alpaca account...")
        try:
            account = await self.alpaca_api.get_account()
            self.balances["USDT"] = float(account.cash)
            
            logger.info(f"Alpaca Balance (USD cash): {self.balances['USDT']:.2f}")

            alpaca_positions = await self.alpaca_api.get_all_positions()
            
            self.positions = {}
            for p in alpaca_positions:
                self.positions[p.symbol] = {
                    "amount": float(p.qty),
                    "entry_price": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "side": p.side
                }
            logger.info(f"Alpaca Positions: {len(self.positions)} open positions.")
            self.last_alpaca_sync_time = now

        except Exception as e:
            logger.error(f"Failed to sync Alpaca account: {e}", exc_info=True)
            await self._safe_notify(
                "Alpaca Sync Error", f"Failed to sync Alpaca account: {e}", "CRITICAL"
            )

    # --- Enhanced Strategy Management ---
    def _load_strategy_classes(self) -> Dict[str, Type[StrategyBase]]:
        """Dynamically loads strategy classes from files"""
        strategy_classes = {}
        strategies_dir = getattr(self.config.original, 'STRATEGIES_DIR', 'strategies') if hasattr(self.config, 'original') else 'strategies'
        
        if not os.path.isdir(strategies_dir):
            logger.error(f"Strategy directory '{strategies_dir}' not found.")
            return {}

        for filename in os.listdir(strategies_dir):
            if filename.endswith(".py") and filename != "__init__.py" and filename != "strategy_base.py":
                module_name = filename[:-3]
                file_path = os.path.join(strategies_dir, filename)
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None:
                        logger.warning(f"Could not load spec for module {module_name} from {file_path}")
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for attribute_name in dir(module):
                        attribute = getattr(module, attribute_name)
                        if (isinstance(attribute, type) and 
                            issubclass(attribute, (StrategyBase, EnhancedStrategyBase)) and 
                            attribute not in (StrategyBase, EnhancedStrategyBase)):
                            strategy_classes[attribute_name] = attribute
                            logger.info(f"Found and loaded strategy class: {attribute_name} from {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to load strategy from {filename}: {e}", exc_info=True)
        return strategy_classes

    def _instantiate_strategy(self, strategy_type: str, config: Dict[str, Any]) -> Optional[StrategyBase]:
        """Instantiates a strategy object from its type name."""
        strategy_class = self.available_strategy_classes.get(strategy_type)
        if not strategy_class:
            logger.error(f"Strategy type '{strategy_type}' not found.")
            return None
        
        try:
            # Check if it's an enhanced strategy
            if issubclass(strategy_class, EnhancedStrategyBase):
                strategy_instance = strategy_class(config, self.ml_engine, self.data_fetcher)
            elif strategy_type == "MLStrategy":
                strategy_instance = strategy_class(config, self.ml_engine, self.data_fetcher)
            else:
                strategy_instance = strategy_class(config)
            
            if not strategy_instance.validate_config():
                logger.error(f"Configuration for strategy {strategy_type} is invalid.")
                return None
            return strategy_instance
        except Exception as e:
            logger.error(f"Failed to instantiate strategy {strategy_type} with config {config}: {e}", exc_info=True)
            return None

    def add_strategy(self, strategy_id: str, strategy_type: str, config: Dict[str, Any]) -> bool:
        """Adds and instantiates an active strategy to the engine."""
        if strategy_id in self.active_strategies:
            logger.warning(f"Strategy ID '{strategy_id}' already active.")
            asyncio.create_task(self._safe_notify("Strategy Add Failed", f"Strategy ID '{strategy_id}' already active.", "WARNING"))
            return False

        strategy_instance = self._instantiate_strategy(strategy_type, config)
        if strategy_instance:
            self.active_strategies[strategy_id] = strategy_instance
            logger.info(f"Strategy '{strategy_id}' ({strategy_type}) added and activated.")
            
            # Log enhanced strategy details if available
            if hasattr(strategy_instance, 'timeframe'):
                logger.info(f"  - Timeframe: {strategy_instance.timeframe}")
            if hasattr(strategy_instance, 'stoploss'):
                logger.info(f"  - Stoploss: {strategy_instance.stoploss}")
            if hasattr(strategy_instance, 'minimal_roi'):
                logger.info(f"  - ROI: {strategy_instance.minimal_roi}")
            
            asyncio.create_task(self._safe_notify("Strategy Added", f"Strategy '{strategy_id}' ({strategy_type}) added.", "INFO"))
            return True
        return False

    def remove_strategy(self, strategy_id: str) -> bool:
        """Removes an active strategy from the engine."""
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            logger.info(f"Strategy '{strategy_id}' removed from active strategies.")
            asyncio.create_task(self._safe_notify("Strategy Removed", f"Strategy '{strategy_id}' removed.", "INFO"))
            return True
        logger.warning(f"Strategy '{strategy_id}' not found among active strategies.")
        return False

    def list_available_strategies(self) -> List[Dict[str, Any]]:
        """Returns metadata for all available strategy types."""
        available_list = []
        for name, cls in self.available_strategy_classes.items():
            strategy_info = {
                "name": name,
                "description": getattr(cls, 'DESCRIPTION', f"A {name} strategy."),
                "config_template": getattr(cls, 'CONFIG_TEMPLATE', {}),
                "is_enhanced": issubclass(cls, EnhancedStrategyBase),
                "supports_freqtrade": hasattr(cls, 'populate_entry_trend'),
            }
            
            # Add FreqTrade-style metadata if available
            if hasattr(cls, 'timeframe'):
                strategy_info['default_timeframe'] = cls.timeframe
            if hasattr(cls, 'stoploss'):
                strategy_info['default_stoploss'] = cls.stoploss
            if hasattr(cls, 'minimal_roi'):
                strategy_info['default_roi'] = cls.minimal_roi
                
            available_list.append(strategy_info)
        return available_list

    def list_active_strategies(self) -> List[Dict[str, Any]]:
        """Returns details of all currently active strategy instances."""
        active_list = []
        for strategy_id, strategy_obj in self.active_strategies.items():
            strategy_info = {
                "id": strategy_id,
                "type": strategy_obj.__class__.__name__,
                "symbol": getattr(strategy_obj, 'symbol', 'N/A'),
                "status": "Running" if self.running else "Inactive",
                "config_summary": strategy_obj.config,
                "is_enhanced": isinstance(strategy_obj, EnhancedStrategyBase),
            }
            
            # Add enhanced strategy details
            if hasattr(strategy_obj, 'timeframe'):
                strategy_info['timeframe'] = strategy_obj.timeframe
            if hasattr(strategy_obj, 'stoploss'):
                strategy_info['stoploss'] = strategy_obj.stoploss
            if hasattr(strategy_obj, 'minimal_roi'):
                strategy_info['roi'] = strategy_obj.minimal_roi
                
            active_list.append(strategy_info)
        return active_list

    # --- Backtesting Support ---
    def create_backtest_engine(self, strategy_name: str):
        """Create backtesting engine for a strategy"""
        if not BACKTESTING_AVAILABLE:
            raise ValueError("Backtesting engine not available. Install required dependencies.")
            
        if strategy_name not in self.active_strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        strategy = self.active_strategies[strategy_name]
        config = {
            'dry_run_wallet': getattr(self.config.trading, 'dry_run_wallet', 10000.0),
            'trading_fee': 0.001,
            'max_open_trades': getattr(self.config.trading, 'max_open_trades', 3),
            'stake_amount': getattr(self.config.trading, 'stake_amount', 100.0),
            'pair': getattr(strategy, 'symbol', 'BTC/USD')
        }
        
        from core.backtesting_engine import BacktestingEngine
        return BacktestingEngine(strategy, config)

    # --- Enhanced Main Trading Loop ---
    async def _main_trading_loop(self):
        """Enhanced core periodic loop for the trading engine."""
        broadcast_interval = getattr(self.config.original, 'BROADCAST_INTERVAL_SECONDS', 15) if hasattr(self.config, 'original') else 15
        error_cooldown = getattr(self.config.original, 'ERROR_RETRY_INTERVAL_SECONDS', 30) if hasattr(self.config, 'original') else 30

        while self.running:
            try:
                # 1. Sync account (periodically for real accounts)
                await self._sync_alpaca_account()

                # 2. Evaluate strategies
                for strategy_id, strategy in list(self.active_strategies.items()):
                    await self._evaluate_strategy(strategy_id, strategy)
            
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}", exc_info=True)
                await self._safe_notify(
                    "Trading Bot Error", f"Critical error in main loop: {e}", "CRITICAL"
                )
                await asyncio.sleep(error_cooldown)
                continue

            await asyncio.sleep(broadcast_interval)

    async def _evaluate_strategy(self, strategy_id: str, strategy: StrategyBase):
        """Evaluate a single strategy for trading signals"""
        symbol = getattr(strategy, 'symbol', strategy_id)
        latest_market_data = self.current_market_data.get(symbol)
        current_position = self.positions.get(symbol, {})

        if not latest_market_data:
            logger.debug(f"No current real-time market data for {symbol}. Skipping strategy evaluation for {strategy_id}.")
            return

        strategy._engine_reference = self

        try:
            # Use enhanced methods if available
            if isinstance(strategy, EnhancedStrategyBase):
                # Check for FreqTrade-style analysis
                if hasattr(strategy, 'analyze_ticker'):
                    # Prepare dataframe for analysis
                    df = pd.DataFrame([latest_market_data])
                    df.index = [pd.Timestamp.now()]
                    
                    metadata = {
                        'pair': symbol,
                        'timeframe': getattr(strategy, 'timeframe', '1h'),
                        'strategy': strategy.__class__.__name__
                    }
                    
                    # Analyze with FreqTrade-style interface
                    analyzed_df = strategy.analyze_ticker(df, metadata)
                    strategy._last_analyzed_dataframe = analyzed_df
                
                # Use enhanced signal methods
                sell_signal = await strategy.should_sell_enhanced(latest_market_data, current_position)
                buy_signal = await strategy.should_buy_enhanced(latest_market_data, current_position)
            else:
                # Use original methods
                sell_signal = await strategy.should_sell(latest_market_data, current_position)
                buy_signal = await strategy.should_buy(latest_market_data, current_position)

            # Process sell signal first
            if sell_signal.signal_type == SignalType.SELL:
                await self._process_sell_signal(strategy_id, symbol, sell_signal, current_position)
                return  # Don't buy if we just sold

            # Process buy signal
            if buy_signal.signal_type == SignalType.BUY:
                await self._process_buy_signal(strategy_id, symbol, buy_signal, current_position)

        except Exception as e:
            logger.error(f"Error evaluating strategy {strategy_id}: {e}", exc_info=True)

    async def _process_sell_signal(self, strategy_id: str, symbol: str, signal: Signal, current_position: Dict[str, Any]):
        """Process a sell signal"""
        logger.info(f"Strategy {strategy_id} generated SELL signal: {signal.reason} ({signal.confidence:.2f})")
        
        qty_to_trade = signal.quantity if signal.quantity and signal.quantity > 0 else current_position.get('amount', 0)
        if qty_to_trade > 0:
            order_result = await self.place_order_enhanced(
                symbol, "sell", qty_to_trade, strategy_id, signal.exit_tag
            )
            logger.info(f"Sell order result for {symbol}: {order_result.get('status')}")
        else:
            logger.info(f"Sell signal for {symbol} but quantity is 0 or no position to sell. Strategy: {strategy_id}")

    async def _process_buy_signal(self, strategy_id: str, symbol: str, signal: Signal, current_position: Dict[str, Any]):
        """Process a buy signal"""
        logger.info(f"Strategy {strategy_id} generated BUY signal: {signal.reason} ({signal.confidence:.2f})")
        
        if signal.quantity and signal.quantity > 0:
            order_result = await self.place_order_enhanced(
                symbol, "buy", signal.quantity, strategy_id, signal.entry_tag
            )
            logger.info(f"Buy order result for {symbol}: {order_result.get('status')}")
        else:
            logger.info(f"Buy signal for {symbol} but quantity is invalid ({signal.quantity}). Strategy: {strategy_id}")

    # --- Enhanced Status and Performance ---
    def get_status(self) -> Dict[str, Any]:
        """Provides a dictionary of the current bot status."""
        status = {
            "running": self.running,
            "alpaca_enabled": self.alpaca_api is not None,
            "balances": self.balances,
            "positions": self.positions,
            "active_strategies_count": len(self.active_strategies),
            "ml_engine_models_loaded": len(self.ml_engine.models),
            "latest_market_data": {s: {"price": d.get("price"), "timestamp": str(d.get("timestamp"))} 
                                   for s, d in self.current_market_data.items()},
            "ml_model_status": self.ml_engine.get_model_status(),
            "enhanced_features": {
                "database_available": DATABASE_AVAILABLE and self.db_manager is not None,
                "risk_management_available": RISK_MANAGER_AVAILABLE and self.risk_manager is not None,
                "backtesting_available": BACKTESTING_AVAILABLE,
            }
        }
        return status

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including enhanced features"""
        base_status = self.get_status()
        
        enhanced_status = {**base_status}
        
        # Add risk management status
        if self.risk_manager:
            enhanced_status['risk_management'] = self.risk_manager.get_risk_report()
        
        # Add database stats
        if self.db_manager:
            enhanced_status['database_stats'] = self.db_manager.get_trade_performance()
        
        # Add configuration info
        enhanced_status['configuration'] = {
            'dry_run': getattr(self.config.trading, 'dry_run', True),
            'max_open_trades': getattr(self.config.trading, 'max_open_trades', 3),
            'stake_amount': getattr(self.config.trading, 'stake_amount', 100.0),
            'trading_mode': getattr(self.config.trading, 'trading_mode', 'spot')
        }
        
        enhanced_status['strategies'] = {
            'active_count': len(self.active_strategies),
            'available_count': len(self.available_strategy_classes),
            'enhanced_strategies': len([s for s in self.active_strategies.values() if isinstance(s, EnhancedStrategyBase)])
        }
        
        return enhanced_status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculates and returns performance metrics."""
        total_value = self.balances.get("USDT", 0) + self.balances.get("USD", 0)
        for symbol, pos in self.positions.items():
            current_price = self.current_market_data.get(symbol, {}).get("price", pos.get("entry_price", 0))
            total_value += pos.get("amount", 0) * current_price
        
        metrics = {
            "total_account_value": round(total_value, 2),
            "timestamp": datetime.now().isoformat(),
            "open_positions": len(self.positions),
            "active_strategies": len(self.active_strategies)
        }
        
        # Add enhanced metrics if database available
        if self.db_manager:
            db_performance = self.db_manager.get_trade_performance()
            metrics.update(db_performance)
        
        return metrics

    def get_active_strategies_details(self) -> List[Dict[str, Any]]:
        """Returns details of all active strategies."""
        return self.list_active_strategies()

    # --- Market Data Helpers ---
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Returns the latest market data for a given symbol."""
        return self.current_market_data.get(symbol)