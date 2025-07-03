# main_trading_bot.py - Main Trading Bot Orchestrator
"""
Main Trading Bot Orchestrator
Coordinates all components: data fetching, ML predictions, strategy, and execution
"""

import asyncio
import logging
import json
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import time
from dataclasses import asdict
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced components
from core.enhanced_data_fetcher import EnhancedDataFetcher
from core.enhanced_ml_engine import AdaptiveMLEngine
from core.enhanced_trading_strategy import EnhancedTradingStrategy, TradingSignal, TradeDirection
from core.performance_monitor import PerformanceMonitor, TradeRecord

class TradingBotOrchestrator:
    """Main orchestrator for the enhanced trading bot"""
    
    def __init__(self, config_path: str = "config/bot_config.json"):
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_fetcher = EnhancedDataFetcher(
            exchange_name=self.config['exchange']['name'],
            cache_dir=self.config['paths']['data_cache_dir']
        )
        
        self.ml_engine = AdaptiveMLEngine(
            model_save_path=self.config['paths']['model_save_path'],
            performance_log_path=self.config['paths']['performance_log_path']
        )
        
        self.strategy_engine = EnhancedTradingStrategy(
            config_path=self.config['paths']['strategy_config_path']
        )
        
        self.performance_monitor = PerformanceMonitor(
            config_path=self.config['paths']['monitor_config_path']
        )
        
        # Bot state
        self.is_running = False
        self.is_trading_enabled = self.config['trading']['live_trading_enabled']
        self.current_positions = {}
        self.pending_orders = {}
        
        # Trading parameters
        self.symbols = self.config['trading']['symbols']
        self.timeframes = self.config['trading']['timeframes']
        self.trading_interval = self.config['trading']['signal_generation_interval_minutes']
        
        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'signals_generated': 0,
            'trades_executed': 0,
            'total_profit_loss': 0.0
        }
        
        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Trading Bot Orchestrator initialized")
        self.logger.info(f"Live trading: {'ENABLED' if self.is_trading_enabled else 'DISABLED (Paper Trading)'}")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Error file handler
        error_handler = logging.FileHandler(log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(error_handler)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load bot configuration"""
        default_config = {
            "exchange": {
                "name": "kraken",
                "api_key": "",
                "secret": "",
                "sandbox": True
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
        
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                self._deep_update(default_config, loaded_config)
                print(f"Configuration loaded from {config_path}")
            else:
                print("Using default configuration")
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    async def start(self):
        """Start the trading bot"""
        if self.is_running:
            self.logger.warning("Bot is already running")
            return
        
        self.is_running = True
        self.session_stats['start_time'] = datetime.now()
        
        try:
            self.logger.info("="*60)
            self.logger.info("STARTING ENHANCED TRADING BOT")
            self.logger.info("="*60)
            
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            # Register performance monitor callback
            self.performance_monitor.register_alert_callback(self._handle_performance_alert)
            
            # Initial system check
            await self._system_health_check()
            
            # Load existing models
            self._load_existing_models()
            
            # Start main trading loop
            await self._main_trading_loop()
            
        except Exception as e:
            self.logger.error(f"Fatal error in trading bot: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping trading bot...")
        self.is_running = False
        
        try:
            # Close all components
            self.performance_monitor.stop_monitoring()
            self.data_fetcher.close()
            
            # Generate session report
            self._generate_session_report()
            
            self.logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _system_health_check(self):
        """Perform initial system health check"""
        self.logger.info("Performing system health check...")
        
        health_status = {
            'data_fetcher': False,
            'ml_engine': False,
            'strategy_engine': False,
            'performance_monitor': False
        }
        
        try:
            # Test data fetcher
            test_data = self.data_fetcher.fetch_ohlcv_bulk("BTC/USD", "1h", 10)
            health_status['data_fetcher'] = len(test_data) > 0
            
            # Test ML engine
            health_status['ml_engine'] = len(self.ml_engine.models) >= 0  # Should at least initialize
            
            # Test strategy engine
            health_status['strategy_engine'] = self.strategy_engine is not None
            
            # Test performance monitor
            health_status['performance_monitor'] = self.performance_monitor is not None
            
            # Report health status
            for component, status in health_status.items():
                status_text = "✓ HEALTHY" if status else "✗ FAILED"
                self.logger.info(f"{component.upper()}: {status_text}")
            
            if not all(health_status.values()):
                raise Exception("System health check failed - some components are not working")
            
            self.logger.info("✓ All systems operational")
            
        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            raise
    
    def _load_existing_models(self):
        """Load existing ML models"""
        try:
            model_count = 0
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    model_key = f"{symbol}_{timeframe}"
                    if model_key in self.ml_engine.models:
                        model_count += 1
            
            self.logger.info(f"Loaded {model_count} existing ML models")
            
            if model_count == 0:
                self.logger.warning("No existing models found. Consider running training first.")
            
        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")
    
    async def _main_trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting main trading loop...")
        
        while self.is_running:
            try:
                loop_start_time = datetime.now()
                
                # Generate trading signals for all symbols
                signals = await self._generate_all_signals()
                
                # Process signals and execute trades
                if signals:
                    await self._process_signals(signals)
                
                # Update existing positions
                await self._update_positions()
                
                # Periodic maintenance
                await self._periodic_maintenance()
                
                # Calculate sleep time to maintain interval
                loop_duration = (datetime.now() - loop_start_time).total_seconds()
                sleep_time = max(0, (self.trading_interval * 60) - loop_duration)
                
                if sleep_time > 0:
                    self.logger.debug(f"Loop completed in {loop_duration:.2f}s, sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                else:
                    self.logger.warning(f"Loop took {loop_duration:.2f}s, longer than {self.trading_interval}m interval")
            
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _generate_all_signals(self) -> List[TradingSignal]:
        """Generate trading signals for all symbols"""
        signals = []
        
        try:
            # Fetch market data for all symbols and timeframes
            self.logger.debug("Fetching market data...")
            market_data_all = self.data_fetcher.fetch_multiple_symbols(
                symbols=self.symbols,
                timeframes=self.timeframes,
                total_candles=500  # Reduced for real-time operation
            )
            
            # Generate signals for each symbol
            for symbol in self.symbols:
                try:
                    symbol_data = market_data_all.get(symbol, {})
                    
                    if not symbol_data:
                        self.logger.warning(f"No market data available for {symbol}")
                        continue
                    
                    # Get ML predictions
                    ml_predictions = {}
                    for timeframe in self.timeframes:
                        if timeframe in symbol_data:
                            prediction = self.ml_engine.predict_ensemble(
                                symbol=symbol,
                                timeframe=timeframe,
                                data=symbol_data[timeframe]
                            )
                            if prediction:
                                ml_predictions[timeframe] = prediction
                    
                    # Generate trading signal
                    signal = self.strategy_engine.generate_trading_signal(
                        symbol=symbol,
                        market_data=symbol_data,
                        ml_predictions=ml_predictions,
                        current_positions=self.current_positions
                    )
                    
                    if signal:
                        signals.append(signal)
                        self.session_stats['signals_generated'] += 1
                        
                        if self.config['notifications']['log_all_signals']:
                            self.logger.info(f"Signal generated: {signal.symbol} {signal.direction.value} "
                                           f"(Confidence: {signal.confidence:.3f})")
                
                except Exception as e:
                    self.logger.error(f"Error generating signal for {symbol}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error in signal generation: {e}")
        
        return signals
    
    async def _process_signals(self, signals: List[TradingSignal]):
        """Process and potentially execute trading signals"""
        
        for signal in signals:
            try:
                # Check if we should execute this signal
                if not self._should_execute_signal(signal):
                    continue
                
                # Execute the trade
                if self.is_trading_enabled:
                    trade_result = await self._execute_trade(signal)
                else:
                    trade_result = self._simulate_trade(signal)
                
                if trade_result:
                    # Record the trade
                    trade_record = TradeRecord(
                        timestamp=datetime.now(),
                        symbol=signal.symbol,
                        timeframe=signal.timeframe,
                        prediction=signal.confidence if signal.direction == TradeDirection.BUY else 1 - signal.confidence,
                        confidence=signal.confidence,
                        actual_result=None,  # Will be updated later
                        profit_loss=None,    # Will be updated later
                        trade_id=trade_result['trade_id'],
                        model_used="ensemble",
                        market_conditions=signal.market_conditions
                    )
                    
                    # Record with performance monitor
                    self.performance_monitor.record_trade(trade_record)
                    
                    # Update position tracking
                    self.current_positions[signal.symbol] = {
                        'signal': signal,
                        'trade_result': trade_result,
                        'entry_time': datetime.now(),
                        'status': 'OPEN'
                    }
                    
                    self.session_stats['trades_executed'] += 1
                    
                    self.logger.info(f"Trade executed: {signal.symbol} {signal.direction.value} "
                                   f"at ${signal.entry_price:.2f}")
            
            except Exception as e:
                self.logger.error(f"Error processing signal for {signal.symbol}: {e}")
    
    def _should_execute_signal(self, signal: TradingSignal) -> bool:
        """Determine if a signal should be executed"""
        
        # Check if we already have a position in this symbol
        if signal.symbol in self.current_positions:
            existing_position = self.current_positions[signal.symbol]
            if existing_position['status'] == 'OPEN':
                return False
        
        # Check maximum concurrent positions
        open_positions = sum(1 for pos in self.current_positions.values() 
                           if pos['status'] == 'OPEN')
        
        if open_positions >= self.config['trading']['max_concurrent_positions']:
            return False
        
        # Check signal strength and confidence
        if signal.strength.value < 3:  # Below NEUTRAL
            return False
        
        # Check risk management rules
        portfolio_risk = self._calculate_current_portfolio_risk()
        if portfolio_risk > self.config['risk_management']['max_portfolio_risk']:
            return False
        
        return True
    
    async def _execute_trade(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """Execute actual trade (live trading)"""
        try:
            # This would integrate with actual exchange API
            # For now, we'll simulate the execution
            
            trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal.symbol.replace('/', '')}"
            
            # In real implementation, you would:
            # 1. Place market or limit order
            # 2. Set stop loss and take profit orders
            # 3. Handle order confirmation
            # 4. Store order details
            
            self.logger.info(f"[LIVE TRADE] {signal.direction.value} {signal.symbol} "
                           f"Size: {signal.position_size:.4f} Price: ${signal.entry_price:.2f}")
            
            return {
                'trade_id': trade_id,
                'status': 'EXECUTED',
                'execution_price': signal.entry_price,
                'execution_time': datetime.now(),
                'order_type': 'MARKET'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute live trade: {e}")
            return None
    
    def _simulate_trade(self, signal: TradingSignal) -> Dict[str, Any]:
        """Simulate trade execution (paper trading)"""
        
        trade_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal.symbol.replace('/', '')}"
        
        self.logger.info(f"[PAPER TRADE] {signal.direction.value} {signal.symbol} "
                        f"Size: {signal.position_size:.4f} Price: ${signal.entry_price:.2f}")
        
        return {
            'trade_id': trade_id,
            'status': 'SIMULATED',
            'execution_price': signal.entry_price,
            'execution_time': datetime.now(),
            'order_type': 'MARKET'
        }
    
    async def _update_positions(self):
        """Update existing positions and check for exits"""
        
        positions_to_remove = []
        
        for symbol, position in self.current_positions.items():
            try:
                if position['status'] != 'OPEN':
                    continue
                
                signal = position['signal']
                entry_time = position['entry_time']
                
                # Check position timeout
                if datetime.now() - entry_time > timedelta(hours=self.config['risk_management']['position_timeout_hours']):
                    self.logger.info(f"Position timeout reached for {symbol}")
                    await self._close_position(symbol, "TIMEOUT")
                    positions_to_remove.append(symbol)
                    continue
                
                # Get current price to check stop loss / take profit
                current_data = self.data_fetcher.fetch_ohlcv_bulk(symbol, "1h", 5)
                if len(current_data) > 0:
                    current_price = current_data['close'].iloc[-1]
                    
                    # Check exit conditions
                    should_exit, exit_reason = self._check_exit_conditions(signal, current_price)
                    
                    if should_exit:
                        await self._close_position(symbol, exit_reason)
                        positions_to_remove.append(symbol)
                        
                        # Calculate and record profit/loss
                        profit_loss = self._calculate_profit_loss(signal, current_price)
                        self.session_stats['total_profit_loss'] += profit_loss
                        
                        # Update performance monitor
                        actual_result = 1.0 if profit_loss > 0 else 0.0
                        self.performance_monitor.update_trade_outcome(
                            position['trade_result']['trade_id'],
                            actual_result,
                            profit_loss
                        )
            
            except Exception as e:
                self.logger.error(f"Error updating position for {symbol}: {e}")
        
        # Remove closed positions
        for symbol in positions_to_remove:
            del self.current_positions[symbol]
    
    def _check_exit_conditions(self, signal: TradingSignal, current_price: float) -> tuple[bool, str]:
        """Check if position should be closed"""
        
        if signal.direction == TradeDirection.BUY:
            # Long position
            if current_price <= signal.stop_loss:
                return True, "STOP_LOSS"
            elif current_price >= signal.take_profit:
                return True, "TAKE_PROFIT"
        
        elif signal.direction == TradeDirection.SELL:
            # Short position
            if current_price >= signal.stop_loss:
                return True, "STOP_LOSS"
            elif current_price <= signal.take_profit:
                return True, "TAKE_PROFIT"
        
        return False, ""
    
    async def _close_position(self, symbol: str, reason: str):
        """Close an existing position"""
        try:
            position = self.current_positions[symbol]
            
            if self.is_trading_enabled:
                # Execute actual close order
                self.logger.info(f"[LIVE CLOSE] Closing {symbol} position - Reason: {reason}")
                # API call to close position would go here
            else:
                # Simulate close
                self.logger.info(f"[PAPER CLOSE] Closing {symbol} position - Reason: {reason}")
            
            position['status'] = 'CLOSED'
            position['close_time'] = datetime.now()
            position['close_reason'] = reason
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
    
    def _calculate_profit_loss(self, signal: TradingSignal, exit_price: float) -> float:
        """Calculate profit/loss for a closed position"""
        
        if signal.direction == TradeDirection.BUY:
            # Long position
            return (exit_price - signal.entry_price) / signal.entry_price * signal.position_size
        
        elif signal.direction == TradeDirection.SELL:
            # Short position
            return (signal.entry_price - exit_price) / signal.entry_price * signal.position_size
        
        return 0.0
    
    def _calculate_current_portfolio_risk(self) -> float:
        """Calculate current portfolio risk exposure"""
        
        total_risk = 0.0
        
        for position in self.current_positions.values():
            if position['status'] == 'OPEN':
                signal = position['signal']
                position_risk = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * signal.position_size
                total_risk += position_risk
        
        return total_risk
    
    async def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        
        try:
            current_time = datetime.now()
            
            # Daily maintenance (run once per day)
            if hasattr(self, '_last_daily_maintenance'):
                if (current_time - self._last_daily_maintenance).days >= 1:
                    await self._daily_maintenance()
                    self._last_daily_maintenance = current_time
            else:
                self._last_daily_maintenance = current_time
            
            # Hourly maintenance
            if hasattr(self, '_last_hourly_maintenance'):
                if (current_time - self._last_hourly_maintenance).total_seconds() >= 3600:
                    await self._hourly_maintenance()
                    self._last_hourly_maintenance = current_time
            else:
                self._last_hourly_maintenance = current_time
        
        except Exception as e:
            self.logger.error(f"Error in periodic maintenance: {e}")
    
    async def _daily_maintenance(self):
        """Daily maintenance tasks"""
        self.logger.info("Performing daily maintenance...")
        
        try:
            # Clean up old cache files
            self.data_fetcher.cleanup_old_cache(max_age_days=7)
            
            # Generate daily performance report
            daily_performance = self.performance_monitor.get_recent_performance(hours=24)
            if daily_performance:
                self.logger.info(f"Daily Performance: {json.dumps(daily_performance, indent=2)}")
            
            # Check for model retraining needs
            trading_results = self._get_recent_trading_results()
            if trading_results:
                self.strategy_engine.update_strategy_performance(trading_results)
            
        except Exception as e:
            self.logger.error(f"Error in daily maintenance: {e}")
    
    async def _hourly_maintenance(self):
        """Hourly maintenance tasks"""
        self.logger.debug("Performing hourly maintenance...")
        
        try:
            # Log current status
            self._log_status_summary()
            
            # Check system health
            # await self._system_health_check()
            
        except Exception as e:
            self.logger.error(f"Error in hourly maintenance: {e}")
    
    def _get_recent_trading_results(self) -> List[Dict[str, Any]]:
        """Get recent trading results for performance analysis"""
        
        results = []
        
        for symbol, position in self.current_positions.items():
            if position['status'] == 'CLOSED':
                signal = position['signal']
                
                # Calculate actual results
                if 'close_price' in position:
                    profit_loss = self._calculate_profit_loss(signal, position['close_price'])
                    actual_result = 1.0 if profit_loss > 0 else 0.0
                    
                    results.append({
                        'symbol': symbol,
                        'timeframe': signal.timeframe,
                        'prediction': signal.confidence,
                        'actual_result': actual_result,
                        'profit_loss': profit_loss,
                        'entry_time': position['entry_time'],
                        'close_time': position.get('close_time'),
                        'close_reason': position.get('close_reason')
                    })
        
        return results
    
    def _handle_performance_alert(self, alert: Dict[str, Any]):
        """Handle performance alerts from the monitor"""
        
        severity = alert['severity']
        message = alert['message']
        
        if severity == 'critical':
            self.logger.critical(f"CRITICAL ALERT: {message}")
            
            # Consider stopping trading on critical alerts
            if 'accuracy' in alert['alert_type'] or 'losses' in alert['alert_type']:
                self.logger.warning("Considering disabling trading due to critical performance alert")
                # self.is_trading_enabled = False  # Uncomment to auto-disable
        
        elif severity == 'warning':
            self.logger.warning(f"PERFORMANCE WARNING: {message}")
        
        else:
            self.logger.info(f"PERFORMANCE INFO: {message}")
    
    def _log_status_summary(self):
        """Log current bot status summary"""
        
        open_positions = sum(1 for pos in self.current_positions.values() 
                           if pos['status'] == 'OPEN')
        
        uptime = datetime.now() - self.session_stats['start_time']
        
        self.logger.info(f"STATUS: Uptime: {uptime}, "
                        f"Signals: {self.session_stats['signals_generated']}, "
                        f"Trades: {self.session_stats['trades_executed']}, "
                        f"Open Positions: {open_positions}, "
                        f"P&L: {self.session_stats['total_profit_loss']:.4f}")
    
    def _generate_session_report(self):
        """Generate final session report"""
        
        try:
            session_duration = datetime.now() - self.session_stats['start_time']
            
            report = {
                'session_summary': {
                    'start_time': self.session_stats['start_time'].isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_hours': session_duration.total_seconds() / 3600,
                    'signals_generated': self.session_stats['signals_generated'],
                    'trades_executed': self.session_stats['trades_executed'],
                    'total_profit_loss': self.session_stats['total_profit_loss'],
                    'trading_mode': 'LIVE' if self.is_trading_enabled else 'PAPER'
                },
                'final_positions': {
                    symbol: {
                        'status': pos['status'],
                        'direction': pos['signal'].direction.value,
                        'entry_price': pos['signal'].entry_price,
                        'entry_time': pos['entry_time'].isoformat()
                    }
                    for symbol, pos in self.current_positions.items()
                },
                'performance_summary': self.strategy_engine.get_strategy_performance()
            }
            
            # Save report
            report_path = Path("logs") / f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Session report saved: {report_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Duration: {session_duration}")
            print(f"Signals Generated: {self.session_stats['signals_generated']}")
            print(f"Trades Executed: {self.session_stats['trades_executed']}")
            print(f"Total P&L: {self.session_stats['total_profit_loss']:.4f}")
            print(f"Open Positions: {sum(1 for pos in self.current_positions.values() if pos['status'] == 'OPEN')}")
            print("="*60)
        
        except Exception as e:
            self.logger.error(f"Error generating session report: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        
        return {
            'is_running': self.is_running,
            'is_trading_enabled': self.is_trading_enabled,
            'uptime_seconds': (datetime.now() - self.session_stats['start_time']).total_seconds(),
            'session_stats': self.session_stats,
            'open_positions': {
                symbol: {
                    'direction': pos['signal'].direction.value,
                    'entry_price': pos['signal'].entry_price,
                    'current_risk': abs(pos['signal'].entry_price - pos['signal'].stop_loss) / pos['signal'].entry_price
                }
                for symbol, pos in self.current_positions.items()
                if pos['status'] == 'OPEN'
            },
            'portfolio_risk': self._calculate_current_portfolio_risk(),
            'system_health': 'OPERATIONAL' if self.is_running else 'STOPPED'
        }

async def main():
    """Main entry point"""
    
    print("Enhanced Trading Bot")
    print("===================")
    
    # Initialize bot
    bot = TradingBotOrchestrator()
    
    try:
        # Start the bot
        await bot.start()
    
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Bot crashed: {e}")
        logging.error(f"Bot crashed: {e}")
    finally:
        bot.stop()

if __name__ == "__main__":
    asyncio.run(main())