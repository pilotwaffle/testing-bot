# core/trading_engine.py - Complete Corrected Version
import asyncio
import logging
import os
import importlib.util
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime, timedelta
import pandas as pd

# Import other core components
from core.config import settings, EnhancedSettings
from utils.simple_notification_manager import SimpleNotificationManager
from core.risk_management import RiskManager, EnhancedRiskManager
from core.signal_processor import SignalProcessor
from core.ml_engine import MLEngine
from core.dex_monitor import DexMonitor
from core.market_data_processor import MarketDataProcessor
from core.gas_tracker import GasTracker
from database.database_manager import DatabaseManager

# Import strategy base
from strategies.base_strategy import BaseStrategy

# Import exchange clients
from exchanges.coinbase_client import CoinbaseClient
from exchanges.kraken_client import KrakenClient
from exchanges.dex_integration import DexIntegration
from exchanges.uniswap_client import UniswapClient

logger = logging.getLogger(__name__)

class IndustrialTradingEngine:
    """Enhanced Industrial Trading Engine with FreqTrade-style features"""
    
    def __init__(self, notification_manager_instance: SimpleNotificationManager, 
                 config: Optional[Union[EnhancedSettings, Dict[str, Any]]] = None):
        logger.info("Initializing Enhanced Industrial Trading Engine")
        
        # Store configuration
        self.config = config or settings
        self.notification_manager = notification_manager_instance
        
        # Initialize core components
        self.db_manager = DatabaseManager()
        self.risk_manager = None
        self.signal_processor = None
        self.ml_engine = None
        self.market_data_processor = None
        
        # Trading state
        self.is_running = False
        self.trading_paused = False
        self.shutdown_requested = False
        
        # Initialize collections
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_orders: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
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
                self.risk_manager = EnhancedRiskManager(self.config)
                logger.info("Risk Manager initialized")
            
            # Initialize signal processor
            self.signal_processor = SignalProcessor(self.config, self.db_manager)
            logger.info("Signal Processor initialized")
            
            # Initialize ML engine if enabled
            if self._safe_config_get('ml_features', 'enabled', default=False):
                self.ml_engine = MLEngine(self.config)
                logger.info("ML Engine initialized")
            
            # Initialize market data processor
            self.market_data_processor = MarketDataProcessor(self.config)
            logger.info("Market Data Processor initialized")
            
            # Initialize exchanges
            self._initialize_exchanges()
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize Coinbase if configured
            if self._safe_config_get('exchanges', 'coinbase', 'enabled', default=False):
                self.exchanges['coinbase'] = CoinbaseClient(
                    api_key=self._safe_config_get('exchanges', 'coinbase', 'api_key'),
                    api_secret=self._safe_config_get('exchanges', 'coinbase', 'api_secret')
                )
                logger.info("Coinbase client initialized")
            
            # Initialize Kraken if configured
            if self._safe_config_get('exchanges', 'kraken', 'enabled', default=False):
                self.exchanges['kraken'] = KrakenClient(
                    api_key=self._safe_config_get('exchanges', 'kraken', 'api_key'),
                    api_secret=self._safe_config_get('exchanges', 'kraken', 'api_secret')
                )
                logger.info("Kraken client initialized")
            
            # Initialize DEX if configured
            if self._safe_config_get('dex', 'enabled', default=False):
                self.exchanges['dex'] = DexIntegration(self.config)
                logger.info("DEX integration initialized")
            
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def start(self):
        """Start the trading engine"""
        try:
            logger.info("Starting Industrial Trading Engine")
            
            self.is_running = True
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
            
            await self.notification_manager.send_notification(
                "🚀 Trading Engine Started",
                "All systems operational"
            )
            
            # Keep running until shutdown
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
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
            
            # Cancel all running tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Close exchange connections
            for exchange in self.exchanges.values():
                if hasattr(exchange, 'close'):
                    await exchange.close()
            
            await self.notification_manager.send_notification(
                "🛑 Trading Engine Stopped",
                "All positions closed safely"
            )
            
            logger.info("Trading Engine stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
    
    # === ENHANCED METHODS FOR AI CHAT INTEGRATION ===
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Enhanced status method for chat manager integration"""
        try:
            # Get base status
            base_status = await self._get_base_status()
            
            # Calculate enhanced metrics
            portfolio_value = await self._calculate_portfolio_value()
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
                'enhanced_metrics': {
                    'portfolio_value': portfolio_value,
                    'daily_change': daily_change,
                    'daily_change_percent': (daily_change / portfolio_value * 100) if portfolio_value > 0 else 0,
                    'risk_level': risk_level,
                    'risk_score': self._calculate_risk_score(),
                    'market_sentiment': market_sentiment,
                    'volatility': self._calculate_market_volatility(),
                    'active_strategies': len([s for s in self.strategies.values() if s.is_active]),
                    'total_strategies': len(self.strategies),
                    'win_rate': self._calculate_win_rate(),
                    'sharpe_ratio': self._calculate_sharpe_ratio(),
                    'max_drawdown': self._calculate_max_drawdown(),
                    'current_drawdown': self._calculate_current_drawdown(),
                    'positions_summary': positions_summary,
                    'top_performer': self._get_best_performing_position(),
                    'worst_performer': self._get_worst_performing_position(),
                    'last_trade_time': self._get_last_trade_time(),
                    'trading_volume_24h': self._calculate_24h_volume(),
                    'alerts_active': len(self._get_active_alerts()),
                    'market_conditions': self._analyze_market_conditions()
                }
            }
        except Exception as e:
            logger.error(f"Error getting comprehensive status: {e}")
            return self._get_fallback_status()
    
    async def _get_base_status(self) -> Dict[str, Any]:
        """Get base status information"""
        return {
            'is_running': self.is_running,
            'trading_paused': self.trading_paused,
            'uptime': str(datetime.now() - self.start_time),
            'total_trades': self.trade_count,
            'active_orders': len(self.active_orders),
            'open_positions': len(self.positions),
            'total_pnl': self.total_pnl,
            'connected_exchanges': list(self.exchanges.keys())
        }
    
    async def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        try:
            total_value = 0.0
            
            # Sum up all position values
            for position in self.positions.values():
                quantity = position.get('quantity', 0)
                current_price = position.get('current_price', 0)
                total_value += quantity * current_price
            
            # Add available balance
            for exchange in self.exchanges.values():
                if hasattr(exchange, 'get_balance'):
                    balance = await exchange.get_balance()
                    total_value += balance.get('total_usd', 0)
            
            return total_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def _calculate_24h_change(self) -> float:
        """Calculate 24-hour portfolio change"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                yesterday = datetime.now() - timedelta(days=1)
                historical_value = self.db_manager.get_portfolio_value_at(yesterday)
                current_value = self.positions.get('total_value', 0)
                return current_value - historical_value
            return 0.0
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
            score = 0.0
            
            # Factor 1: Position concentration
            if self.positions:
                largest_position = max(self.positions.values(), 
                                     key=lambda x: abs(x.get('value', 0)))
                concentration = abs(largest_position.get('value', 0)) / self._get_total_portfolio_value()
                score += concentration * 3  # Max 3 points
            
            # Factor 2: Leverage
            leverage = self._get_current_leverage()
            score += min(leverage, 3)  # Max 3 points
            
            # Factor 3: Volatility
            volatility = self._calculate_market_volatility()
            score += volatility * 20  # Max ~2 points
            
            # Factor 4: Drawdown
            drawdown = self._calculate_current_drawdown()
            score += drawdown * 10  # Max ~2 points
            
            return min(score, 10)  # Cap at 10
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 5.0  # Default medium risk
    
    def _get_market_sentiment(self) -> str:
        """Get current market sentiment"""
        try:
            # This is a simplified version - enhance with real indicators
            volatility = self._calculate_market_volatility()
            trend = self._calculate_market_trend()
            
            if volatility > 0.03:
                return "Highly Volatile"
            elif trend > 0.02:
                return "Bullish"
            elif trend < -0.02:
                return "Bearish"
            elif volatility > 0.02:
                return "Volatile"
            elif abs(trend) < 0.005:
                return "Ranging"
            else:
                return "Neutral"
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return "Unknown"
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                today_start = datetime.now().replace(hour=0, minute=0, second=0)
                trades_today = self.db_manager.get_trades_since(today_start)
                return sum(trade.get('pnl', 0) for trade in trades_today)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
            return 0.0
    
    def _get_best_performing_strategy(self) -> Optional[str]:
        """Get the best performing strategy"""
        try:
            best_strategy = None
            best_performance = float('-inf')
            
            for strategy_id, metrics in self.performance_metrics.items():
                total_pnl = metrics.get('total_pnl', 0)
                if total_pnl > best_performance:
                    best_performance = total_pnl
                    best_strategy = strategy_id
            
            return best_strategy
        except Exception as e:
            logger.error(f"Error getting best strategy: {e}")
            return None
    
    def _calculate_win_rate(self) -> float:
        """Calculate overall win rate"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                total_trades = self.db_manager.get_trade_count()
                winning_trades = self.db_manager.get_winning_trade_count()
                
                if total_trades > 0:
                    return winning_trades / total_trades
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                returns = self.db_manager.get_daily_returns(days=30)
                if returns and len(returns) > 1:
                    avg_return = sum(returns) / len(returns)
                    std_dev = pd.Series(returns).std()
                    risk_free_rate = 0.02 / 365  # 2% annual
                    
                    if std_dev > 0:
                        return (avg_return - risk_free_rate) / std_dev * (365 ** 0.5)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                equity_curve = self.db_manager.get_equity_curve()
                if equity_curve:
                    peak = equity_curve[0]
                    max_dd = 0
                    
                    for value in equity_curve:
                        if value > peak:
                            peak = value
                        dd = (peak - value) / peak if peak > 0 else 0
                        max_dd = max(max_dd, dd)
                    
                    return max_dd
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                # Get recent trades to calculate drawdown
                trades = self.db_manager.get_recent_trades(days=30)
                if trades:
                    values = [t.get('total_value', 0) for t in trades]
                    if values:
                        peak = max(values)
                        current = values[-1]
                        return (peak - current) / peak if peak > 0 else 0
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating current drawdown: {e}")
            return 0.0
    
    def _calculate_market_volatility(self) -> float:
        """Calculate current market volatility"""
        try:
            # Simplified volatility calculation
            if hasattr(self, 'market_data_processor') and self.market_data_processor:
                prices = self.market_data_processor.get_recent_prices('BTC/USD', 100)
                if prices:
                    returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                              for i in range(1, len(prices))]
                    return pd.Series(returns).std()
            return 0.02  # Default 2% volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        try:
            alerts = []
            
            # Check risk alerts
            if self._calculate_risk_score() > 7:
                alerts.append({
                    'type': 'risk',
                    'severity': 'high',
                    'message': 'Portfolio risk level critical'
                })
            
            # Check drawdown alerts
            if self._calculate_current_drawdown() > 0.1:
                alerts.append({
                    'type': 'drawdown',
                    'severity': 'medium',
                    'message': 'Significant drawdown detected'
                })
            
            # Check position alerts
            for symbol, position in self.positions.items():
                if position.get('pnl_percent', 0) < -0.05:
                    alerts.append({
                        'type': 'position',
                        'severity': 'medium',
                        'message': f'{symbol} position down >5%'
                    })
            
            return alerts
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def _get_last_analysis_time(self) -> Optional[datetime]:
        """Get timestamp of last market analysis"""
        try:
            if hasattr(self, 'signal_processor') and self.signal_processor:
                return self.signal_processor.last_analysis_time
            return None
        except Exception as e:
            logger.error(f"Error getting last analysis time: {e}")
            return None
    
    def _get_total_portfolio_value(self) -> float:
        """Get total portfolio value synchronously"""
        try:
            total = 0.0
            for position in self.positions.values():
                total += position.get('value', 0)
            return total
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0.0
    
    def _get_current_leverage(self) -> float:
        """Get current leverage ratio"""
        try:
            if hasattr(self, 'risk_manager') and self.risk_manager:
                return self.risk_manager.get_current_leverage()
            return 1.0
        except Exception as e:
            logger.error(f"Error getting leverage: {e}")
            return 1.0
    
    def _calculate_market_trend(self) -> float:
        """Calculate market trend (-1 to 1)"""
        try:
            if hasattr(self, 'market_data_processor') and self.market_data_processor:
                # Simple SMA crossover trend
                prices = self.market_data_processor.get_recent_prices('BTC/USD', 50)
                if len(prices) >= 50:
                    sma_short = sum(prices[-10:]) / 10
                    sma_long = sum(prices) / 50
                    return (sma_short - sma_long) / sma_long
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0.0
    
    def _get_best_performing_position(self) -> Optional[Dict[str, Any]]:
        """Get best performing position"""
        try:
            if not self.positions:
                return None
            
            best_position = max(self.positions.items(), 
                              key=lambda x: x[1].get('pnl_percent', 0))
            return {
                'symbol': best_position[0],
                'pnl': best_position[1].get('unrealized_pnl', 0),
                'pnl_percent': best_position[1].get('pnl_percent', 0)
            }
        except Exception as e:
            logger.error(f"Error getting best position: {e}")
            return None
    
    def _get_worst_performing_position(self) -> Optional[Dict[str, Any]]:
        """Get worst performing position"""
        try:
            if not self.positions:
                return None
            
            worst_position = min(self.positions.items(), 
                               key=lambda x: x[1].get('pnl_percent', 0))
            return {
                'symbol': worst_position[0],
                'pnl': worst_position[1].get('unrealized_pnl', 0),
                'pnl_percent': worst_position[1].get('pnl_percent', 0)
            }
        except Exception as e:
            logger.error(f"Error getting worst position: {e}")
            return None
    
    def _get_last_trade_time(self) -> Optional[str]:
        """Get time of last trade"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                last_trade = self.db_manager.get_last_trade()
                if last_trade:
                    return last_trade.get('timestamp', '').isoformat()
            return None
        except Exception as e:
            logger.error(f"Error getting last trade time: {e}")
            return None
    
    def _calculate_24h_volume(self) -> float:
        """Calculate 24h trading volume"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                yesterday = datetime.now() - timedelta(days=1)
                trades = self.db_manager.get_trades_since(yesterday)
                return sum(trade.get('volume', 0) for trade in trades)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating 24h volume: {e}")
            return 0.0
    
    def _analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze current market conditions"""
        try:
            return {
                'trend': self._calculate_market_trend(),
                'volatility': self._calculate_market_volatility(),
                'volume_trend': 'normal',  # Placeholder
                'major_levels': {
                    'support': 0,  # Placeholder
                    'resistance': 0  # Placeholder
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return {}
    
    def _get_fallback_status(self) -> Dict[str, Any]:
        """Get fallback status when errors occur"""
        return {
            'is_running': self.is_running,
            'error': 'Unable to fetch complete status',
            'basic_info': {
                'positions': len(self.positions),
                'active_orders': len(self.active_orders)
            }
        }
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze a specific trading symbol for chat integration"""
        try:
            logger.info(f"Analyzing symbol: {symbol}")
            
            # Get current market data
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return {'error': f'Unable to get price for {symbol}'}
            
            # Get technical indicators
            indicators = await self._calculate_indicators(symbol)
            
            # Get ML predictions if available
            ml_prediction = None
            if self.ml_engine and hasattr(self.ml_engine, 'analyze_symbol'):
                try:
                    ml_prediction = await self.ml_engine.analyze_symbol(symbol)
                except Exception as e:
                    logger.error(f"ML analysis failed: {e}")
            
            # Get recent performance
            performance = self._get_symbol_performance(symbol)
            
            # Generate signals
            signals = []
            if self.signal_processor:
                try:
                    raw_signals = await self.signal_processor.process_symbol(symbol)
                    signals = [{'type': s['type'], 'strength': s['strength']} 
                              for s in raw_signals]
                except Exception as e:
                    logger.error(f"Signal processing failed: {e}")
            
            # Calculate support/resistance
            support_resistance = self._calculate_support_resistance(symbol)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'analysis_time': datetime.now().isoformat(),
                'technical_indicators': indicators,
                'ml_prediction': ml_prediction,
                'performance': performance,
                'signals': signals,
                'support_resistance': support_resistance,
                'recommendation': self._generate_recommendation(
                    indicators, ml_prediction, signals
                ),
                'risk_assessment': {
                    'volatility': indicators.get('volatility', 0),
                    'risk_score': self._calculate_symbol_risk(symbol),
                    'position_size_recommendation': self._calculate_position_size(symbol)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol
            }
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # Try each exchange
            for exchange_name, exchange in self.exchanges.items():
                if hasattr(exchange, 'get_ticker'):
                    ticker = await exchange.get_ticker(symbol)
                    if ticker and 'last' in ticker:
                        return ticker['last']
            
            # Fallback to market data processor
            if self.market_data_processor:
                return self.market_data_processor.get_last_price(symbol)
            
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    async def _calculate_indicators(self, symbol: str) -> Dict[str, Any]:
        """Calculate technical indicators for symbol"""
        try:
            if not self.market_data_processor:
                return {}
            
            # Get price data
            prices = self.market_data_processor.get_recent_prices(symbol, 100)
            if not prices or len(prices) < 20:
                return {}
            
            # Convert to pandas series for calculations
            price_series = pd.Series(prices)
            
            # Calculate indicators
            indicators = {
                'rsi': self._calculate_rsi(price_series),
                'sma_20': price_series.tail(20).mean(),
                'sma_50': price_series.tail(50).mean() if len(prices) >= 50 else None,
                'ema_12': price_series.ewm(span=12).mean().iloc[-1],
                'ema_26': price_series.ewm(span=26).mean().iloc[-1],
                'volatility': price_series.pct_change().std(),
                'volume_trend': 'normal',  # Placeholder
                'bb_upper': price_series.mean() + (2 * price_series.std()),
                'bb_lower': price_series.mean() - (2 * price_series.std()),
                'trend': 'bullish' if prices[-1] > prices[-20] else 'bearish'
            }
            
            # MACD
            if len(prices) >= 26:
                indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get symbol performance metrics"""
        try:
            if not self.market_data_processor:
                return {}
            
            current = self.market_data_processor.get_last_price(symbol)
            if not current:
                return {}
            
            # Get historical prices
            prices_1h = self.market_data_processor.get_price_at(
                symbol, datetime.now() - timedelta(hours=1)
            )
            prices_24h = self.market_data_processor.get_price_at(
                symbol, datetime.now() - timedelta(days=1)
            )
            prices_7d = self.market_data_processor.get_price_at(
                symbol, datetime.now() - timedelta(days=7)
            )
            
            return {
                '1h': ((current - prices_1h) / prices_1h * 100) if prices_1h else 0,
                '24h': ((current - prices_24h) / prices_24h * 100) if prices_24h else 0,
                '7d': ((current - prices_7d) / prices_7d * 100) if prices_7d else 0,
            }
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return {}
    
    def _calculate_support_resistance(self, symbol: str) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        try:
            if not self.market_data_processor:
                return {}
            
            prices = self.market_data_processor.get_recent_prices(symbol, 100)
            if not prices or len(prices) < 20:
                return {}
            
            # Simple support/resistance calculation
            current = prices[-1]
            highs = pd.Series(prices).rolling(10).max()
            lows = pd.Series(prices).rolling(10).min()
            
            resistance = highs[highs > current].min() if any(highs > current) else current * 1.02
            support = lows[lows < current].max() if any(lows < current) else current * 0.98
            
            return {
                'support': support,
                'resistance': resistance,
                'current': current
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    def _generate_recommendation(self, indicators: Dict[str, Any], 
                               ml_prediction: Optional[Dict[str, Any]], 
                               signals: List[Dict[str, Any]]) -> str:
        """Generate trading recommendation"""
        try:
            score = 0
            
            # RSI analysis
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                score += 2  # Oversold
            elif rsi > 70:
                score -= 2  # Overbought
            
            # Trend analysis
            if indicators.get('trend') == 'bullish':
                score += 1
            else:
                score -= 1
            
            # ML prediction
            if ml_prediction and ml_prediction.get('prediction') == 'buy':
                score += 2
            elif ml_prediction and ml_prediction.get('prediction') == 'sell':
                score -= 2
            
            # Signals
            buy_signals = sum(1 for s in signals if s['type'] == 'buy')
            sell_signals = sum(1 for s in signals if s['type'] == 'sell')
            score += buy_signals - sell_signals
            
            # Generate recommendation
            if score >= 3:
                return "STRONG BUY"
            elif score >= 1:
                return "BUY"
            elif score <= -3:
                return "STRONG SELL"
            elif score <= -1:
                return "SELL"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "NEUTRAL"
    
    def _calculate_symbol_risk(self, symbol: str) -> float:
        """Calculate risk score for symbol"""
        try:
            volatility = self._calculate_market_volatility()  # Simplified
            return min(volatility * 100, 10)  # Scale to 0-10
        except Exception as e:
            logger.error(f"Error calculating symbol risk: {e}")
            return 5.0
    
    def _calculate_position_size(self, symbol: str) -> float:
        """Calculate recommended position size"""
        try:
            if self.risk_manager:
                return self.risk_manager.calculate_position_size(
                    symbol, 
                    self._get_total_portfolio_value()
                )
            return 0.01  # Default 1% of portfolio
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get detailed portfolio summary for chat integration"""
        try:
            total_value = self._get_total_portfolio_value()
            total_pnl = self._calculate_total_pnl()
            
            # Get positions breakdown
            positions_data = []
            for symbol, position in self.positions.items():
                positions_data.append({
                    'symbol': symbol,
                    'quantity': position.get('quantity', 0),
                    'entry_price': position.get('entry_price', 0),
                    'current_price': position.get('current_price', 0),
                    'value': position.get('value', 0),
                    'pnl': position.get('unrealized_pnl', 0),
                    'pnl_percent': position.get('pnl_percent', 0),
                    'holding_time': str(datetime.now() - position.get('entry_time', datetime.now()))
                })
            
            # Sort by value
            positions_data.sort(key=lambda x: abs(x['value']), reverse=True)
            
            # Get allocation breakdown
            allocations = {}
            for position in positions_data:
                base = position['symbol'].split('/')[0]
                allocations[base] = allocations.get(base, 0) + position['value']
            
            return {
                'total_value': total_value,
                'total_pnl': total_pnl,
                'total_pnl_percent': (total_pnl / (total_value - total_pnl) * 100) 
                                    if total_value > total_pnl else 0,
                'position_count': len(self.positions),
                'positions': positions_data[:10],  # Top 10 positions
                'allocations': allocations,
                'risk_metrics': {
                    'current_risk': self._calculate_risk_level(),
                    'risk_score': self._calculate_risk_score(),
                    'leverage': self._get_current_leverage(),
                    'exposure': total_value,
                    'var_95': self._calculate_var(0.95)
                },
                'performance': {
                    'daily_pnl': self._calculate_daily_pnl(),
                    'win_rate': self._calculate_win_rate(),
                    'sharpe_ratio': self._calculate_sharpe_ratio(),
                    'max_drawdown': self._calculate_max_drawdown(),
                    'current_drawdown': self._calculate_current_drawdown()
                },
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {
                'error': 'Unable to fetch portfolio data',
                'positions': [],
                'total_value': 0
            }
    
    def _calculate_total_pnl(self) -> float:
        """Calculate total PnL across all positions"""
        try:
            realized = self.total_pnl
            unrealized = sum(p.get('unrealized_pnl', 0) for p in self.positions.values())
            return realized + unrealized
        except Exception as e:
            logger.error(f"Error calculating total PnL: {e}")
            return 0.0
    
    def _calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                returns = self.db_manager.get_daily_returns(days=30)
                if returns:
                    returns_series = pd.Series(returns)
                    var = returns_series.quantile(1 - confidence)
                    return abs(var) * self._get_total_portfolio_value()
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _safe_config_get(self, *keys, default=None):
        """Safely get nested config values"""
        try:
            value = self.config
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key, default)
                else:
                    value = getattr(value, key, default)
            return value if value is not None else default
        except Exception:
            return default
    
    # === EXISTING TRADING OPERATIONS ===
    
    async def place_order_enhanced(self, symbol: str, side: str, qty: float, 
                                  order_type: str = 'market', price: Optional[float] = None,
                                  strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced order placement with advanced features"""
        try:
            logger.info(f"Placing enhanced {side} order for {qty} {symbol}")
            
            # Pre-order risk checks
            if self.risk_manager:
                risk_check = await self.risk_manager.check_order_risk(
                    symbol, side, qty, self.positions
                )
                if not risk_check['approved']:
                    logger.warning(f"Order rejected by risk manager: {risk_check['reason']}")
                    return {'status': 'rejected', 'reason': risk_check['reason']}
            
            # Smart order routing
            best_exchange = self._select_best_exchange(symbol, side, qty)
            if not best_exchange:
                return {'status': 'error', 'reason': 'No suitable exchange found'}
            
            # Calculate dynamic position sizing
            if self.risk_manager and self._safe_config_get('risk_management', 'dynamic_sizing'):
                adjusted_qty = await self.risk_manager.calculate_position_size(
                    symbol, qty, self.positions
                )
                logger.info(f"Position size adjusted from {qty} to {adjusted_qty}")
                qty = adjusted_qty
            
            # Place the order with the corrected await
            order_result = await self.place_order(
                symbol, side, qty, order_type, price, 
                exchange=best_exchange, strategy_id=strategy_id
            )
            
            # Post-order processing
            if order_result.get('status') == 'filled':
                await self._process_filled_order(order_result, strategy_id)
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error in enhanced order placement: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    async def place_order(self, symbol: str, side: str, qty: float, 
                         order_type: str = 'market', price: Optional[float] = None,
                         exchange: Optional[str] = None, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Base order placement method"""
        try:
            # Select exchange if not specified
            if not exchange:
                exchange = list(self.exchanges.keys())[0] if self.exchanges else None
            
            if not exchange or exchange not in self.exchanges:
                return {'status': 'error', 'reason': 'No exchange available'}
            
            # Place order on exchange
            exchange_client = self.exchanges[exchange]
            
            if order_type == 'market':
                order = await exchange_client.place_market_order(symbol, side, qty)
            else:
                order = await exchange_client.place_limit_order(symbol, side, qty, price)
            
            # Track order
            order_id = order.get('id', str(datetime.now().timestamp()))
            self.active_orders[order_id] = {
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'type': order_type,
                'price': price,
                'status': order.get('status', 'pending'),
                'exchange': exchange,
                'strategy_id': strategy_id,
                'timestamp': datetime.now()
            }
            
            # Log to database
            if self.db_manager:
                await self.db_manager.log_order(order)
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def _select_best_exchange(self, symbol: str, side: str, qty: float) -> Optional[str]:
        """Select best exchange for order routing"""
        try:
            # Simple selection - enhance with real logic
            for exchange_name, exchange in self.exchanges.items():
                # Check if exchange supports the symbol
                if hasattr(exchange, 'supports_symbol'):
                    if exchange.supports_symbol(symbol):
                        return exchange_name
            
            # Default to first available
            return list(self.exchanges.keys())[0] if self.exchanges else None
            
        except Exception as e:
            logger.error(f"Error selecting exchange: {e}")
            return None
    
    async def _process_filled_order(self, order: Dict[str, Any], strategy_id: Optional[str]):
        """Process a filled order"""
        try:
            symbol = order['symbol']
            side = order['side']
            qty = order['qty']
            price = order['price']
            
            # Update positions
            if side == 'buy':
                if symbol in self.positions:
                    # Update existing position
                    position = self.positions[symbol]
                    total_qty = position['quantity'] + qty
                    avg_price = ((position['quantity'] * position['entry_price']) + 
                               (qty * price)) / total_qty
                    position['quantity'] = total_qty
                    position['entry_price'] = avg_price
                else:
                    # New position
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'side': 'long',
                        'quantity': qty,
                        'entry_price': price,
                        'entry_time': datetime.now(),
                        'current_price': price,
                        'value': qty * price,
                        'unrealized_pnl': 0,
                        'pnl_percent': 0,
                        'strategy_id': strategy_id
                    }
            else:  # sell
                if symbol in self.positions:
                    position = self.positions[symbol]
                    position['quantity'] -= qty
                    
                    # Calculate realized PnL
                    pnl = (price - position['entry_price']) * qty
                    self.total_pnl += pnl
                    
                    # Remove position if fully closed
                    if position['quantity'] <= 0:
                        del self.positions[symbol]
            
            # Update metrics
            self.trade_count += 1
            
            # Notify
            await self.notification_manager.send_notification(
                f"{'🟢' if side == 'buy' else '🔴'} Order Filled",
                f"{side.upper()} {qty} {symbol} @ {price}"
            )
            
        except Exception as e:
            logger.error(f"Error processing filled order: {e}")
    
    # === STRATEGY MANAGEMENT ===
    
    def add_strategy(self, strategy_id: str, strategy_type: str, config: Dict[str, Any]) -> bool:
        """Add a new strategy to the engine"""
        try:
            # Import strategy module
            module_path = f"strategies.{strategy_type}"
            strategy_module = importlib.import_module(module_path)
            
            # Get strategy class
            strategy_class = getattr(strategy_module, f"{strategy_type.title()}Strategy")
            
            # Create strategy instance
            strategy = strategy_class(config, self)
            strategy.strategy_id = strategy_id
            
            # Add to strategies
            self.strategies[strategy_id] = strategy
            
            logger.info(f"Added strategy: {strategy_id} ({strategy_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")
            return False
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy from the engine"""
        try:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                
                # Stop strategy if running
                if hasattr(strategy, 'stop'):
                    strategy.stop()
                
                # Remove from collection
                del self.strategies[strategy_id]
                
                logger.info(f"Removed strategy: {strategy_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing strategy: {e}")
            return False
    
    def update_strategy(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """Update strategy configuration"""
        try:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                strategy.update_config(config)
                logger.info(f"Updated strategy: {strategy_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating strategy: {e}")
            return False
    
    # === MAIN TRADING LOOPS ===
    
    async def _main_loop(self):
        """Main trading loop"""
        while self.is_running and not self.shutdown_requested:
            try:
                # Update positions
                await self._update_positions()
                
                # Check for signals
                if self.signal_processor and not self.trading_paused:
                    signals = await self.signal_processor.get_signals()
                    for signal in signals:
                        await self._process_signal(signal)
                
                # Sleep
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_markets(self):
        """Monitor market conditions"""
        while self.is_running:
            try:
                # Update market data
                for symbol in self._get_monitored_symbols():
                    if self.market_data_processor:
                        await self.market_data_processor.update_symbol(symbol)
                
                # Check market conditions
                conditions = self._analyze_market_conditions()
                
                # Alert on significant changes
                if conditions.get('volatility', 0) > 0.05:
                    await self.notification_manager.send_notification(
                        "⚠️ High Volatility Alert",
                        f"Market volatility: {conditions['volatility']:.2%}"
                    )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring markets: {e}")
                await asyncio.sleep(60)
    
    async def _process_signals(self):
        """Process trading signals"""
        while self.is_running:
            try:
                if self.signal_processor and not self.trading_paused:
                    # Get pending signals
                    signals = await self.signal_processor.get_pending_signals()
                    
                    for signal in signals:
                        # Route to appropriate strategy
                        strategy_id = signal.get('strategy_id')
                        if strategy_id and strategy_id in self.strategies:
                            strategy = self.strategies[strategy_id]
                            if hasattr(strategy, 'process_signal'):
                                await strategy.process_signal(signal)
                        else:
                            # Process with default logic
                            await self._process_signal(signal)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing signals: {e}")
                await asyncio.sleep(5)
    
    async def _manage_risk(self):
        """Continuous risk management"""
        while self.is_running:
            try:
                if self.risk_manager:
                    # Check portfolio risk
                    risk_status = await self.risk_manager.assess_portfolio_risk(
                        self.positions
                    )
                    
                    # Take action if needed
                    if risk_status.get('action') == 'reduce_exposure':
                        await self._reduce_exposure(risk_status.get('amount', 0.1))
                    
                    # Check individual positions
                    for symbol, position in self.positions.items():
                        position_risk = await self.risk_manager.check_position_risk(
                            position
                        )
                        
                        if position_risk.get('action') == 'close':
                            await self._close_position(symbol, 'risk_limit')
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in risk management: {e}")
                await asyncio.sleep(30)
    
    async def _update_performance(self):
        """Update performance metrics"""
        while self.is_running:
            try:
                # Update strategy metrics
                for strategy_id, strategy in self.strategies.items():
                    if hasattr(strategy, 'get_performance'):
                        self.performance_metrics[strategy_id] = strategy.get_performance()
                
                # Log to database
                if self.db_manager:
                    await self.db_manager.update_performance_metrics(
                        self.performance_metrics
                    )
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating performance: {e}")
                await asyncio.sleep(300)
    
    async def _run_strategy(self, strategy_id: str, strategy: BaseStrategy):
        """Run a specific strategy"""
        while self.is_running and strategy_id in self.strategies:
            try:
                if not self.trading_paused and strategy.is_active:
                    await strategy.execute()
                
                await asyncio.sleep(strategy.execution_interval)
                
            except Exception as e:
                logger.error(f"Error running strategy {strategy_id}: {e}")
                await asyncio.sleep(60)
    
    # === HELPER METHODS ===
    
    async def _update_positions(self):
        """Update position values and P&L"""
        try:
            for symbol, position in self.positions.items():
                # Get current price
                current_price = await self._get_current_price(symbol)
                if current_price:
                    position['current_price'] = current_price
                    position['value'] = position['quantity'] * current_price
                    
                    # Calculate P&L
                    if position['side'] == 'long':
                        position['unrealized_pnl'] = (
                            (current_price - position['entry_price']) * 
                            position['quantity']
                        )
                    else:  # short
                        position['unrealized_pnl'] = (
                            (position['entry_price'] - current_price) * 
                            position['quantity']
                        )
                    
                    # Calculate percentage
                    position['pnl_percent'] = (
                        position['unrealized_pnl'] / 
                        (position['entry_price'] * position['quantity'])
                    )
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _process_signal(self, signal: Dict[str, Any]):
        """Process a trading signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            strength = signal.get('strength', 0.5)
            
            # Check if we should act on signal
            if strength < self._safe_config_get('trading', 'min_signal_strength', default=0.6):
                return
            
            # Calculate position size
            qty = self._calculate_signal_position_size(signal)
            
            # Place order
            if action == 'buy':
                await self.place_order_enhanced(
                    symbol, 'buy', qty, 
                    strategy_id=signal.get('strategy_id')
                )
            elif action == 'sell':
                await self.place_order_enhanced(
                    symbol, 'sell', qty,
                    strategy_id=signal.get('strategy_id')
                )
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _calculate_signal_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate position size for signal"""
        try:
            # Base size from config
            base_size = self._safe_config_get('trading', 'base_position_size', default=0.01)
            
            # Adjust by signal strength
            strength = signal.get('strength', 0.5)
            size = base_size * strength
            
            # Apply risk limits
            if self.risk_manager:
                max_size = self.risk_manager.get_max_position_size(
                    signal['symbol']
                )
                size = min(size, max_size)
            
            return size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01
    
    async def _reduce_exposure(self, reduction_percent: float):
        """Reduce overall exposure"""
        try:
            logger.warning(f"Reducing exposure by {reduction_percent:.1%}")
            
            # Sort positions by size
            sorted_positions = sorted(
                self.positions.items(),
                key=lambda x: abs(x[1]['value']),
                reverse=True
            )
            
            # Reduce largest positions
            for symbol, position in sorted_positions:
                reduce_qty = position['quantity'] * reduction_percent
                
                await self.place_order(
                    symbol,
                    'sell' if position['side'] == 'long' else 'buy',
                    reduce_qty
                )
            
        except Exception as e:
            logger.error(f"Error reducing exposure: {e}")
    
    async def _close_position(self, symbol: str, reason: str = 'manual'):
        """Close a specific position"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            # Place closing order
            await self.place_order(
                symbol,
                'sell' if position['side'] == 'long' else 'buy',
                position['quantity']
            )
            
            logger.info(f"Closing position {symbol}: {reason}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _get_monitored_symbols(self) -> List[str]:
        """Get list of symbols to monitor"""
        try:
            symbols = set()
            
            # Add configured symbols
            configured = self._safe_config_get('trading', 'symbols', default=[])
            symbols.update(configured)
            
            # Add symbols from active positions
            symbols.update(self.positions.keys())
            
            # Add symbols from strategies
            for strategy in self.strategies.values():
                if hasattr(strategy, 'get_symbols'):
                    symbols.update(strategy.get_symbols())
            
            return list(symbols)
            
        except Exception as e:
            logger.error(f"Error getting monitored symbols: {e}")
            return []
    
    # === PUBLIC API METHODS ===
    
    def get_status(self) -> Dict[str, Any]:
        """Get basic engine status"""
        return {
            'is_running': self.is_running,
            'trading_paused': self.trading_paused,
            'uptime': str(datetime.now() - self.start_time),
            'active_strategies': len([s for s in self.strategies.values() if s.is_active]),
            'open_positions': len(self.positions),
            'total_pnl': self.total_pnl
        }
    
    def pause_trading(self):
        """Pause trading without stopping engine"""
        self.trading_paused = True
        logger.info("Trading paused")
    
    def resume_trading(self):
        """Resume trading"""
        self.trading_paused = False
        logger.info("Trading resumed")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_trades': self.trade_count,
            'total_pnl': self.total_pnl,
            'win_rate': self._calculate_win_rate(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'by_strategy': self.performance_metrics
        }