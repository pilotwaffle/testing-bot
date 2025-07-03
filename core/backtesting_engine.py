# backtesting_engine.py - Advanced Backtesting Engine
"""
Advanced Backtesting Engine
Comprehensive backtesting framework with walk-forward analysis and performance metrics
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import our components
from enhanced_data_fetcher import EnhancedDataFetcher
from enhanced_ml_engine import AdaptiveMLEngine
from enhanced_trading_strategy import EnhancedTradingStrategy, TradingSignal, TradeDirection

@dataclass
class BacktestTrade:
    """Individual backtest trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    commission: float
    slippage: float
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    hold_time: Optional[timedelta]
    exit_reason: Optional[str]
    signal_confidence: float
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_hold_time: timedelta
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    drawdown_series: pd.Series

class AdvancedBacktester:
    """Advanced backtesting engine with comprehensive analysis"""
    
    def __init__(self, config_path: str = "config/backtest_config.json"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_fetcher = EnhancedDataFetcher(
            exchange_name=self.config['data']['exchange_name'],
            cache_dir=self.config['data']['cache_dir']
        )
        
        self.ml_engine = AdaptiveMLEngine(
            model_save_path=self.config['strategy']['model_path']
        )
        
        self.strategy_engine = EnhancedTradingStrategy(
            config_path=self.config['strategy']['strategy_config_path']
        )
        
        # Backtest state
        self.current_positions = {}
        self.trade_history = []
        self.equity_history = []
        self.cash = self.config['execution']['initial_capital']
        self.portfolio_value = self.cash
        
        # Performance tracking
        self.daily_returns = []
        self.portfolio_values = []
        self.timestamps = []
        
        self.logger.info("Advanced Backtester initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load backtest configuration"""
        default_config = {
            "data": {
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "symbols": ["BTC/USD", "ETH/USD", "ADA/USD"],
                "timeframes": ["1h", "4h", "1d"],
                "exchange_name": "kraken",
                "cache_dir": "data/cache/"
            },
            "execution": {
                "initial_capital": 10000,
                "commission_rate": 0.001,
                "slippage_rate": 0.0005,
                "execution_delay_bars": 1,
                "partial_fills_enabled": True,
                "max_position_size": 0.25,
                "position_sizing_method": "fixed_fractional"
            },
            "strategy": {
                "model_path": "models/",
                "strategy_config_path": "config/strategy_config.json",
                "walk_forward_enabled": True,
                "walk_forward_period_months": 3,
                "retraining_frequency_days": 30
            },
            "analysis": {
                "benchmark_symbol": "BTC/USD",
                "risk_free_rate": 0.02,
                "confidence_level": 0.05,
                "monte_carlo_simulations": 1000
            },
            "output": {
                "save_trades": True,
                "save_equity_curve": True,
                "generate_plots": True,
                "output_directory": "backtest_results/"
            }
        }
        
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                self._deep_update(default_config, loaded_config)
                self.logger.info(f"Backtest configuration loaded from {config_path}")
            else:
                self.logger.info("Using default backtest configuration")
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error loading backtest config: {e}. Using defaults.")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def run_backtest(self) -> BacktestResults:
        """Run comprehensive backtest"""
        self.logger.info("Starting comprehensive backtest...")
        
        try:
            # Parse dates
            start_date = pd.to_datetime(self.config['data']['start_date'])
            end_date = pd.to_datetime(self.config['data']['end_date'])
            
            # Fetch historical data
            self.logger.info("Fetching historical data...")
            historical_data = self._fetch_historical_data(start_date, end_date)
            
            # Run walk-forward analysis if enabled
            if self.config['strategy']['walk_forward_enabled']:
                results = self._run_walk_forward_backtest(historical_data, start_date, end_date)
            else:
                results = self._run_simple_backtest(historical_data, start_date, end_date)
            
            # Generate comprehensive analysis
            self._generate_analysis_report(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    def _fetch_historical_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch historical data for all symbols and timeframes"""
        
        # Calculate required number of candles
        total_days = (end_date - start_date).days
        max_candles = max(2000, total_days * 24)  # Ensure enough data
        
        self.logger.info(f"Fetching {max_candles} candles for {len(self.config['data']['symbols'])} symbols...")
        
        historical_data = self.data_fetcher.fetch_multiple_symbols(
            symbols=self.config['data']['symbols'],
            timeframes=self.config['data']['timeframes'],
            total_candles=max_candles
        )
        
        # Filter data to backtest period
        filtered_data = {}
        for symbol, timeframe_data in historical_data.items():
            filtered_data[symbol] = {}
            for timeframe, data in timeframe_data.items():
                if len(data) > 0:
                    # Filter to backtest period
                    mask = (data.index >= start_date) & (data.index <= end_date)
                    filtered_data[symbol][timeframe] = data[mask]
                    
                    self.logger.info(f"{symbol} ({timeframe}): {len(filtered_data[symbol][timeframe])} bars")
        
        return filtered_data
    
    def _run_walk_forward_backtest(self, historical_data: Dict[str, Dict[str, pd.DataFrame]], 
                                   start_date: datetime, end_date: datetime) -> BacktestResults:
        """Run walk-forward analysis backtest"""
        self.logger.info("Running walk-forward analysis...")
        
        # Reset backtest state
        self._reset_backtest_state()
        
        # Walk-forward parameters
        training_period = timedelta(days=self.config['strategy']['walk_forward_period_months'] * 30)
        test_period = timedelta(days=30)  # Test on 1 month of data
        retrain_frequency = timedelta(days=self.config['strategy']['retraining_frequency_days'])
        
        current_date = start_date + training_period
        last_retrain_date = start_date
        
        while current_date <= end_date:
            test_start = current_date
            test_end = min(current_date + test_period, end_date)
            
            self.logger.info(f"Testing period: {test_start.date()} to {test_end.date()}")
            
            # Check if we need to retrain models
            if current_date - last_retrain_date >= retrain_frequency:
                self.logger.info("Retraining models...")
                training_start = max(start_date, current_date - training_period)
                self._retrain_models(historical_data, training_start, current_date)
                last_retrain_date = current_date
            
            # Run backtest for this period
            self._run_backtest_period(historical_data, test_start, test_end)
            
            current_date = test_end
        
        # Calculate final results
        return self._calculate_backtest_results(start_date, end_date)
    
    def _run_simple_backtest(self, historical_data: Dict[str, Dict[str, pd.DataFrame]], 
                            start_date: datetime, end_date: datetime) -> BacktestResults:
        """Run simple backtest without walk-forward analysis"""
        self.logger.info("Running simple backtest...")
        
        # Reset backtest state
        self._reset_backtest_state()
        
        # Use existing models or train once at the beginning
        training_end = start_date + timedelta(days=90)  # Use first 3 months for training
        self._retrain_models(historical_data, start_date, training_end)
        
        # Run backtest for entire period
        backtest_start = training_end
        self._run_backtest_period(historical_data, backtest_start, end_date)
        
        return self._calculate_backtest_results(backtest_start, end_date)
    
    def _reset_backtest_state(self):
        """Reset backtest state"""
        self.current_positions = {}
        self.trade_history = []
        self.equity_history = []
        self.cash = self.config['execution']['initial_capital']
        self.portfolio_value = self.cash
        self.daily_returns = []
        self.portfolio_values = []
        self.timestamps = []
    
    def _retrain_models(self, historical_data: Dict[str, Dict[str, pd.DataFrame]], 
                       start_date: datetime, end_date: datetime):
        """Retrain ML models for walk-forward analysis"""
        try:
            for symbol in self.config['data']['symbols']:
                for timeframe in self.config['data']['timeframes']:
                    if symbol in historical_data and timeframe in historical_data[symbol]:
                        data = historical_data[symbol][timeframe]
                        
                        # Filter to training period
                        training_mask = (data.index >= start_date) & (data.index <= end_date)
                        training_data = data[training_mask]
                        
                        if len(training_data) > 100:  # Minimum data requirement
                            # Enrich data
                            enriched_data = self.data_fetcher.enrich_with_market_data(training_data, symbol)
                            
                            # Train models
                            self.ml_engine.train_ensemble_model(
                                symbol=symbol,
                                timeframe=timeframe,
                                data=enriched_data
                            )
        
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    def _run_backtest_period(self, historical_data: Dict[str, Dict[str, pd.DataFrame]], 
                            start_date: datetime, end_date: datetime):
        """Run backtest for specific period"""
        
        # Get the primary timeframe for iteration (usually the shortest)
        primary_timeframe = self.config['data']['timeframes'][0]
        
        # Find date range that exists in all symbols
        all_dates = set()
        for symbol in self.config['data']['symbols']:
            if symbol in historical_data and primary_timeframe in historical_data[symbol]:
                data = historical_data[symbol][primary_timeframe]
                period_mask = (data.index >= start_date) & (data.index <= end_date)
                period_data = data[period_mask]
                all_dates.update(period_data.index)
        
        # Sort dates
        sorted_dates = sorted(all_dates)
        
        if not sorted_dates:
            self.logger.warning(f"No data available for period {start_date} to {end_date}")
            return
        
        self.logger.info(f"Processing {len(sorted_dates)} time steps...")
        
        # Process each time step
        for i, current_time in enumerate(sorted_dates):
            try:
                # Update portfolio value tracking
                self._update_portfolio_value(current_time, historical_data)
                
                # Process existing positions
                self._process_existing_positions(current_time, historical_data)
                
                # Generate new signals every N bars (to simulate signal generation frequency)
                if i % self.config['execution']['execution_delay_bars'] == 0:
                    signals = self._generate_signals_at_time(current_time, historical_data)
                    
                    # Process signals
                    for signal in signals:
                        self._process_signal(signal, current_time, historical_data)
                
                # Log progress
                if i % 1000 == 0:
                    progress = (i / len(sorted_dates)) * 100
                    self.logger.debug(f"Progress: {progress:.1f}% - Portfolio: ${self.portfolio_value:.2f}")
            
            except Exception as e:
                self.logger.error(f"Error processing time {current_time}: {e}")
                continue
    
    def _update_portfolio_value(self, current_time: datetime, 
                               historical_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Update portfolio value at current time"""
        
        position_value = 0.0
        
        # Calculate value of current positions
        for symbol, position in self.current_positions.items():
            try:
                current_price = self._get_price_at_time(symbol, current_time, historical_data)
                if current_price:
                    position_value += position['quantity'] * current_price
            except Exception as e:
                self.logger.error(f"Error calculating position value for {symbol}: {e}")
        
        self.portfolio_value = self.cash + position_value
        
        # Record for equity curve
        self.portfolio_values.append(self.portfolio_value)
        self.timestamps.append(current_time)
        
        # Calculate daily return
        if len(self.portfolio_values) > 1:
            daily_return = (self.portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.daily_returns.append(daily_return)
    
    def _process_existing_positions(self, current_time: datetime, 
                                   historical_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Process existing positions for stop loss, take profit, etc."""
        
        positions_to_close = []
        
        for symbol, position in self.current_positions.items():
            try:
                current_price = self._get_price_at_time(symbol, current_time, historical_data)
                
                if not current_price:
                    continue
                
                signal = position['signal']
                entry_price = position['entry_price']
                
                # Update MFE and MAE
                if signal.direction == TradeDirection.BUY:
                    # Long position
                    unrealized_pnl_pct = (current_price - entry_price) / entry_price
                    position['mfe'] = max(position.get('mfe', 0), unrealized_pnl_pct)
                    position['mae'] = min(position.get('mae', 0), unrealized_pnl_pct)
                    
                    # Check exit conditions
                    if current_price <= signal.stop_loss:
                        positions_to_close.append((symbol, current_price, "STOP_LOSS"))
                    elif current_price >= signal.take_profit:
                        positions_to_close.append((symbol, current_price, "TAKE_PROFIT"))
                
                elif signal.direction == TradeDirection.SELL:
                    # Short position
                    unrealized_pnl_pct = (entry_price - current_price) / entry_price
                    position['mfe'] = max(position.get('mfe', 0), unrealized_pnl_pct)
                    position['mae'] = min(position.get('mae', 0), unrealized_pnl_pct)
                    
                    # Check exit conditions
                    if current_price >= signal.stop_loss:
                        positions_to_close.append((symbol, current_price, "STOP_LOSS"))
                    elif current_price <= signal.take_profit:
                        positions_to_close.append((symbol, current_price, "TAKE_PROFIT"))
                
                # Check position timeout
                hold_time = current_time - position['entry_time']
                max_hold_time = timedelta(days=7)  # Max 7 days
                
                if hold_time > max_hold_time:
                    positions_to_close.append((symbol, current_price, "TIMEOUT"))
            
            except Exception as e:
                self.logger.error(f"Error processing position for {symbol}: {e}")
        
        # Close positions
        for symbol, exit_price, exit_reason in positions_to_close:
            self._close_position(symbol, exit_price, current_time, exit_reason)
    
    def _generate_signals_at_time(self, current_time: datetime, 
                                 historical_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[TradingSignal]:
        """Generate trading signals at specific time"""
        
        signals = []
        
        for symbol in self.config['data']['symbols']:
            try:
                # Skip if we already have a position
                if symbol in self.current_positions:
                    continue
                
                # Get market data up to current time
                symbol_data = {}
                for timeframe in self.config['data']['timeframes']:
                    if symbol in historical_data and timeframe in historical_data[symbol]:
                        data = historical_data[symbol][timeframe]
                        # Get data up to current time
                        historical_mask = data.index <= current_time
                        symbol_data[timeframe] = data[historical_mask].tail(500)  # Last 500 bars
                
                if not symbol_data:
                    continue
                
                # Get ML predictions
                ml_predictions = {}
                for timeframe in self.config['data']['timeframes']:
                    if timeframe in symbol_data and len(symbol_data[timeframe]) > 50:
                        prediction = self.ml_engine.predict_ensemble(
                            symbol=symbol,
                            timeframe=timeframe,
                            data=symbol_data[timeframe]
                        )
                        if prediction:
                            ml_predictions[timeframe] = prediction
                
                # Generate signal using strategy engine
                signal = self.strategy_engine.generate_trading_signal(
                    symbol=symbol,
                    market_data=symbol_data,
                    ml_predictions=ml_predictions,
                    current_positions=self.current_positions
                )
                
                if signal and signal.direction != TradeDirection.HOLD:
                    signals.append(signal)
            
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol} at {current_time}: {e}")
        
        return signals
    
    def _process_signal(self, signal: TradingSignal, current_time: datetime, 
                       historical_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Process trading signal"""
        
        try:
            # Get execution price (with slippage)
            base_price = self._get_price_at_time(signal.symbol, current_time, historical_data)
            
            if not base_price:
                return
            
            slippage_rate = self.config['execution']['slippage_rate']
            
            if signal.direction == TradeDirection.BUY:
                execution_price = base_price * (1 + slippage_rate)
            else:
                execution_price = base_price * (1 - slippage_rate)
            
            # Calculate position size
            max_position_value = self.portfolio_value * self.config['execution']['max_position_size']
            position_value = min(max_position_value, self.cash * signal.position_size)
            
            if position_value < 100:  # Minimum position size
                return
            
            quantity = position_value / execution_price
            commission = position_value * self.config['execution']['commission_rate']
            
            # Check if we have enough cash
            total_cost = position_value + commission
            if total_cost > self.cash:
                return
            
            # Execute trade
            self.cash -= total_cost
            
            # Create position
            position = {
                'signal': signal,
                'entry_time': current_time,
                'entry_price': execution_price,
                'quantity': quantity,
                'commission': commission,
                'mfe': 0.0,
                'mae': 0.0
            }
            
            self.current_positions[signal.symbol] = position
            
            self.logger.debug(f"Opened {signal.direction.value} position: {signal.symbol} "
                            f"@ ${execution_price:.2f}, Qty: {quantity:.6f}")
        
        except Exception as e:
            self.logger.error(f"Error processing signal for {signal.symbol}: {e}")
    
    def _close_position(self, symbol: str, exit_price: float, exit_time: datetime, exit_reason: str):
        """Close existing position"""
        
        try:
            position = self.current_positions[symbol]
            signal = position['signal']
            
            # Calculate slippage on exit
            slippage_rate = self.config['execution']['slippage_rate']
            
            if signal.direction == TradeDirection.BUY:
                actual_exit_price = exit_price * (1 - slippage_rate)
            else:
                actual_exit_price = exit_price * (1 + slippage_rate)
            
            # Calculate P&L
            quantity = position['quantity']
            entry_price = position['entry_price']
            
            if signal.direction == TradeDirection.BUY:
                gross_profit = (actual_exit_price - entry_price) * quantity
            else:
                gross_profit = (entry_price - actual_exit_price) * quantity
            
            exit_commission = quantity * actual_exit_price * self.config['execution']['commission_rate']
            net_profit = gross_profit - position['commission'] - exit_commission
            profit_pct = net_profit / (entry_price * quantity) * 100
            
            # Add proceeds to cash
            proceeds = quantity * actual_exit_price - exit_commission
            self.cash += proceeds
            
            # Create trade record
            trade = BacktestTrade(
                entry_time=position['entry_time'],
                exit_time=exit_time,
                symbol=symbol,
                direction=signal.direction.value,
                entry_price=entry_price,
                exit_price=actual_exit_price,
                quantity=quantity,
                commission=position['commission'] + exit_commission,
                slippage=abs(exit_price - actual_exit_price) * quantity,
                profit_loss=net_profit,
                profit_loss_pct=profit_pct,
                hold_time=exit_time - position['entry_time'],
                exit_reason=exit_reason,
                signal_confidence=signal.confidence,
                max_favorable_excursion=position.get('mfe', 0) * 100,
                max_adverse_excursion=position.get('mae', 0) * 100
            )
            
            self.trade_history.append(trade)
            
            # Remove from current positions
            del self.current_positions[symbol]
            
            self.logger.debug(f"Closed {signal.direction.value} position: {symbol} "
                            f"@ ${actual_exit_price:.2f}, P&L: ${net_profit:.2f} ({profit_pct:.2f}%)")
        
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
    
    def _get_price_at_time(self, symbol: str, time: datetime, 
                          historical_data: Dict[str, Dict[str, pd.DataFrame]]) -> Optional[float]:
        """Get price for symbol at specific time"""
        
        try:
            # Use shortest timeframe for most accurate price
            for timeframe in self.config['data']['timeframes']:
                if symbol in historical_data and timeframe in historical_data[symbol]:
                    data = historical_data[symbol][timeframe]
                    
                    # Find closest time
                    if time in data.index:
                        return float(data.loc[time, 'close'])
                    else:
                        # Find nearest previous time
                        before_time = data.index[data.index <= time]
                        if len(before_time) > 0:
                            return float(data.loc[before_time[-1], 'close'])
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol} at {time}: {e}")
            return None
    
    def _calculate_backtest_results(self, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        try:
            # Close any remaining positions
            final_time = end_date
            for symbol in list(self.current_positions.keys()):
                # Use last known price
                if self.portfolio_values:
                    # Estimate final price
                    position = self.current_positions[symbol]
                    entry_price = position['entry_price']
                    # Use entry price as approximation for final close
                    self._close_position(symbol, entry_price, final_time, "BACKTEST_END")
            
            # Basic metrics
            initial_capital = self.config['execution']['initial_capital']
            final_capital = self.portfolio_value
            total_return = final_capital - initial_capital
            total_return_pct = (total_return / initial_capital) * 100
            
            # Time period calculations
            total_days = (end_date - start_date).days
            years = total_days / 365.25
            annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
            
            # Create equity curve
            equity_curve = pd.Series(self.portfolio_values, index=self.timestamps[:len(self.portfolio_values)])
            
            # Drawdown calculations
            running_max = equity_curve.expanding().max()
            drawdown_series = (equity_curve - running_max) / running_max * 100
            max_drawdown = abs(drawdown_series.min())
            
            # Find max drawdown duration
            drawdown_periods = []
            in_drawdown = False
            start_dd = None
            
            for date, dd in drawdown_series.items():
                if dd < -0.01 and not in_drawdown:  # Start of drawdown
                    in_drawdown = True
                    start_dd = date
                elif dd >= -0.01 and in_drawdown:  # End of drawdown
                    in_drawdown = False
                    if start_dd:
                        drawdown_periods.append(date - start_dd)
            
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else timedelta(0)
            
            # Risk metrics
            risk_free_rate = self.config['analysis']['risk_free_rate']
            
            if len(self.daily_returns) > 1:
                daily_returns_series = pd.Series(self.daily_returns)
                excess_returns = daily_returns_series - risk_free_rate / 252  # Daily risk-free rate
                
                sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
                
                # Sortino ratio (downside deviation)
                downside_returns = excess_returns[excess_returns < 0]
                sortino_ratio = (excess_returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
                
                # Calmar ratio
                calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            else:
                sharpe_ratio = sortino_ratio = calmar_ratio = 0
            
            # Trade statistics
            if self.trade_history:
                winning_trades = [t for t in self.trade_history if t.profit_loss > 0]
                losing_trades = [t for t in self.trade_history if t.profit_loss < 0]
                
                win_rate = len(winning_trades) / len(self.trade_history) * 100
                
                avg_win = np.mean([t.profit_loss for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t.profit_loss for t in losing_trades]) if losing_trades else 0
                
                profit_factor = abs(sum(t.profit_loss for t in winning_trades) / sum(t.profit_loss for t in losing_trades)) if losing_trades and sum(t.profit_loss for t in losing_trades) != 0 else 0
                
                largest_win = max(t.profit_loss for t in self.trade_history)
                largest_loss = min(t.profit_loss for t in self.trade_history)
                
                # Calculate consecutive wins/losses
                consecutive_wins = consecutive_losses = 0
                current_wins = current_losses = 0
                
                for trade in self.trade_history:
                    if trade.profit_loss > 0:
                        current_wins += 1
                        current_losses = 0
                        consecutive_wins = max(consecutive_wins, current_wins)
                    else:
                        current_losses += 1
                        current_wins = 0
                        consecutive_losses = max(consecutive_losses, current_losses)
                
                # Average hold time
                hold_times = [t.hold_time for t in self.trade_history if t.hold_time]
                avg_hold_time = sum(hold_times, timedelta(0)) / len(hold_times) if hold_times else timedelta(0)
            
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
                largest_win = largest_loss = 0
                consecutive_wins = consecutive_losses = 0
                avg_hold_time = timedelta(0)
                winning_trades = losing_trades = []
            
            return BacktestResults(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                total_return_pct=total_return_pct,
                annualized_return=annualized_return,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                avg_hold_time=avg_hold_time,
                total_trades=len(self.trade_history),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                trades=self.trade_history,
                equity_curve=equity_curve,
                drawdown_series=drawdown_series
            )
        
        except Exception as e:
            self.logger.error(f"Error calculating backtest results: {e}")
            raise
    
    def _generate_analysis_report(self, results: BacktestResults):
        """Generate comprehensive analysis report"""
        
        try:
            output_dir = Path(self.config['output']['output_directory'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save results as JSON
            if self.config['output']['save_trades']:
                results_dict = asdict(results)
                # Convert non-serializable objects
                results_dict['trades'] = [asdict(trade) for trade in results.trades]
                results_dict['equity_curve'] = results.equity_curve.to_dict()
                results_dict['drawdown_series'] = results.drawdown_series.to_dict()
                
                with open(output_dir / f'backtest_results_{timestamp}.json', 'w') as f:
                    json.dump(results_dict, f, indent=2, default=str)
            
            # Generate plots
            if self.config['output']['generate_plots']:
                self._generate_performance_plots(results, output_dir, timestamp)
            
            # Generate text report
            self._generate_text_report(results, output_dir, timestamp)
            
            self.logger.info(f"Analysis report saved to {output_dir}")
        
        except Exception as e:
            self.logger.error(f"Error generating analysis report: {e}")
    
    def _generate_performance_plots(self, results: BacktestResults, output_dir: Path, timestamp: str):
        """Generate performance visualization plots"""
        
        try:
            # Set style
            plt.style.use('dark_background')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Backtest Performance Analysis', fontsize=16)
            
            # 1. Equity Curve
            ax1 = axes[0, 0]
            results.equity_curve.plot(ax=ax1, color='cyan', linewidth=2)
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Drawdown
            ax2 = axes[0, 1]
            results.drawdown_series.plot(ax=ax2, color='red', linewidth=2)
            ax2.fill_between(results.drawdown_series.index, results.drawdown_series, 0, alpha=0.3, color='red')
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            
            # 3. Trade P&L Distribution
            ax3 = axes[1, 0]
            if results.trades:
                pnl_values = [trade.profit_loss for trade in results.trades]
                ax3.hist(pnl_values, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                ax3.axvline(0, color='red', linestyle='--', linewidth=2)
                ax3.set_title('Trade P&L Distribution')
                ax3.set_xlabel('Profit/Loss ($)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
            
            # 4. Monthly Returns Heatmap
            ax4 = axes[1, 1]
            if len(results.equity_curve) > 30:
                monthly_returns = results.equity_curve.resample('M').last().pct_change().dropna()
                if len(monthly_returns) > 1:
                    # Create monthly returns matrix
                    monthly_returns.index = pd.to_datetime(monthly_returns.index)
                    monthly_data = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
                    
                    if not monthly_data.empty:
                        sns.heatmap(monthly_data * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax4)
                        ax4.set_title('Monthly Returns (%)')
                        ax4.set_xlabel('Month')
                        ax4.set_ylabel('Year')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'backtest_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Additional trade analysis plot
            if results.trades:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Cumulative returns by trade
                ax1 = axes[0]
                cumulative_pnl = np.cumsum([trade.profit_loss for trade in results.trades])
                ax1.plot(range(len(cumulative_pnl)), cumulative_pnl, color='cyan', linewidth=2)
                ax1.set_title('Cumulative P&L by Trade')
                ax1.set_xlabel('Trade Number')
                ax1.set_ylabel('Cumulative P&L ($)')
                ax1.grid(True, alpha=0.3)
                
                # Trade duration vs P&L
                ax2 = axes[1]
                durations = [trade.hold_time.total_seconds() / 3600 for trade in results.trades if trade.hold_time]  # Hours
                pnl_values = [trade.profit_loss for trade in results.trades if trade.hold_time]
                colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
                
                ax2.scatter(durations, pnl_values, c=colors, alpha=0.6)
                ax2.set_title('Trade Duration vs P&L')
                ax2.set_xlabel('Hold Time (Hours)')
                ax2.set_ylabel('P&L ($)')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'trade_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
    
    def _generate_text_report(self, results: BacktestResults, output_dir: Path, timestamp: str):
        """Generate detailed text report"""
        
        try:
            report_lines = [
                "="*80,
                "ENHANCED TRADING BOT - BACKTEST RESULTS",
                "="*80,
                "",
                f"Backtest Period: {results.start_date.date()} to {results.end_date.date()}",
                f"Total Duration: {(results.end_date - results.start_date).days} days",
                "",
                "PERFORMANCE SUMMARY",
                "-"*40,
                f"Initial Capital:      ${results.initial_capital:,.2f}",
                f"Final Capital:        ${results.final_capital:,.2f}",
                f"Total Return:         ${results.total_return:,.2f} ({results.total_return_pct:.2f}%)",
                f"Annualized Return:    {results.annualized_return:.2f}%",
                "",
                "RISK METRICS",
                "-"*40,
                f"Maximum Drawdown:     {results.max_drawdown:.2f}%",
                f"Max DD Duration:      {results.max_drawdown_duration}",
                f"Sharpe Ratio:         {results.sharpe_ratio:.3f}",
                f"Sortino Ratio:        {results.sortino_ratio:.3f}",
                f"Calmar Ratio:         {results.calmar_ratio:.3f}",
                "",
                "TRADE STATISTICS",
                "-"*40,
                f"Total Trades:         {results.total_trades}",
                f"Winning Trades:       {results.winning_trades} ({results.win_rate:.1f}%)",
                f"Losing Trades:        {results.losing_trades}",
                f"Profit Factor:        {results.profit_factor:.2f}",
                f"Average Win:          ${results.avg_win:.2f}",
                f"Average Loss:         ${results.avg_loss:.2f}",
                f"Largest Win:          ${results.largest_win:.2f}",
                f"Largest Loss:         ${results.largest_loss:.2f}",
                f"Average Hold Time:    {results.avg_hold_time}",
                f"Max Consecutive Wins: {results.consecutive_wins}",
                f"Max Consecutive Loss: {results.consecutive_losses}",
                ""
            ]
            
            # Add trade details if available
            if results.trades:
                report_lines.extend([
                    "DETAILED TRADE LOG",
                    "-"*40,
                    "Entry Time          | Symbol    | Dir | Entry     | Exit      | P&L      | Hold Time",
                    "-"*80
                ])
                
                for trade in results.trades[-20:]:  # Last 20 trades
                    entry_time = trade.entry_time.strftime('%Y-%m-%d %H:%M')
                    hold_time_str = str(trade.hold_time).split('.')[0] if trade.hold_time else "N/A"
                    
                    line = f"{entry_time} | {trade.symbol:8s} | {trade.direction:3s} | ${trade.entry_price:8.2f} | ${trade.exit_price or 0:8.2f} | ${trade.profit_loss or 0:8.2f} | {hold_time_str}"
                    report_lines.append(line)
            
            # Save report
            with open(output_dir / f'backtest_report_{timestamp}.txt', 'w') as f:
                f.write('\n'.join(report_lines))
            
            # Print summary to console
            print('\n'.join(report_lines[:30]))  # Print first 30 lines
        
        except Exception as e:
            self.logger.error(f"Error generating text report: {e}")

def main():
    """Main entry point for backtesting"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Trading Bot Backtester')
    parser.add_argument('--config', default='config/backtest_config.json',
                       help='Path to backtest configuration file')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to backtest')
    parser.add_argument('--initial-capital', type=float, help='Initial capital')
    parser.add_argument('--walk-forward', action='store_true', help='Enable walk-forward analysis')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize backtester
        backtester = AdvancedBacktester(config_path=args.config)
        
        # Override config with command line arguments
        if args.start_date:
            backtester.config['data']['start_date'] = args.start_date
        if args.end_date:
            backtester.config['data']['end_date'] = args.end_date
        if args.symbols:
            backtester.config['data']['symbols'] = args.symbols
        if args.initial_capital:
            backtester.config['execution']['initial_capital'] = args.initial_capital
        if args.walk_forward:
            backtester.config['strategy']['walk_forward_enabled'] = True
        
        # Run backtest
        results = backtester.run_backtest()
        
        print(f"\nBacktest completed successfully!")
        print(f"Results saved to: {backtester.config['output']['output_directory']}")
        
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
    except Exception as e:
        print(f"Backtest failed: {e}")
        logging.error(f"Backtest failed: {e}")

if __name__ == "__main__":
    main()