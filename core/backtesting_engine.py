# === NEW FILE: core/backtesting_engine.py ===
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"

@dataclass 
class BacktestTrade:
    """FreqTrade-style trade record"""
    pair: str
    open_date: datetime
    close_date: Optional[datetime] = None
    open_rate: float = 0.0
    close_rate: Optional[float] = None
    amount: float = 0.0
    fee_open: float = 0.001
    fee_close: float = 0.001
    profit_abs: Optional[float] = None
    profit_ratio: Optional[float] = None
    exit_reason: str = ""
    entry_tag: Optional[str] = None
    is_short: bool = False
    leverage: float = 1.0
    
    def calculate_profit(self):
        """Calculate trade profit"""
        if self.close_rate is None:
            return
            
        if self.is_short:
            profit_ratio = (self.open_rate - self.close_rate) / self.open_rate
        else:
            profit_ratio = (self.close_rate - self.open_rate) / self.open_rate
            
        # Account for fees
        profit_ratio = profit_ratio - self.fee_open - self.fee_close
        
        self.profit_ratio = profit_ratio
        self.profit_abs = profit_ratio * self.amount * self.open_rate

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    trades: List[BacktestTrade] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    profit_total_abs: float = 0.0
    profit_total_ratio: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return
            
        profits = [t.profit_ratio for t in self.trades if t.profit_ratio is not None]
        if not profits:
            return
            
        self.total_trades = len(self.trades)
        self.winning_trades = len([p for p in profits if p > 0])
        self.losing_trades = len([p for p in profits if p < 0])
        
        self.profit_total_ratio = sum(profits)
        self.profit_total_abs = sum([t.profit_abs for t in self.trades if t.profit_abs])
        
        # Calculate drawdown
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        self.max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Sharpe ratio
        if len(profits) > 1:
            self.sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(252) if np.std(profits) > 0 else 0
        
        # Profit factor
        winning_profits = sum([p for p in profits if p > 0])
        losing_profits = abs(sum([p for p in profits if p < 0]))
        self.profit_factor = winning_profits / losing_profits if losing_profits > 0 else float('inf')
        
        # Expectancy
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        avg_win = np.mean([p for p in profits if p > 0]) if self.winning_trades > 0 else 0
        avg_loss = np.mean([p for p in profits if p < 0]) if self.losing_trades > 0 else 0
        self.expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

class BacktestingEngine:
    """FreqTrade-style backtesting engine"""
    
    def __init__(self, strategy, config: Dict[str, Any]):
        self.strategy = strategy
        self.config = config
        self.starting_balance = config.get('dry_run_wallet', 10000)
        self.fee = config.get('trading_fee', 0.001)
        self.max_open_trades = config.get('max_open_trades', 3)
        self.stake_amount = config.get('stake_amount', 100)
        
        # Portfolio tracking
        self.balance = self.starting_balance
        self.open_trades: List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []
        
    def run_backtest(self, data: pd.DataFrame, start_date: str, end_date: str) -> BacktestResult:
        """Run comprehensive backtest"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Filter data by date range
        data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
        
        if data.empty:
            logger.error("No data available for backtesting period")
            return BacktestResult()
        
        # Prepare strategy data
        metadata = {
            'pair': self.config.get('pair', 'BTC/USDT'),
            'timeframe': self.strategy.timeframe,
            'strategy': self.strategy.__class__.__name__
        }
        
        # Analyze data with strategy
        analyzed_data = self.strategy.analyze_ticker(data, metadata)
        
        # Process each candle
        for i in range(len(analyzed_data)):
            current_candle = analyzed_data.iloc[i]
            current_time = analyzed_data.index[i]
            
            # Check exit conditions for open trades
            self._check_exit_conditions(current_candle, current_time)
            
            # Check entry conditions
            self._check_entry_conditions(current_candle, current_time, metadata['pair'])
            
            # Apply time-based stops and ROI
            self._apply_roi_and_stoploss(current_candle, current_time)
        
        # Close any remaining open trades
        self._close_remaining_trades(analyzed_data.iloc[-1], analyzed_data.index[-1])
        
        # Generate results
        result = BacktestResult(trades=self.closed_trades)
        result.calculate_metrics()
        
        logger.info(f"Backtest completed: {result.total_trades} trades, "
                   f"{result.profit_total_ratio:.2%} total return")
        
        return result
    
    def _check_entry_conditions(self, candle: pd.Series, timestamp: datetime, pair: str):
        """Check for entry signals"""
        if len(self.open_trades) >= self.max_open_trades:
            return
            
        # Long entry
        if candle.get('enter_long', False):
            if self.balance >= self.stake_amount:
                trade = BacktestTrade(
                    pair=pair,
                    open_date=timestamp,
                    open_rate=candle['close'],  # Assume market order at close
                    amount=self.stake_amount / candle['close'],
                    fee_open=self.fee,
                    entry_tag=candle.get('entry_tag', 'long_entry'),
                    is_short=False
                )
                self.open_trades.append(trade)
                self.balance -= self.stake_amount
                logger.debug(f"Opened long position: {pair} at {candle['close']}")
        
        # Short entry (if strategy supports it)
        if self.strategy.can_short and candle.get('enter_short', False):
            if self.balance >= self.stake_amount:
                trade = BacktestTrade(
                    pair=pair,
                    open_date=timestamp,
                    open_rate=candle['close'],
                    amount=self.stake_amount / candle['close'],
                    fee_open=self.fee,
                    entry_tag=candle.get('entry_tag', 'short_entry'),
                    is_short=True
                )
                self.open_trades.append(trade)
                self.balance -= self.stake_amount
                logger.debug(f"Opened short position: {pair} at {candle['close']}")
    
    def _check_exit_conditions(self, candle: pd.Series, timestamp: datetime):
        """Check for exit signals"""
        trades_to_close = []
        
        for trade in self.open_trades:
            exit_reason = None
            
            # Check strategy exit conditions
            if not trade.is_short and candle.get('exit_long', False):
                exit_reason = 'exit_signal'
            elif trade.is_short and candle.get('exit_short', False):
                exit_reason = 'exit_signal'
                
            # Check custom exit
            if not exit_reason:
                custom_exit = self.strategy.custom_exit(
                    pair=trade.pair,
                    trade=trade.__dict__,
                    current_time=timestamp.isoformat(),
                    current_rate=candle['close'],
                    current_profit=self._calculate_current_profit(trade, candle['close'])
                )
                if custom_exit:
                    exit_reason = custom_exit
            
            if exit_reason:
                self._close_trade(trade, candle['close'], timestamp, exit_reason)
                trades_to_close.append(trade)
        
        # Remove closed trades
        for trade in trades_to_close:
            self.open_trades.remove(trade)
    
    def _apply_roi_and_stoploss(self, candle: pd.Series, timestamp: datetime):
        """Apply ROI and stoploss rules"""
        trades_to_close = []
        
        for trade in self.open_trades:
            current_profit = self._calculate_current_profit(trade, candle['close'])
            trade_duration = (timestamp - trade.open_date).total_seconds() / 60  # minutes
            
            # Check ROI
            roi_limit = self._get_roi_limit(trade_duration)
            if current_profit >= roi_limit:
                self._close_trade(trade, candle['close'], timestamp, 'roi')
                trades_to_close.append(trade)
                continue
            
            # Check stoploss
            stoploss_rate = self._get_stoploss_rate(trade, candle, timestamp, candle['close'], current_profit)
            if stoploss_rate and (
                (not trade.is_short and candle['close'] <= stoploss_rate) or
                (trade.is_short and candle['close'] >= stoploss_rate)
            ):
                self._close_trade(trade, candle['close'], timestamp, 'stoploss')
                trades_to_close.append(trade)
        
        for trade in trades_to_close:
            self.open_trades.remove(trade)
    
    def _calculate_current_profit(self, trade: BacktestTrade, current_rate: float) -> float:
        """Calculate current profit ratio for open trade"""
        if trade.is_short:
            return (trade.open_rate - current_rate) / trade.open_rate
        else:
            return (current_rate - trade.open_rate) / trade.open_rate
    
    def _get_roi_limit(self, trade_duration: float) -> float:
        """Get ROI limit based on trade duration"""
        roi_table = self.strategy.minimal_roi
        for duration_str, roi_value in sorted(roi_table.items(), key=lambda x: int(x[0])):
            if trade_duration >= int(duration_str):
                return roi_value
        return list(roi_table.values())[-1]  # Default to shortest duration ROI
    
    def _get_stoploss_rate(self, trade: BacktestTrade, candle: pd.Series, 
                          timestamp: datetime, current_rate: float, current_profit: float) -> Optional[float]:
        """Calculate stoploss rate"""
        # Check custom stoploss first
        custom_sl = self.strategy.custom_stoploss(
            pair=trade.pair,
            trade=trade.__dict__,
            current_time=timestamp.isoformat(),
            current_rate=current_rate,
            current_profit=current_profit
        )
        
        stoploss_ratio = custom_sl if custom_sl is not None else self.strategy.stoploss
        
        if trade.is_short:
            return trade.open_rate * (1 - stoploss_ratio)  # Inverted for short
        else:
            return trade.open_rate * (1 + stoploss_ratio)
    
    def _close_trade(self, trade: BacktestTrade, close_rate: float, 
                    close_time: datetime, exit_reason: str):
        """Close a trade and update balance"""
        trade.close_rate = close_rate
        trade.close_date = close_time
        trade.exit_reason = exit_reason
        trade.fee_close = self.fee
        trade.calculate_profit()
        
        # Update balance
        if trade.is_short:
            # For short trades, profit calculation is inverted
            trade_value = trade.amount * trade.open_rate
            profit = trade_value * trade.profit_ratio
            self.balance += trade_value + profit
        else:
            # For long trades
            trade_value = trade.amount * close_rate
            self.balance += trade_value
        
        self.closed_trades.append(trade)
        logger.debug(f"Closed {trade.pair} position: {trade.profit_ratio:.2%} profit")
    
    def _close_remaining_trades(self, final_candle: pd.Series, final_time: datetime):
        """Close any remaining open trades at the end of backtest"""
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            self._close_trade(trade, final_candle['close'], final_time, 'force_exit')
        self.open_trades.clear()

# === NEW FILE: core/hyperopt.py ===
import itertools
import random
from typing import Dict, Any, List, Callable, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

class HyperOptSpace:
    """Define hyperparameter optimization space"""
    
    @staticmethod
    def discrete(values: List[Any]) -> Dict[str, Any]:
        return {'type': 'discrete', 'values': values}
    
    @staticmethod
    def uniform(low: float, high: float) -> Dict[str, Any]:
        return {'type': 'uniform', 'low': low, 'high': high}
    
    @staticmethod
    def choice(options: List[Any]) -> Dict[str, Any]:
        return {'type': 'choice', 'options': options}

class StrategyOptimizer:
    """FreqTrade-style strategy optimization"""
    
    def __init__(self, strategy_class, backtesting_engine_class, base_config: Dict[str, Any]):
        self.strategy_class = strategy_class
        self.backtesting_engine_class = backtesting_engine_class
        self.base_config = base_config
        self.optimization_space = {}
    
    def add_parameter(self, name: str, space: Dict[str, Any]):
        """Add parameter to optimization space"""
        self.optimization_space[name] = space
    
    def generate_parameter_combinations(self, max_combinations: int = 100) -> List[Dict[str, Any]]:
        """Generate parameter combinations for testing"""
        if not self.optimization_space:
            return [{}]
        
        combinations = []
        
        # Generate all possible combinations (grid search)
        param_lists = []
        param_names = list(self.optimization_space.keys())
        
        for param_name in param_names:
            space = self.optimization_space[param_name]
            if space['type'] == 'discrete':
                param_lists.append(space['values'])
            elif space['type'] == 'choice':
                param_lists.append(space['options'])
            elif space['type'] == 'uniform':
                # For continuous variables, sample discrete points
                values = [space['low'] + (space['high'] - space['low']) * i / 9 for i in range(10)]
                param_lists.append(values)
        
        # Generate combinations
        for combo in itertools.product(*param_lists):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
            if len(combinations) >= max_combinations:
                break
        
        # If too many combinations, randomly sample
        if len(combinations) > max_combinations:
            combinations = random.sample(combinations, max_combinations)
        
        return combinations
    
    def optimize(self, data: pd.DataFrame, start_date: str, end_date: str, 
                max_combinations: int = 100, jobs: int = 1) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        logger.info(f"Starting optimization with {max_combinations} combinations")
        
        param_combinations = self.generate_parameter_combinations(max_combinations)
        
        if jobs == 1:
            # Single-threaded execution
            results = []
            for i, params in enumerate(param_combinations):
                logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
                result = self._evaluate_parameters(params, data, start_date, end_date)
                results.append((params, result))
        else:
            # Multi-threaded execution
            results = []
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(self._evaluate_parameters, params, data, start_date, end_date): params
                    for params in param_combinations
                }
                
                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        result = future.result()
                        results.append((params, result))
                        logger.info(f"Completed optimization for: {params}")
                    except Exception as e:
                        logger.error(f"Optimization failed for {params}: {e}")
        
        # Find best parameters
        best_params, best_result = max(results, key=lambda x: x[1].profit_total_ratio)
        
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best result: {best_result.profit_total_ratio:.2%} return, "
                   f"{best_result.total_trades} trades")
        
        return {
            'best_params': best_params,
            'best_result': best_result,
            'all_results': results
        }
    
    def _evaluate_parameters(self, params: Dict[str, Any], data: pd.DataFrame, 
                           start_date: str, end_date: str) -> 'BacktestResult':
        """Evaluate a single parameter combination"""
        # Create strategy with parameters
        config = self.base_config.copy()
        config.update(params)
        
        strategy = self.strategy_class(config)
        
        # Run backtest
        engine = self.backtesting_engine_class(strategy, config)
        result = engine.run_backtest(data, start_date, end_date)
        
        return result