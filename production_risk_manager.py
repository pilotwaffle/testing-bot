#!/usr/bin/env python3
"""
Production Risk Management System for Trading Bot
FILE LOCATION: E:\Trade Chat Bot\G Trading Bot\production_risk_manager.py

USAGE:
1. Save this as production_risk_manager.py in your trading bot directory
2. Import into your main trading bot: from production_risk_manager import ProductionTradingEngine
3. Integrate with your enhanced model predictions for live trading

FEATURES:
✅ Kelly Criterion position sizing
✅ Dynamic stop-loss and take-profit levels
✅ Portfolio risk limits and emergency exits
✅ Real-time signal filtering
✅ Professional risk management for live trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

class ProductionRiskManager:
    """Professional risk management for live trading"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk per trade
        self.max_position_size = 0.10   # 10% max position size
        self.max_daily_loss = 0.05      # 5% max daily loss
        self.max_drawdown = 0.15        # 15% max drawdown before stopping
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, signal_confidence: float, volatility: float, 
                              entry_price: float, stop_loss_price: float) -> float:
        """Calculate optimal position size using Kelly Criterion + risk management"""
        
        # Kelly Criterion-based sizing
        win_rate = signal_confidence
        avg_win = 0.025  # 2.5% average win (adjust based on backtesting)
        avg_loss = abs(entry_price - stop_loss_price) / entry_price
        
        # Kelly fraction
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Risk-based position sizing
        risk_per_trade = abs(entry_price - stop_loss_price) / entry_price
        max_risk_amount = self.current_capital * self.max_portfolio_risk
        risk_based_size = max_risk_amount / (risk_per_trade * entry_price)
        
        # Volatility adjustment
        volatility_adjustment = 1 / (1 + volatility * 10)  # Reduce size in high volatility
        
        # Final position size
        position_value = min(
            kelly_fraction * self.current_capital,
            risk_based_size * entry_price,
            self.max_position_size * self.current_capital
        ) * volatility_adjustment
        
        position_size = position_value / entry_price
        
        self.logger.info(f"Position sizing: Kelly={kelly_fraction:.3f}, "
                        f"Risk-based=${risk_based_size:.2f}, "
                        f"Vol-adj={volatility_adjustment:.3f}, "
                        f"Final size={position_size:.4f}")
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, direction: str, 
                           atr: float, signal_confidence: float) -> float:
        """Calculate dynamic stop loss based on ATR and confidence"""
        
        # Base stop loss: 2x ATR for trend trades, 1.5x for mean reversion
        base_multiplier = 2.0 if signal_confidence > 0.7 else 1.5
        
        # Adjust based on confidence
        confidence_multiplier = 1.5 - signal_confidence  # Higher confidence = tighter stops
        
        stop_distance = atr * base_multiplier * confidence_multiplier
        
        if direction == 'long':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                            direction: str, signal_confidence: float) -> List[float]:
        """Calculate multiple take profit levels"""
        
        risk = abs(entry_price - stop_loss)
        
        # Risk-reward ratios based on confidence
        if signal_confidence > 0.75:
            reward_ratios = [1.5, 2.5, 4.0]  # High confidence: let winners run
        elif signal_confidence > 0.65:
            reward_ratios = [1.2, 2.0, 3.0]  # Medium confidence
        else:
            reward_ratios = [1.0, 1.5, 2.0]  # Low confidence: take profits early
        
        take_profits = []
        for ratio in reward_ratios:
            if direction == 'long':
                tp = entry_price + (risk * ratio)
            else:
                tp = entry_price - (risk * ratio)
            take_profits.append(tp)
        
        return take_profits
    
    def check_portfolio_limits(self) -> Dict[str, bool]:
        """Check if portfolio is within risk limits"""
        
        total_portfolio_value = self.calculate_portfolio_value()
        
        # Calculate current drawdown
        peak_value = max(self.current_capital, total_portfolio_value)
        current_drawdown = (peak_value - total_portfolio_value) / peak_value
        
        # Calculate daily P&L
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_trades = [t for t in self.trade_history if t['timestamp'] >= today_start]
        daily_pnl = sum(t['pnl'] for t in daily_trades)
        daily_loss_pct = abs(daily_pnl) / self.current_capital if daily_pnl < 0 else 0
        
        limits = {
            'max_drawdown_ok': current_drawdown < self.max_drawdown,
            'daily_loss_ok': daily_loss_pct < self.max_daily_loss,
            'position_count_ok': len(self.positions) < 10,  # Max 10 concurrent positions
            'capital_positive': total_portfolio_value > 0
        }
        
        if not all(limits.values()):
            self.logger.warning(f"Portfolio limits breached: {limits}")
            self.logger.warning(f"Drawdown: {current_drawdown:.2%}, Daily loss: {daily_loss_pct:.2%}")
        
        return limits
    
    def calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        # This would fetch real-time prices in production
        position_values = sum(pos['quantity'] * pos['current_price'] for pos in self.positions.values())
        return self.current_capital + position_values
    
    def emergency_exit_all(self, reason: str):
        """Emergency exit all positions"""
        self.logger.critical(f"EMERGENCY EXIT TRIGGERED: {reason}")
        
        # In production, this would place market orders to close all positions
        for symbol, position in self.positions.items():
            self.logger.critical(f"Closing position: {symbol}, Size: {position['quantity']}")
            # close_position(symbol, position['quantity'])
        
        self.positions.clear()

class ProductionSignalFilter:
    """Filter trading signals for production deployment"""
    
    def __init__(self):
        self.min_confidence = 0.65      # Minimum model confidence
        self.min_volume_ratio = 1.2     # Minimum volume vs average
        self.max_spread_bps = 20        # Maximum bid-ask spread in basis points
        self.market_hours_only = True   # Trade only during market hours
        
    def filter_signal(self, signal: Dict) -> Tuple[bool, str]:
        """Filter signal for production trading"""
        
        # Confidence filter
        if signal['confidence'] < self.min_confidence:
            return False, f"Low confidence: {signal['confidence']:.3f} < {self.min_confidence}"
        
        # Volume filter
        if signal['volume_ratio'] < self.min_volume_ratio:
            return False, f"Low volume: {signal['volume_ratio']:.2f} < {self.min_volume_ratio}"
        
        # Spread filter (liquidity check)
        spread_bps = (signal['ask'] - signal['bid']) / signal['mid'] * 10000
        if spread_bps > self.max_spread_bps:
            return False, f"Wide spread: {spread_bps:.1f}bps > {self.max_spread_bps}bps"
        
        # Market hours filter
        if self.market_hours_only and not self.is_market_hours():
            return False, "Outside market hours"
        
        # Volatility filter (avoid trading in extreme volatility)
        if signal['volatility'] > signal['avg_volatility'] * 3:
            return False, f"Extreme volatility: {signal['volatility']:.4f}"
        
        return True, "Signal passed all filters"
    
    def is_market_hours(self) -> bool:
        """Check if it's market hours (24/7 for crypto, business hours for stocks)"""
        # For crypto: always True
        # For stocks: check business hours
        return True  # Crypto markets are 24/7

class ProductionTradingEngine:
    """Complete production trading engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.risk_manager = ProductionRiskManager(initial_capital)
        self.signal_filter = ProductionSignalFilter()
        self.logger = logging.getLogger(__name__)
        
    def process_trading_signal(self, symbol: str, signal: Dict, market_data: Dict) -> Optional[Dict]:
        """Process a trading signal for production execution"""
        
        # 1. Filter signal
        is_valid, filter_reason = self.signal_filter.filter_signal(signal)
        if not is_valid:
            self.logger.info(f"Signal filtered out: {filter_reason}")
            return None
        
        # 2. Check portfolio limits
        limits = self.risk_manager.check_portfolio_limits()
        if not all(limits.values()):
            self.logger.warning(f"Portfolio limits prevent trading: {limits}")
            return None
        
        # 3. Calculate trade parameters
        entry_price = market_data['current_price']
        direction = 'long' if signal['direction'] > 0 else 'short'
        
        # Stop loss
        stop_loss = self.risk_manager.calculate_stop_loss(
            entry_price, direction, market_data['atr'], signal['confidence']
        )
        
        # Position size
        position_size = self.risk_manager.calculate_position_size(
            signal['confidence'], market_data['volatility'], entry_price, stop_loss
        )
        
        # Take profits
        take_profits = self.risk_manager.calculate_take_profit(
            entry_price, stop_loss, direction, signal['confidence']
        )
        
        # 4. Create trade order
        trade_order = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'confidence': signal['confidence'],
            'timestamp': datetime.now(),
            'order_type': 'market',  # or 'limit' for better fills
            'reason': f"Model confidence: {signal['confidence']:.3f}"
        }
        
        self.logger.info(f"Trade order created: {trade_order}")
        return trade_order
    
    def execute_trade(self, trade_order: Dict) -> bool:
        """Execute trade order (placeholder for actual broker integration)"""
        
        # In production, this would:
        # 1. Place entry order with broker/exchange
        # 2. Set stop loss order
        # 3. Set take profit orders
        # 4. Monitor order fills
        # 5. Update position tracking
        
        self.logger.info(f"Executing trade: {trade_order['symbol']} "
                        f"{trade_order['direction']} {trade_order['position_size']:.4f} "
                        f"@ {trade_order['entry_price']:.2f}")
        
        # Simulate successful execution
        return True

# Usage example:
def production_trading_example():
    """Example of how to use the production system with your enhanced trading bot"""
    
    trading_engine = ProductionTradingEngine(initial_capital=10000)
    
    # Simulate a trading signal from your enhanced model
    signal = {
        'confidence': 0.72,        # Model prediction confidence
        'direction': 1,            # 1 for long, -1 for short
        'volume_ratio': 1.5,       # Current volume vs average
        'ask': 61500,             # Current ask price
        'bid': 61480,             # Current bid price
        'mid': 61490,             # Mid price
        'volatility': 0.025,      # Current volatility
        'avg_volatility': 0.030   # Average volatility
    }
    
    # Market data
    market_data = {
        'current_price': 61490,
        'atr': 1200,              # Average True Range
        'volatility': 0.025
    }
    
    # Process signal
    trade_order = trading_engine.process_trading_signal('BTC/USD', signal, market_data)
    
    if trade_order:
        # Execute if valid
        success = trading_engine.execute_trade(trade_order)
        if success:
            print("Trade executed successfully!")
        else:
            print("Trade execution failed!")
    else:
        print("Signal filtered out or limits breached")

def integrate_with_enhanced_bot():
    """Example integration with your existing enhanced trading bot"""
    
    # Import your enhanced model trainer
    # from optimized_model_trainer import OptimizedModelTrainer
    
    trading_engine = ProductionTradingEngine(initial_capital=10000)
    
    print("🚀 Production Trading Bot Integration Example")
    print("=" * 60)
    print("📁 File: E:\\Trade Chat Bot\\G Trading Bot\\production_risk_manager.py")
    print("🔗 Integrates with: optimized_model_trainer.py")
    print("🎯 Purpose: Live trading with professional risk management")
    print()
    print("📋 Integration Steps:")
    print("1. Save this file as production_risk_manager.py")
    print("2. Import: from production_risk_manager import ProductionTradingEngine")
    print("3. Use trading_engine.process_trading_signal() with your model predictions")
    print("4. Paper trade first, then deploy with real capital")
    print()
    print("⚠️  IMPORTANT: Test thoroughly in paper trading before live deployment!")

if __name__ == "__main__":
    print("🎯 Production Risk Management System")
    print("📁 Save as: E:\\Trade Chat Bot\\G Trading Bot\\production_risk_manager.py")
    print()
    production_trading_example()
    print()
    integrate_with_enhanced_bot()