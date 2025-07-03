"""
Risk Management System for Crypto Trading Bot

Handles position sizing, risk controls, and portfolio risk management.
Improvements:
- Dependency injection for database (testability)
- Async compatibility for future-proofing
- Configurable parameters as dataclass
- Logging & error improvements
- Clean separation between calculation, validation, and reporting
- Ready for extension/customization
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum

from database import get_database, Trade, Position

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.10
    max_position_size: float = 0.20
    max_correlation: float = 0.7
    max_drawdown_limit: float = 0.15
    min_risk_reward: float = 1.5
    max_daily_trades: int = 10
    max_daily_loss: float = 0.05
    circuit_breaker_loss: float = 0.20
    volatility_threshold: float = 0.05

@dataclass
class RiskMetrics:
    var_1d: float
    var_7d: float
    max_drawdown: float
    sharpe_ratio: float
    portfolio_volatility: float
    correlation_risk: float
    concentration_risk: float
    leverage_ratio: float
    risk_level: RiskLevel

@dataclass
class PositionRisk:
    symbol: str
    position_size: float
    risk_amount: float
    risk_percentage: float
    stop_loss_price: float
    take_profit_price: float
    max_loss: float
    risk_reward_ratio: float

@dataclass
class TradeDecision:
    allowed: bool
    reason: str
    suggested_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

class RiskManager:
    """
    Comprehensive risk management system for crypto trading.
    """
    def __init__(
        self,
        initial_balance: float = 10000.0,
        config: RiskConfig = RiskConfig(),
        db_factory: Callable[[], Any] = get_database
    ):
        self.database = db_factory()
        self.initial_balance = initial_balance
        self.config = config
        logger.info("Risk Manager initialized")

    def get_current_balance(self) -> float:
        # Placeholder: replace with actual logic to fetch balance from database or broker
        return self.initial_balance

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, risk_amount: float = None) -> float:
        try:
            current_balance = self.get_current_balance()
            risk_amount = risk_amount if risk_amount is not None else current_balance * self.config.max_risk_per_trade
            price_risk = abs(entry_price - stop_loss)
            if price_risk == 0:
                logger.warning("Invalid stop loss - no risk per unit")
                return 0.0
            position_size = risk_amount / price_risk
            max_position_value = current_balance * self.config.max_position_size
            max_size_by_limit = max_position_value / entry_price
            position_size = min(position_size, max_size_by_limit)
            min_trade_value = 10.0
            min_size = min_trade_value / entry_price
            if position_size < min_size:
                logger.info(f"Position size too small: {position_size:.6f}, minimum: {min_size:.6f}")
                return 0.0
            logger.info(f"Calculated position size for {symbol}: {position_size:.6f} units")
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def validate_trade(self, symbol: str, side: str, quantity: float, price: float, stop_loss: float = None, take_profit: float = None) -> TradeDecision:
        try:
            if self.is_circuit_breaker_triggered():
                return TradeDecision(False, "Circuit breaker triggered - excessive losses")
            daily_check = self.check_daily_limits()
            if not daily_check['allowed']:
                return TradeDecision(False, daily_check['reason'])
            concentration_check = self.check_position_concentration(symbol, quantity, price)
            if not concentration_check['allowed']:
                return TradeDecision(False, concentration_check['reason'])
            if stop_loss:
                risk_reward_check = self.validate_risk_reward(price, stop_loss, take_profit, side)
                if not risk_reward_check['allowed']:
                    return TradeDecision(False, risk_reward_check['reason'])
            correlation_check = self.check_correlation_risk(symbol)
            if not correlation_check['allowed']:
                return TradeDecision(False, correlation_check['reason'])
            volatility_check = self.check_market_volatility(symbol)
            if not volatility_check['allowed']:
                return TradeDecision(False, volatility_check['reason'])
            if not stop_loss:
                stop_loss = price * (0.98 if side == 'buy' else 1.02)
            optimal_size = self.calculate_position_size(symbol, price, stop_loss)
            if quantity > optimal_size * 1.1:
                return TradeDecision(True, "Position size too large, suggesting optimal size", optimal_size, stop_loss, take_profit)
            return TradeDecision(True, "Trade approved", quantity, stop_loss, take_profit)
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return TradeDecision(False, f"Validation error: {e}")

    def calculate_portfolio_risk(self) -> RiskMetrics:
        try:
            positions = self.database.get_positions()
            current_balance = self.get_current_balance()
            if not positions:
                return RiskMetrics(0,0,0,0,0,0,0,0,RiskLevel.LOW)
            total_position_value = sum(pos.quantity * pos.current_price for pos in positions)
            portfolio_value = current_balance + total_position_value
            weights = [(pos.quantity * pos.current_price) / portfolio_value for pos in positions]
            concentration_risk = max(weights) if weights else 0.0
            returns_data = self.get_portfolio_returns(days=30)
            portfolio_volatility = np.std(returns_data) if len(returns_data) > 1 else 0.0
            if len(returns_data) > 1:
                var_1d = np.percentile(returns_data, 5) * portfolio_value
                var_7d = var_1d * np.sqrt(7)
            else:
                var_1d = var_7d = 0.0
            max_drawdown = self.calculate_max_drawdown(days=90)
            if portfolio_volatility > 0:
                avg_return = np.mean(returns_data) if len(returns_data) > 1 else 0.0
                sharpe_ratio = avg_return / portfolio_volatility
            else:
                sharpe_ratio = 0.0
            leverage_ratio = total_position_value / current_balance if current_balance > 0 else 0.0
            correlation_risk = self.calculate_correlation_risk()
            risk_level = self.determine_risk_level(concentration_risk, portfolio_volatility, max_drawdown, leverage_ratio)
            return RiskMetrics(
                var_1d=var_1d,
                var_7d=var_7d,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                portfolio_volatility=portfolio_volatility,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                risk_level=risk_level
            )
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return RiskMetrics(0,0,0,0,0,0,0,0,RiskLevel.CRITICAL)

    def check_position_concentration(self, symbol: str, quantity: float, price: float) -> Dict[str, Any]:
        """Check position concentration risk"""
        try:
            current_balance = self.get_current_balance()
            position_value = quantity * price
            concentration = position_value / current_balance
            if concentration > self.config.max_position_size:
                return {
                    'allowed': False,
                    'reason': f"Position size too large: {concentration:.2%} > {self.config.max_position_size:.2%}"
                }
            return {'allowed': True, 'reason': 'Position size OK'}
        except Exception as e:
            logger.error(f"Error checking position concentration: {e}")
            return {'allowed': False, 'reason': 'Error checking concentration'}

    # --- Stub implementations for illustration ---
    def is_circuit_breaker_triggered(self) -> bool:
        # Placeholder: implement actual circuit breaker logic
        return False

    def check_daily_limits(self) -> Dict[str, Any]:
        # Placeholder: implement daily trade and loss limits
        return {'allowed': True, 'reason': 'Daily limits OK'}

    def validate_risk_reward(self, price, stop_loss, take_profit, side) -> Dict[str, Any]:
        # Placeholder: implement actual risk/reward validation
        return {'allowed': True, 'reason': 'Risk/reward OK'}

    def check_correlation_risk(self, symbol: str) -> Dict[str, Any]:
        # Placeholder: implement actual correlation risk check
        return {'allowed': True, 'reason': 'Correlation OK'}

    def check_market_volatility(self, symbol: str) -> Dict[str, Any]:
        # Placeholder: implement volatility check
        return {'allowed': True, 'reason': 'Volatility OK'}

    def get_portfolio_returns(self, days: int = 30) -> List[float]:
        # Placeholder: return simulated returns
        return [0.01, -0.005, 0.002, 0.006, -0.003] * (days // 5)

    def calculate_max_drawdown(self, days: int = 90) -> float:
        # Placeholder: mock drawdown
        return 0.1

    def calculate_correlation_risk(self) -> float:
        # Placeholder: mock correlation
        return 0.2

    def determine_risk_level(self, concentration, volatility, drawdown, leverage) -> RiskLevel:
        # Placeholder: threshold logic
        if drawdown > self.config.max_drawdown_limit or leverage > 2.0:
            return RiskLevel.CRITICAL
        if concentration > self.config.max_position_size or volatility > self.config.volatility_threshold:
            return RiskLevel.HIGH
        return RiskLevel.LOW

    def get_risk_summary(self) -> Dict[str, Any]:
        # Placeholder: return summary
        return {"summary": "Risk summary not implemented."}

# Singleton for global usage
_risk_manager = None

def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager

class EnhancedRiskManager(RiskManager):
    """
    Enhanced risk manager with additional or custom logic.
    Extend this class to implement more sophisticated risk controls.
    """
    def __init__(self, *args, trailing_stop_atr_mult: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.trailing_stop_atr_mult = trailing_stop_atr_mult

    def get_trailing_stop(self, entry_price: float, atr: float, is_long: bool = True) -> float:
        """ATR-based trailing stop calculation."""
        if is_long:
            return entry_price - atr * self.trailing_stop_atr_mult
        else:
            return entry_price + atr * self.trailing_stop_atr_mult

if __name__ == "__main__":
    rm = get_risk_manager()
    position_size = rm.calculate_position_size(symbol="BTC/USDT", entry_price=50000, stop_loss=48000)
    print(f"Calculated position size: {position_size}")
    decision = rm.validate_trade(symbol="BTC/USDT", side="buy", quantity=0.1, price=50000, stop_loss=48000, take_profit=54000)
    print(f"Trade decision: {decision}")
    summary = rm.get_risk_summary()
    print(f"Risk summary: {summary}")