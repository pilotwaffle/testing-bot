# strategies/strategy_template.py
import logging
from typing import Dict, Any
import pandas as pd
from strategies.strategy_base import StrategyBase, Signal, SignalType

logger = logging.getLogger(__name__)

class GenericBuyStrategy(StrategyBase):
    """
    A simple example strategy that always buys if no position is held.
    For testing purposes; not for actual trading.
    """

    DESCRIPTION = "A simple strategy for testing purposes. Buys if no position, sells if profit/loss is out of band."
    CONFIG_TEMPLATE = {
        "symbol": "BTC/USDT",
        "buy_confidence": 0.8,
        "sell_profit_percent": 0.01,
        "sell_loss_percent": -0.01,
        "allocation_percent": 0.05
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.symbol = config.get("symbol")
        self.buy_confidence = config.get("buy_confidence", 0.8)
        self.sell_profit_percent = config.get("sell_profit_percent", 0.01)
        self.sell_loss_percent = config.get("sell_loss_percent", -0.01)
        self.allocation_percent = config.get("allocation_percent", 0.05)
        # _engine_reference will be set by the trading engine

    def validate_config(self) -> bool:
        if not self.symbol:
            logger.error("GenericBuyStrategy: 'symbol' not found in configuration.")
            return False
        if not (0 < self.allocation_percent <= 1):
            logger.error("GenericBuyStrategy: 'allocation_percent' must be between 0 and 1.")
            return False
        return True

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # This simple strategy doesn't use complex indicators, so just return the original df.
        return df

    async def should_buy(self, market_data: Dict[str, Any], current_position: Dict[str, Any]) -> Signal:
        if current_position: # Already in a position
            return Signal(SignalType.NONE, 0, "Already in position.")
        
        # Check if we have enough balance to place an order
        current_balance = self._engine_reference.balances.get("USDT", 0) # Access balance from engine
        current_price = market_data.get("price")
        if not current_price:
            return Signal(SignalType.NONE, 0, "No current price for buy decision.")

        # Calculate potential quantity
        allocation_amount = current_balance * self.allocation_percent
        if allocation_amount <= 0:
            return Signal(SignalType.NONE, 0, "Insufficient allocation amount to buy.")

        quantity = allocation_amount / current_price
        if quantity <= 0:
            return Signal(SignalType.NONE, 0, "Calculated quantity is too small to buy.")

        logger.debug(f"GenericBuyStrategy: Considering BUY for {self.symbol}. Current balance: {current_balance:.2f} USDT.")
        return Signal(SignalType.BUY, self.buy_confidence, f"No position held. Attempting to buy {self.allocation_percent*100}%", quantity)

    async def should_sell(self, market_data: Dict[str, Any], current_position: Dict[str, Any]) -> Signal:
        if not current_position:
            return Signal(SignalType.NONE, 0, "No position to sell.")

        current_price = market_data.get("price")
        if not current_price:
            return Signal(SignalType.NONE, 0, "No current price for sell decision.")

        entry_price = current_position.get("entry_price", current_price)
        profit_loss_ratio = (current_price - entry_price) / entry_price

        if profit_loss_ratio >= self.sell_profit_percent:
            return Signal(SignalType.SELL, 0.95, f"Take profit triggered: ({profit_loss_ratio*100:.2f}%)", current_position.get("amount", 0))
        
        if profit_loss_ratio <= self.sell_loss_percent:
            return Signal(SignalType.SELL, 0.99, f"Stop loss triggered: ({profit_loss_ratio*100:.2f}%)", current_position.get("amount", 0))

        return Signal(SignalType.NONE, 0, "No sell condition met.")