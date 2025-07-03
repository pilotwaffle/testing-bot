import logging
from typing import Optional

logger = logging.getLogger(__name__)

class GasTracker:
    """
    Tracks the balance and usage of a blockchain gas token (e.g. ETH for Ethereum).
    You can extend this to also record gas spent per transaction if integrating with web3.py or similar.
    """

    def __init__(self, initial_balance: float = 0.0, symbol: str = "GAS"):
        self.symbol = symbol.upper()
        self.free = initial_balance
        self.used = 0.0
        self.total = initial_balance

    def add_gas(self, amount: float):
        """Add gas tokens to the free balance."""
        logger.info(f"Adding {amount} {self.symbol} to balance.")
        self.free += amount
        self.total += amount

    def use_gas(self, amount: float):
        """
        Deduct gas tokens from the free balance and add to used.
        """
        if amount > self.free:
            logger.warning(f"Insufficient {self.symbol} for this operation. Attempted to use {amount}, available: {self.free}")
            amount = self.free  # Use up what's left

        self.free -= amount
        self.used += amount
        logger.info(f"Used {amount} {self.symbol} for transaction. Remaining free: {self.free}")

    def reset_used(self):
        """Reset used gas (e.g., after a settlement)."""
        logger.info(f"Resetting used {self.symbol} counter.")
        self.used = 0.0

    def get_free(self) -> float:
        return self.free

    def get_used(self) -> float:
        return self.used

    def get_total(self) -> float:
        return self.total

    def __repr__(self):
        return (
            f"<GasTracker(symbol={self.symbol}, free={self.free}, used={self.used}, total={self.total})>"
        )

# Example usage
if __name__ == "__main__":
    gt = GasTracker(initial_balance=1.0, symbol="ETH")
    print(gt)
    gt.add_gas(0.5)
    gt.use_gas(0.3)
    print("Free gas:", gt.get_free())
    print("Used gas:", gt.get_used())
    print("Total gas:", gt.get_total())
    gt.reset_used()
    print(gt)