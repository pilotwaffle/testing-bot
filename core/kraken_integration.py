# core/kraken_integration.py
# This file provides a direct CCXT-based integration for Kraken exchange operations.

import logging
import os
import ccxt # Import the CCXT library for exchange interaction
import asyncio # Used for running synchronous CCXT calls in an asynchronous context
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class KrakenIntegration:
    """
    Manages interaction with the Kraken exchange (Futures/Spot) using CCXT.
    This class is designed to be initialized with API key, secret, and sandbox status.
    """
    def __init__(self, api_key: str, secret: str, sandbox: bool = False):
        """
        Initializes the KrakenIntegration with API credentials and sandbox setting.

        Args:
            api_key (str): Your Kraken API key.
            secret (str): Your Kraken secret key.
            sandbox (bool): True to connect to the Kraken demo/sandbox environment, False for live.
        """
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox
        self.exchange = None # Will hold the CCXT exchange instance
        self.logger = logging.getLogger(__name__)

        # Immediately initialize the CCXT exchange upon object creation
        self._initialize_exchange()

        if self.exchange:
            self.logger.info("✅ Kraken Integration initialized successfully with CCXT.")
        else:
            self.logger.error("❌ Kraken Integration failed to initialize. Check logs for details.")

    def _initialize_exchange(self):
        """
        Internal method to set up the CCXT Kraken exchange instance.
        Handles API key validation and specific Kraken Futures/Spot settings.
        """
        if not self.api_key or not self.secret:
            self.logger.error("❌ Kraken API Key or Secret is missing. Cannot initialize exchange.")
            return

        try:
            # Configure CCXT for Kraken Futures. If you intend to use Kraken Spot,
            # you would use `ccxt.kraken({})` without the 'options' and 'hostname' for futures.
            self.exchange = ccxt.kraken({
                'apiKey': self.api_key,
                'secret': self.secret,
                'options': {
                    'defaultType': 'future', # Set to 'future' for Kraken Futures, or 'spot' for spot trading
                },
                # Use demo hostname for sandbox, live hostname for production
                'hostname': 'futures.kraken.com' if not self.sandbox else 'demo-futures.kraken.com',
                'enableRateLimit': True, # Enable CCXT's built-in rate limit handling for robustness
                'timeout': 30000, # Set a timeout for API requests (30 seconds)
            })

            self.logger.info(f"Kraken {'Sandbox' if self.sandbox else 'Live'} exchange instance created.")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Kraken exchange instance: {e}", exc_info=True)
            self.exchange = None # Ensure exchange is None if initialization fails

    async def fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        Asynchronously fetches the account balance from Kraken.
        Wraps the synchronous CCXT call in an executor to avoid blocking the event loop.
        """
        if not self.exchange:
            self.logger.warning("Kraken exchange not initialized. Cannot fetch balance.")
            return None
        try:
            # Use run_in_executor for synchronous CCXT methods within an async function
            loop = asyncio.get_event_loop()
            balance = await loop.run_in_executor(None, self.exchange.fetch_balance)
            self.logger.info("Fetched Kraken balance successfully.")
            return balance
        except Exception as e:
            self.logger.error(f"❌ Error fetching Kraken balance: {e}", exc_info=True)
            return None

    async def place_order(self, symbol: str, type: str, side: str, amount: float, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Asynchronously places a trade order on Kraken.
        Wraps the synchronous CCXT call in an executor.

        Args:
            symbol (str): Trading pair (e.g., 'XBT/USD' for Bitcoin futures, 'BTC/USDT' for spot).
            type (str): Order type ('limit', 'market', etc.).
            side (str): Order side ('buy' or 'sell').
            amount (float): The amount of currency to trade.
            price (Optional[float]): The price for 'limit' orders. Required for limit orders.
        """
        if not self.exchange:
            self.logger.warning("Kraken exchange not initialized. Cannot place order.")
            return None
        try:
            loop = asyncio.get_event_loop()
            # Use lambda to pass arguments to the synchronous create_order method
            order = await loop.run_in_executor(None, lambda: self.exchange.create_order(symbol, type, side, amount, price))
            self.logger.info(f"Placed Kraken order: {order.get('id', 'N/A')} for {amount} {symbol}")
            return order
        except Exception as e:
            self.logger.error(f"❌ Error placing Kraken order for {symbol}: {e}", exc_info=True)
            return None

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Asynchronously cancels an order on Kraken.

        Args:
            order_id (str): The ID of the order to cancel.
            symbol (Optional[str]): The symbol of the order (sometimes required by exchanges).
        """
        if not self.exchange:
            self.logger.warning("Kraken exchange not initialized. Cannot cancel order.")
            return None
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.exchange.cancel_order(order_id, symbol))
            self.logger.info(f"Canceled Kraken order: {order_id}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error canceling Kraken order {order_id}: {e}", exc_info=True)
            return None

    async def get_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the Kraken integration.
        """
        return {
            "initialized": self.exchange is not None,
            "connected_to_sandbox": self.sandbox,
            "exchange_name": "Kraken",
            "api_key_configured": bool(self.api_key),
            # This assumes that if initialized, it supports futures if defaultType is 'future'
            "supports_futures": True if self.exchange and self.exchange.options.get('defaultType') == 'future' else False
        }

if __name__ == "__main__":
    # This block allows you to test the KrakenIntegration class directly
    # from your core directory without running the full FastAPI app.
    from dotenv import load_dotenv
    # Load environment variables from .env file (ensure it's in the project root)
    load_dotenv()

    # Get API keys and sandbox setting from environment variables
    # Replace "YOUR_KRAKEN_API_KEY" and "YOUR_KRAKEN_SECRET" with actual values in .env
    test_api_key = os.getenv("KRAKEN_API_KEY")
    test_secret = os.getenv("KRAKEN_SECRET")
    test_sandbox = os.getenv("KRAKEN_SANDBOX", "false").lower() == "true"

    if not test_api_key or not test_secret:
        print("\n⚠️ Please configure KRAKEN_API_KEY and KRAKEN_SECRET in your .env file for testing.")
        print("   Optionally set KRAKEN_SANDBOX=true for demo environment testing.")
    else:
        print(f"\n🚀 Initializing KrakenIntegration for direct test (Sandbox: {test_sandbox})...")
        kraken_client = KrakenIntegration(test_api_key, test_secret, sandbox=test_sandbox)
        
        async def run_test_kraken_integration():
            print("\n--- Getting Status ---")
            status = await kraken_client.get_status()
            print(f"Kraken Integration Status: {status}")

            if status["initialized"]:
                print("\n--- Fetching Balance ---")
                balance = await kraken_client.fetch_balance()
                if balance:
                    print(f"Kraken Balance: Total: {balance.get('total')}, Free: {balance.get('free')}")
                else:
                    print("Could not fetch balance.")

                # Example: Place a dummy order (CAUTION: Ensure this is on sandbox or a very small amount!)
                # For live testing, adjust symbol, amount, and price carefully.
                # print("\n--- Placing Test Order (Limit Buy BTC/USD) ---")
                # try:
                #     # Adjust symbol (e.g., 'PI_XBTUSD' for futures, 'BTC/USD' for spot)
                #     # Adjust amount and price for testing
                #     test_order = await kraken_client.place_order("PI_XBTUSD", "limit", "buy", 0.0001, 60000.0)
                #     print(f"Test Order Result: {test_order}")
                # except Exception as e:
                #     print(f"Failed to place test order: {e}")

                # print("\n--- Testing Cancel Order (if you placed one) ---")
                # # Replace 'your_order_id_here' with an actual order ID from your sandbox/live account
                # # cancel_result = await kraken_client.cancel_order("your_order_id_here")
                # # print(f"Cancel Order Result: {cancel_result}")

            else:
                print("\nIntegration not initialized. Please check API keys and connection.")
        
        # Run the asynchronous test function
        asyncio.run(run_test_kraken_integration())
        print("\n✅ Kraken Integration direct test complete.")