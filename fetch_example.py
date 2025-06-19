# fetch_example.py

import asyncio
import logging
import pandas as pd
from core.data_fetcher import CryptoDataFetcher
from core.config import settings # Needed to ensure configuration is loaded

# Set up basic logging (optional, but good practice for standalone scripts)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main_fetch_data():
    """
    Main asynchronous function to demonstrate fetching historical data.
    """
    # Ensure settings are loaded from .env or config files.
    # This implicit access will trigger dotenv loading if used by core.config.
    _ = settings

    logger.info("Initializing CryptoDataFetcher...")
    data_fetcher = CryptoDataFetcher()

    # --- Parameters for fetching ---
    symbol_to_fetch = "BTC/USDT"
    timeframe_to_fetch = "1h" # e.g., '1m', '5m', '15m', '1h', '4h', '1d'
    limit_of_candles = 500   # Number of historical candles to fetch

    logger.info(f"Attempting to fetch {limit_of_candles} {timeframe_to_fetch} candles for {symbol_to_fetch}...")
    try:
        df: pd.DataFrame = await data_fetcher.fetch_ohlcv(
            symbol=symbol_to_fetch,
            timeframe=timeframe_to_fetch,
            limit=limit_of_candles
        )

        if not df.empty:
            logger.info(f"Successfully fetched {len(df)} candles for {symbol_to_fetch}.")
            logger.info("First 5 rows of fetched data:")
            print(df.head())
            logger.info("\nLast 5 rows of fetched data:")
            print(df.tail())
            logger.info(f"\nDataFrame Info:\n{df.info()}")
        else:
            logger.warning(f"No data fetched for {symbol_to_fetch}. Check settings, connectivity, or specified parameters.")

    except Exception as e:
        logger.error(f"An error occurred while fetching data: {e}", exc_info=True)

if __name__ == "__main__":
    # Run the asynchronous main function
    asyncio.run(main_fetch_data())