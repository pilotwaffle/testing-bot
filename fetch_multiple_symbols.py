# fetch_multiple_symbols.py

import asyncio
import logging
import pandas as pd
from core.data_fetcher import CryptoDataFetcher
from core.config import settings # Needed to ensure configuration is loaded

# Set up basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_multiple_symbols_data():
    """
    Fetches 90 days of historical data for a list of cryptocurrencies.
    """
    # Ensure settings are loaded (e.g., from .env)
    _ = settings

    logger.info("Initializing CryptoDataFetcher...")
    data_fetcher = CryptoDataFetcher()

    # List of symbols to fetch
    symbols_to_fetch = [
        "BTC/USDT",
        "ETH/USDT",
        "ADA/USDT",
        "XRP/USDT",
        "DOT/USDT",
        "LINK/USDT",
    ]

    # --- Configuration for historical data ---
    timeframe = "1d"  # '1d' for daily candles
    limit = 90        # 90 candles for 90 days of daily data

    # If you want 90 days of HOURLY data instead:
    # timeframe = "1h"
    # limit = 90 * 24 # 90 days * 24 hours/day = 2160 hourly candles

    logger.info(f"Attempting to fetch {limit} {timeframe} candles for {len(symbols_to_fetch)} symbols.")

    all_fetched_data = {} # Dictionary to store DataFrames for each symbol

    for symbol in symbols_to_fetch:
        logger.info(f"\n--- Fetching data for {symbol} ---")
        try:
            df: pd.DataFrame = await data_fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} candles for {symbol}.")
                all_fetched_data[symbol] = df
                # Optional: Print first/last few rows for confirmation
                # logger.info(f"First 3 rows for {symbol}:\n{df.head(3)}")
                # logger.info(f"Last 3 rows for {symbol}:\n{df.tail(3)}")
            else:
                logger.warning(f"No data fetched for {symbol}. Moving to next symbol.")

        except Exception as e:
            logger.error(f"An error occurred while fetching data for {symbol}: {e}", exc_info=True)
            logger.warning(f"Failed to fetch data for {symbol}. Skipping this symbol.")

    logger.info("\n--- Data Fetching Summary ---")
    if all_fetched_data:
        logger.info(f"Successfully fetched data for {len(all_fetched_data)} out of {len(symbols_to_fetch)} symbols.")
        for symbol, df_data in all_fetched_data.items():
            logger.info(f"- {symbol}: {len(df_data)} candles (stored in 'all_fetched_data[\"{symbol}\"]' DataFrame)")
            # You can now use all_fetched_data dictionary for further processing,
            # e.g., passing it to your ML models or strategy backtesting.
    else:
        logger.warning("No data was successfully fetched for any symbol.")

if __name__ == "__main__":
    asyncio.run(fetch_multiple_symbols_data())