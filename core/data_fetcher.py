# core/data_fetcher.py
import asyncio
import logging
import random
from datetime import datetime
from typing import Dict, Any, Callable, Awaitable, TYPE_CHECKING, Optional
import numpy as np
import pandas as pd
import os 

# Import ccxt exceptions for specific error handling
import ccxt
import ccxt.pro # Ensure ccxt.pro is imported for async operations

from core.config import settings

logger = logging.getLogger(__name__)

# --- Global Flags for CCXT Library Availability ---
_CCXT_PRO_AVAILABLE_GLOBAL = False
_CCXT_SYNC_AVAILABLE_GLOBAL = False

try:
    import ccxt
    _CCXT_SYNC_AVAILABLE_GLOBAL = True
    logger.info("CCXT (synchronous) library loaded.")
except ImportError:
    logger.warning("CCXT (synchronous) library not found. Historical data fetches might be limited.")
    
try:
    import ccxt.pro
    _CCXT_PRO_AVAILABLE_GLOBAL = True
    logger.info("CCXT Pro (asynchronous) library loaded.")
except ImportError:
    logger.warning("CCXT Pro (asynchronous) library not found. Real-time data feed will use demo data.")

# For static type checkers, to acknowledge the dynamically set attributes
if TYPE_CHECKING:
    import builtins
    builtins.CCXT_PRO_AVAILABLE = _CCXT_PRO_AVAILABLE_GLOBAL
    builtins.CCXT_SYNC_AVAILABLE = _CCXT_SYNC_AVAILABLE_GLOBAL


class CryptoDataFetcher:
    """Fetches real-time and historical crypto data from exchanges."""
    def __init__(self):
        self._crypto_ex_sync: Optional[ccxt.Exchange] = None
        self._crypto_ex_async: Optional[ccxt.pro.Exchange] = None
        self.running_feed = False

        # Flags to track successful market loading for sync/async clients
        self._markets_loaded_async = False 
        self._markets_loaded_sync = False # Corrected: Flag for synchronous client markets

        # Symbols to fetch for the real-time feed (from settings)
        self.symbols = settings.DEFAULT_TRAINING_SYMBOLS
        self.exchange_id = settings.DEFAULT_EXCHANGE

        # CACHING SETUP (Ensures data/ohlcv_cache directory exists)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
        self.cache_dir = os.path.join(project_root_dir, 'data', 'ohlcv_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"OHLCV data cache directory: {self.cache_dir}")

        # --- Initialize synchronous CCXT client for historical data ---
        if _CCXT_SYNC_AVAILABLE_GLOBAL:
            try:
                sync_exchange_class = getattr(ccxt, self.exchange_id, None)
                if sync_exchange_class:
                    self._crypto_ex_sync = sync_exchange_class({
                        'enableRateLimit': True, 
                        'timeout': 15000,        
                    })
                    logger.info(f"Initialized CCXT (sync) {self.exchange_id} for historical data.")
                else:
                    logger.warning(f"CCXT (sync) exchange '{self.exchange_id}' not found. "
                                   "Historical data fetching may fall back to demo.")
            except Exception as e:
                logger.error(f"Failed to instantiate CCXT (sync) client: {e}.", exc_info=True)
                self._crypto_ex_sync = None


        # --- Initialize asynchronous CCXT Pro client for real-time feed ---
        if _CCXT_PRO_AVAILABLE_GLOBAL:
            try:
                async_exchange_class = getattr(ccxt.pro, self.exchange_id, None)
                if async_exchange_class:
                    self._crypto_ex_async = async_exchange_class({
                        'enableRateLimit': True,
                        'timeout': 15000,        
                    })
                    logger.info(f"Initialized CCXT Pro (async) {self.exchange_id} for real-time feed.")
                else:
                    logger.warning(f"CCXT Pro (async) exchange '{self.exchange_id}' not found. "
                                   "Real-time feed will use demo data.")
            except Exception as e:
                logger.error(f"Failed to instantiate CCXT Pro (async) client: {e}. "
                             "Real-time feed will use demo data.", exc_info=True)
                self._crypto_ex_async = None 
        else:
            logger.warning("CCXT Pro (async) is not available. Real-time feed will use demo data.")

        if not self._crypto_ex_sync and not self._crypto_ex_async:
            logger.error("No CCXT exchanges initialized. All data fetching will use demo data.")


    async def fetch_ohlcv(self, symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        """
        Fetches historical OHLCV data.
        Prioritizes cached data, then fetches from exchange (using the synchronous client via asyncio.to_thread),
        and finally falls back to demo data if fetching fails.
        """
        normalized_symbol = symbol.replace('/', '_').lower()
        cache_filename = f"{normalized_symbol}_{timeframe}_{limit}.csv"
        file_path = os.path.join(self.cache_dir, cache_filename)

        # 1. Attempt to load from cache
        if os.path.exists(file_path):
            try:
                df_cached = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                if not df_cached.empty and len(df_cached) >= limit:
                    logger.info(f"Loaded {len(df_cached)} candles for {symbol} ({timeframe}) from cache: {file_path}")
                    return df_cached
                else:
                    logger.warning(f"Cached data for {symbol} is incomplete or empty. Refetching from exchange.")
            except Exception as e:
                logger.error(f"Error loading cached data from {file_path}: {e}. Refreshing from exchange.")

        # 2. If cache failed or not available, try to fetch from exchange (using sync client via to_thread)
        if self._crypto_ex_sync: # Proceed only if sync client is initialized
            logger.info(f"Fetching {limit} candles for symbol {symbol} ({timeframe}) from exchange (sync)...")
            try:
                # Load markets only if not already loaded for the sync client
                # Use the custom _markets_loaded_sync flag
                if not self._markets_loaded_sync: 
                    await asyncio.to_thread(self._crypto_ex_sync.load_markets)
                    self._markets_loaded_sync = True # Set flag to True on success
                    logger.info(f"CCXT (sync) {self.exchange_id} markets loaded successfully.")

                raw_data = await asyncio.to_thread(
                    self._crypto_ex_sync.fetch_ohlcv,
                    symbol, timeframe=timeframe, limit=limit
                )

                if not raw_data: # If CCXT returns an empty list, it's not a failure, just no data
                    logger.warning(f"No OHLCV data obtained from exchange for {symbol} ({timeframe}).")
                    return pd.DataFrame()

                df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                if df.empty:
                    logger.warning(f"No data after processing for {symbol} ({timeframe}).")
                    return pd.DataFrame()

                # 3. Cache the fetched data
                try:
                    df.to_csv(file_path, index=True)
                    logger.info(f"Successfully fetched and cached {len(df)} candles for {symbol} ({timeframe}) to {file_path}")
                except Exception as e:
                    logger.error(f"Failed to save fetched data to cache {file_path}: {e}")
                return df

            except ccxt.NetworkError as e:
                logger.warning(f"Network error fetching OHLCV data for {symbol} ({timeframe}): {e}. Fallback to demo data.")
            except ccxt.ExchangeError as e:
                logger.warning(f"Exchange error fetching OHLCV data for {symbol} ({timeframe}): {e}. Fallback to demo data.")
            except Exception as e:
                logger.warning(f"An unexpected error occurred while fetching OHLCV data for {symbol} ({timeframe}): {e}. Fallback to demo data.", exc_info=True)

        # 4. Fallback to demo data if real fetch failed or no sync exchange available
        logger.info(f"Generating demo OHLCV data for {symbol} (limit={limit}).")
        return self._generate_demo_ohlcv(symbol, limit)

    def _generate_demo_ohlcv(self, symbol: str, limit: int) -> pd.DataFrame:
        """Helper to generate synthetic OHLCV data for demo mode."""
        np.random.seed(42)  # For reproducible demo data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1h') 
        base_price = 50000 if 'BTC' in symbol else (3000 if 'ETH' in symbol else 1)
        prices_series = base_price + np.random.normal(0, base_price * 0.005, limit).cumsum()
        volumes_series = np.random.exponential(1000, limit) * (1 + np.random.normal(0, 0.2, limit))

        data = []
        for i in range(limit):
            dt = dates[i]
            price = prices_series[i]
            volume = volumes_series[i]
            open_price = price * (1 + random.uniform(-0.001, 0.001))
            high_price = max(open_price, price) * (1 + random.uniform(0.0005, 0.005))
            low_price = min(open_price, price) * (1 - random.uniform(0.0005, 0.005))
            close_price = price
            data.append([dt.timestamp() * 1000, open_price, high_price, low_price, close_price, volume])

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.ffill().bfill() 

    async def _feed_loop(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Asynchronous internal loop for fetching real-time market data."""
        feed_symbols = list(self.symbols) 

        # --- Lazy Load Markets for async client (only once per instance) ---
        if self._crypto_ex_async and not self._markets_loaded_async: 
            try:
                logger.info(f"Attempting to load markets for CCXT Pro {self.exchange_id}...")
                await self._crypto_ex_async.load_markets() 
                self._markets_loaded_async = True 
                logger.info(f"CCXT Pro {self.exchange_id} markets loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load markets for CCXT Pro {self.exchange_id}: {e}. "
                             "Real-time feed will remain in demo mode.", exc_info=True)
                self._crypto_ex_async = None 

        while self.running_feed:
            market_data_batch = {}
            for symbol in feed_symbols:
                try:
                    if self._crypto_ex_async and self._markets_loaded_async: # Ensure client is ready
                        if symbol not in self._crypto_ex_async.symbols: 
                            logger.warning(f"Symbol {symbol} not found on {self.exchange_id} exchange. Skipping real-time fetch.")
                            market_data_batch[symbol] = self._generate_demo_ticker(symbol) 
                            continue

                        ticker = await self._crypto_ex_async.fetch_ticker(symbol)
                        
                        timestamp_ms = ticker.get('timestamp')
                        timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000) if timestamp_ms is not None else datetime.now()

                        market_data_batch[symbol] = {
                            'symbol': symbol,
                            'price': ticker['last'],
                            'volume': ticker.get('baseVolume', 0),
                            'change_24h': ticker.get('percentage', 0),
                            'timestamp': timestamp_dt 
                        }
                    else: # Fallback to demo if async client is not available (or was disabled due to market load error)
                        market_data_batch[symbol] = self._generate_demo_ticker(symbol)
                except (ccxt.NetworkError, ccxt.ExchangeError) as e: 
                    logger.warning(f"Failed to fetch real-time data for {symbol} ({self.exchange_id}): {e}. Generating demo data for this cycle.")
                    market_data_batch[symbol] = self._generate_demo_ticker(symbol)
                except Exception as e: 
                    logger.warning(f"An unexpected error in _feed_loop for {symbol}: {e}. Generating demo data for this cycle.", exc_info=False)
                    market_data_batch[symbol] = self._generate_demo_ticker(symbol)

            if market_data_batch:
                await callback(market_data_batch) # Pass batch to callback
            await asyncio.sleep(settings.BROADCAST_INTERVAL_SECONDS)

    def _generate_demo_ticker(self, symbol: str) -> Dict[str, Any]:
        """Helper to generate synthetic ticker data for demo mode."""
        base_price = 50000 if 'BTC' in symbol else (3000 if 'ETH' in symbol else 1)
        price = base_price * (1 + random.uniform(-0.01, 0.01))
        volume = random.uniform(500, 2000)
        change_24h = random.uniform(-5, 5)
        return {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'change_24h': change_24h,
            'timestamp': datetime.now()
        }

    async def start_real_time_feed(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Starts the real-time market data feed as an asyncio task."""
        if not self.running_feed:
            self.running_feed = True
            try:
                # Create the task without blocking the main event loop
                asyncio.create_task(self._feed_loop(callback))
                logger.info("Real-time crypto data feed task scheduled. (Can still take time to load markets/data)")
            except Exception as e:
                logger.error(f"Failed to start real-time data feed: {e}", exc_info=True)
                self.running_feed = False

    async def stop_feed(self):
        """Stops the real-time market data feed."""
        if not self.running_feed:
            logger.info("Crypto data feed is already stopped.")
            return

        self.running_feed = False
        logger.info("Signaling crypto data feed to stop...")
        
        if self._crypto_ex_async and hasattr(self._crypto_ex_async, 'close'):
            try:
                await self._crypto_ex_async.close()
                logger.info("Closed CCXT async exchange connection.")
            except Exception as e:
                logger.error(f"Error closing CCXT async exchange connection: {e}", exc_info=True)
        else: 
            pass 

        logger.info("Crypto data feed stopped.")