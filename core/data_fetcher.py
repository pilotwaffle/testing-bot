import ccxt
import ccxt.pro as ccxtpro
import os
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import time
import random
from typing import Dict, List, Optional, Tuple, Any
import pickle
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define custom exceptions for better error handling
class DataFetcherError(Exception):
    """Base exception for DataFetcher errors."""
    pass

class RateLimitError(DataFetcherError):
    """Raised when an API rate limit is hit."""
    pass

class ExchangeConnectionError(DataFetcherError):
    """Raised when there's an issue connecting to the exchange."""
    pass

class DataFetcher:
    def __init__(self, exchange_name: str = 'kraken', cache_dir: str = 'data/cache',
                 trading_engine: Optional[Any] = None): # ADDED trading_engine parameter
        self.exchange_name = exchange_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        self.trading_engine = trading_engine # Store the trading engine instance

        # Initialize exchanges
        self.sync_exchange = None
        self.async_exchange = None
        self._initialize_exchanges()

        # Data validation parameters
        self.min_required_samples = 500
        self.max_gap_tolerance = 0.05  # 5% missing data tolerance

        # Rate limiting configuration (per request, additional to ccxt's internal)
        self.request_delay_sec = 0.5 # A small delay between individual requests
        self.last_request_time = 0

        self.logger.info(f"Data Fetcher initialized with {exchange_name}")

    def _initialize_exchanges(self):
        """Initialize both sync and async exchange instances"""
        try:
            # IMPORTANT: Replace with your actual Kraken API keys.
            # Leaving them empty will likely cause authentication errors for private endpoints.
            # For demonstration, placeholders are used.
            api_key = os.getenv('KRAKEN_API_KEY', '') # Recommended: Use environment variables
            secret = os.getenv('KRAKEN_SECRET', '') # Recommended: Use environment variables

            # Sync exchange
            exchange_class = getattr(ccxt, self.exchange_name)
            self.sync_exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'timeout': 30000,
                'enableRateLimit': True,
                'rateLimit': 2000,   # Increased rate limit to 2000ms (2 seconds)
                                     # to be more conservative with Kraken's limits
            })

            # Async exchange
            async_exchange_class = getattr(ccxtpro, self.exchange_name)
            self.async_exchange = async_exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'timeout': 30000,
                'enableRateLimit': True,
                'rateLimit': 2000, # Increased rate limit to 2000ms (2 seconds)
            })

            self.logger.info("Exchanges initialized successfully")

        except AttributeError:
            # Raised if exchange_name is not a valid ccxt exchange
            raise ExchangeConnectionError(f"Exchange '{self.exchange_name}' not found in CCXT. Check name or installation.")
        except Exception as e:
            self.logger.error(f"Failed to initialize exchanges: {e}")
            raise ExchangeConnectionError(f"Failed to initialize exchanges: {e}")

    def _rate_limit_local_pause(self):
        """Apply a local rate limiting pause between individual requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay_sec:
            sleep_time = self.request_delay_sec - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _get_cache_filename(self, symbol: str, timeframe: str, limit: int,
                           start_date: Optional[datetime] = None) -> Path:
        """Generate cache filename"""
        symbol_clean = symbol.replace('/', '_').lower()

        if start_date:
            date_str = start_date.strftime('%Y%m%d')
            return self.cache_dir / f"{symbol_clean}_{timeframe}_{limit}_{date_str}.pkl"
        else:
            return self.cache_dir / f"{symbol_clean}_{timeframe}_{limit}_latest.pkl"

    def _is_cache_valid(self, cache_file: Path, max_age_hours: int = 1) -> bool:
        """Check if cached data is still valid"""
        if not cache_file.exists():
            return False

        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)

    def fetch_ohlcv_bulk(self, symbol: str, timeframe: str,
                         total_candles: int = 5000, use_cache: bool = True,
                         max_retries: int = 5, retry_delay_base: int = 5) -> pd.DataFrame:
        """
        Fetch large amounts of OHLCV data using pagination with retry logic for rate limits.
        """
        self.logger.info(f"Fetching {total_candles} candles for {symbol} ({timeframe})")

        # Check cache first
        cache_file = self._get_cache_filename(symbol, timeframe, total_candles)

        if use_cache and self._is_cache_valid(cache_file, max_age_hours=6):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.logger.info(f"Loaded {len(cached_data)} candles from cache")
                return cached_data
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Proceeding with live fetch.")

        # Load markets
        if not self.sync_exchange.markets:
            try:
                self.sync_exchange.load_markets()
            except Exception as e:
                raise ExchangeConnectionError(f"Failed to load markets: {e}")

        # Calculate timeframe in milliseconds
        timeframe_ms = self._timeframe_to_ms(timeframe)

        # Fetch data in chunks
        all_data = []
        max_limit = 1000  # Most exchanges limit to 1000 candles per request

        # Calculate how many requests we need
        # Start from current time and go backwards
        end_time = int(time.time() * 1000)

        # To fetch total_candles, we might need multiple requests, going backward in time
        # We start from current_time and fetch chunks backwards until total_candles are accumulated.
        # This approach ensures we always get the *latest* total_candles.
        current_data_count = 0
        while current_data_count < total_candles:
            retries = 0
            while retries < max_retries:
                try:
                    # Apply local rate limiting before each request
                    self._rate_limit_local_pause()

                    # Calculate 'since' based on the earliest timestamp in collected data
                    # If all_data is empty, start from end_time and go back total_candles * timeframe_ms
                    # Otherwise, fetch from the earliest timestamp already collected.
                    if not all_data:
                        since = end_time - total_candles * timeframe_ms
                    else:
                        earliest_timestamp_ms = all_data[-1].index.min().value // 1_000_000 # Convert nanoseconds to milliseconds
                        since = earliest_timestamp_ms - max_limit * timeframe_ms # Go back one chunk from the earliest

                    self.logger.debug(f"Fetching chunk (retry {retries+1}/{max_retries}): "
                                      f"limit={max_limit} from {datetime.fromtimestamp(since/1000)}")

                    # Fetch data
                    ohlcv = self.sync_exchange.fetch_ohlcv(
                        symbol, timeframe, since=since, limit=max_limit
                    )

                    if not ohlcv:
                        self.logger.warning(f"No data received for chunk after {retries+1} attempts.")
                        # If no data is received consistently, break from retries for this chunk
                        break

                    # Convert to DataFrame
                    chunk_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='ms')
                    chunk_df.set_index('timestamp', inplace=True)

                    all_data.append(chunk_df)
                    current_data_count += len(chunk_df)
                    self.logger.debug(f"Fetched {len(chunk_df)} candles. Total collected: {current_data_count}")
                    break # Break from retry loop, chunk fetched successfully

                except (ccxt.RateLimitExceeded, requests.exceptions.RequestException) as e:
                    retries += 1
                    sleep_duration = retry_delay_base * (2 ** (retries - 1)) + random.uniform(0, 1) # Exponential backoff with jitter
                    self.logger.warning(f"Rate limit hit or request error for {symbol} ({timeframe}): {e}. Retrying in {sleep_duration:.2f} seconds...")
                    time.sleep(sleep_duration)
                    if retries >= max_retries:
                        self.logger.error(f"Failed to fetch chunk for {symbol} ({timeframe}) after {max_retries} retries due to rate limit/request error.")
                        break # Give up on this chunk after max retries
                except Exception as e:
                    self.logger.error(f"Unexpected error fetching chunk for {symbol} ({timeframe}): {e}")
                    break # Break from retry loop for unexpected errors

            if retries >= max_retries: # If max retries reached for a chunk, try to continue with available data
                self.logger.warning(f"Skipping further fetching for {symbol} ({timeframe}) due to persistent errors.")
                break


        if not all_data:
            raise ValueError(f"No data could be fetched for {symbol} ({timeframe}) after all attempts.")

        # Combine all chunks
        combined_df = pd.concat(all_data, axis=0)

        # Remove duplicates and sort
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df.sort_index(inplace=True)

        # Take only the most recent candles we need
        if len(combined_df) > total_candles:
            combined_df = combined_df.tail(total_candles)

        # Data validation
        self._validate_data_quality(combined_df, symbol, timeframe)

        # Cache the data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(combined_df, f)
            self.logger.info(f"Cached {len(combined_df)} candles")
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")

        self.logger.info(f"Successfully fetched {len(combined_df)} candles for {symbol} ({timeframe})")
        return combined_df

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }

        return timeframe_map.get(timeframe, 60 * 60 * 1000)  # Default to 1h

    def _validate_data_quality(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Validate the quality of fetched data"""
        if len(data) < self.min_required_samples:
            self.logger.warning(f"Insufficient data for {symbol} ({timeframe}): "
                                f"{len(data)} samples (minimum: {self.min_required_samples})")

        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > self.max_gap_tolerance:
            self.logger.warning(f"High missing data ratio for {symbol} ({timeframe}): "
                                f"{missing_ratio:.2%}")

        # Check for data gaps
        expected_interval = self._timeframe_to_ms(timeframe) / 1000
        time_diffs = data.index.to_series().diff().dt.total_seconds()
        large_gaps = (time_diffs > expected_interval * 2).sum()

        if large_gaps > len(data) * 0.01:  # More than 1% gaps
            self.logger.warning(f"Data gaps detected for {symbol} ({timeframe}): "
                                f"{large_gaps} gaps")

        # Check for anomalous values
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                # Check for zero or negative prices
                invalid_prices = (data[col] <= 0).sum()
                if invalid_prices > 0:
                    self.logger.warning(f"Invalid prices in {col} for {symbol}: {invalid_prices} values")

                # Check for extreme outliers (price changes > 50%)
                returns = data[col].pct_change()
                extreme_moves = (abs(returns) > 0.5).sum()
                if extreme_moves > len(data) * 0.001:  # More than 0.1% extreme moves
                    self.logger.warning(f"Extreme price movements detected in {col} for {symbol}: "
                                        f"{extreme_moves} occurrences")

    def fetch_multiple_symbols(self, symbols: List[str], timeframes: List[str],
                              total_candles: int = 5000) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch data for multiple symbols and timeframes in parallel"""
        self.logger.info(f"Fetching data for {len(symbols)} symbols, {len(timeframes)} timeframes")

        results = {}

        # Create tasks for parallel execution
        tasks = []
        for symbol in symbols:
            results[symbol] = {}
            for timeframe in timeframes:
                tasks.append((symbol, timeframe, total_candles))

        # Execute tasks with thread pool
        # Increased max_workers to allow more parallelism, but individual fetches have backoff
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_task = {
                executor.submit(self.fetch_ohlcv_bulk, symbol, timeframe, total_candles): (symbol, timeframe)
                for symbol, timeframe, total_candles in tasks
            }

            for future in as_completed(future_to_task):
                symbol, timeframe = future_to_task[future]
                try:
                    data = future.result()
                    results[symbol][timeframe] = data
                    self.logger.info(f"Completed {symbol} ({timeframe}): {len(data)} candles")
                except Exception as e:
                    self.logger.error(f"Failed to fetch {symbol} ({timeframe}): {e}")
                    results[symbol][timeframe] = pd.DataFrame() # Ensure an empty DataFrame is returned on failure

        return results

    def enrich_with_market_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Enrich OHLCV data with additional market information"""
        enriched_data = data.copy()

        try:
            # Add market cap data (if available)
            market_cap = self._get_market_cap(symbol)
            if market_cap:
                enriched_data['market_cap'] = market_cap

            # Add trading volume ranking
            enriched_data['volume_rank'] = enriched_data['volume'].rolling(100).rank(pct=True)

            # Add volatility regime classification
            returns = enriched_data['close'].pct_change()
            vol_20 = returns.rolling(20).std()
            vol_percentile = vol_20.rolling(100).rank(pct=True)

            enriched_data['volatility_regime'] = pd.cut(
                vol_percentile,
                bins=[0, 0.33, 0.67, 1.0],
                labels=['Low', 'Medium', 'High']
            )

            # Add market session indicators (if timestamp available)
            if isinstance(enriched_data.index, pd.DatetimeIndex):
                enriched_data['market_session'] = self._classify_market_session(enriched_data.index)

        except Exception as e:
            self.logger.warning(f"Failed to enrich market data: {e}")

        return enriched_data

    def _get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market cap data from external API"""
        try:
            # Example using CoinGecko API (replace with your preferred source)
            base_currency = symbol.split('/')[0].lower()

            if base_currency in ['btc', 'eth', 'ada']:  # Only for major cryptos
                url = f"https://api.coingecko.com/api/v3/simple/price"
                params = {
                    'ids': {'btc': 'bitcoin', 'eth': 'ethereum', 'ada': 'cardano'}[base_currency],
                    'vs_currencies': 'usd',
                    'include_market_cap': 'true'
                }
                self._rate_limit_local_pause() # Add local rate limit for external API calls
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()
                return list(data.values())[0].get('usd_market_cap')

        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Could not fetch market cap for {symbol} (Request Error): {e}")
        except Exception as e:
            self.logger.debug(f"Could not fetch market cap for {symbol} (Generic Error): {e}")

        return None

    def _classify_market_session(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """Classify market sessions based on time"""
        hours = timestamps.hour

        conditions = [
            (hours >= 0) & (hours < 8),    # Asian session
            (hours >= 8) & (hours < 16),   # European session
            (hours >= 16) & (hours < 24),  # American session
        ]

        choices = ['Asian', 'European', 'American']

        return pd.Series(
            np.select(conditions, choices, default='Unknown'),
            index=timestamps,
            name='market_session'
        )

    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time ticker data for multiple symbols"""
        real_time_data = {}

        try:
            for symbol in symbols:
                # Asynchronous rate limiting might be handled by ccxt.pro's internal mechanisms
                # but adding a small sleep here can provide an extra layer of caution
                await asyncio.sleep(self.request_delay_sec)
                ticker = await self.async_exchange.fetch_ticker(symbol)
                real_time_data[symbol] = {
                    'last_price': ticker['last'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'volume_24h': ticker['quoteVolume'],
                    'change_24h': ticker['percentage'],
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Failed to fetch real-time data: {e}")

        return real_time_data

    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        summary = {
            'total_samples': len(data),
            'date_range': {
                'start': data.index.min().isoformat() if len(data) > 0 else None,
                'end': data.index.max().isoformat() if len(data) > 0 else None,
            },
            'data_quality': {
                'missing_values': data.isnull().sum().to_dict(),
                'complete_rows': len(data) - data.isnull().any(axis=1).sum(),
            },
            'basic_stats': {
                'price_range': {
                    'min': float(data['close'].min()) if 'close' in data.columns else None,
                    'max': float(data['close'].max()) if 'close' in data.columns else None,
                    'mean': float(data['close'].mean()) if 'close' in data.columns else None,
                },
                'volume_stats': {
                    'min': float(data['volume'].min()) if 'volume' in data.columns else None,
                    'max': float(data['volume'].max()) if 'volume' in data.columns else None,
                    'mean': float(data['volume'].mean()) if 'volume' in data.columns else None,
                } if 'volume' in data.columns else None,
            }
        }

        return summary

    def cleanup_old_cache(self, max_age_days: int = 7):
        """Remove old cache files"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0

        for cache_file in self.cache_dir.glob("*.pkl"):
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_time < cutoff_time:
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} old cache files")

    def close(self):
        """Close exchange connections"""
        try:
            if self.async_exchange:
                # Ensure the async exchange is properly closed.
                # asyncio.run() needs to be called from a synchronous context,
                # if close() is always called from a sync context this is fine.
                # If close() can be called from an async context, you'd need to await it.
                if self.async_exchange.session:
                    asyncio.run(self.async_exchange.session.close())
                asyncio.run(self.async_exchange.close()) # Call close on the exchange itself
        except Exception as e:
            self.logger.warning(f"Error closing async exchange: {e}")

        self.logger.info("Data fetcher closed")