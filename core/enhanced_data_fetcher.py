# enhanced_data_fetcher.py - Enhanced Data Fetcher with Multi-Timeframe Support
"""
Enhanced Data Fetcher with comprehensive data collection and caching
Supports multiple timeframes, large datasets, and market data enrichment
"""

import ccxt
import ccxt.pro as ccxtpro
import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any
import pickle
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class EnhancedDataFetcher:
    def __init__(self, exchange_name: str = 'kraken', cache_dir: str = 'data/cache'):
        self.exchange_name = exchange_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize exchanges
        self.sync_exchange = None
        self.async_exchange = None
        self._initialize_exchanges()
        
        # Data validation parameters
        self.min_required_samples = 500
        self.max_gap_tolerance = 0.05  # 5% missing data tolerance
        
        # Rate limiting
        self.rate_limit_delay = 1.2  # seconds between requests
        self.last_request_time = 0
        
        self.logger.info(f"Enhanced Data Fetcher initialized with {exchange_name}")
    
    def _initialize_exchanges(self):
        """Initialize both sync and async exchange instances"""
        try:
            # Sync exchange
            exchange_class = getattr(ccxt, self.exchange_name)
            self.sync_exchange = exchange_class({
                'apiKey': '',  # Add your API keys
                'secret': '',
                'timeout': 30000,
                'enableRateLimit': True,
                'rateLimit': 1200,  # Kraken rate limit
            })
            
            # Async exchange
            async_exchange_class = getattr(ccxtpro, self.exchange_name)
            self.async_exchange = async_exchange_class({
                'apiKey': '',
                'secret': '',
                'timeout': 30000,
                'enableRateLimit': True,
                'rateLimit': 1200,
            })
            
            self.logger.info("Exchanges initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchanges: {e}")
            raise
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
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
                        total_candles: int = 5000, use_cache: bool = True) -> pd.DataFrame:
        """Fetch large amounts of OHLCV data using pagination"""
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
                self.logger.warning(f"Failed to load cache: {e}")
        
        # Load markets
        if not self.sync_exchange.markets:
            self.sync_exchange.load_markets()
        
        # Calculate timeframe in milliseconds
        timeframe_ms = self._timeframe_to_ms(timeframe)
        
        # Fetch data in chunks
        all_data = []
        max_limit = 1000  # Most exchanges limit to 1000 candles per request
        
        # Calculate how many requests we need
        num_requests = (total_candles + max_limit - 1) // max_limit
        
        # Start from current time and go backwards
        end_time = int(time.time() * 1000)
        
        for i in range(num_requests):
            try:
                self._rate_limit()
                
                # Calculate limit for this request
                remaining_candles = total_candles - len(all_data)
                current_limit = min(max_limit, remaining_candles)
                
                if current_limit <= 0:
                    break
                
                # Calculate since parameter (going backwards in time)
                since = end_time - (i + 1) * current_limit * timeframe_ms
                
                self.logger.debug(f"Fetching chunk {i+1}/{num_requests}: "
                                f"{current_limit} candles from {datetime.fromtimestamp(since/1000)}")
                
                # Fetch data
                ohlcv = self.sync_exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=current_limit
                )
                
                if not ohlcv:
                    self.logger.warning(f"No data received for chunk {i+1}")
                    break
                
                # Convert to DataFrame
                chunk_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='ms')
                chunk_df.set_index('timestamp', inplace=True)
                
                all_data.append(chunk_df)
                
                self.logger.debug(f"Fetched {len(chunk_df)} candles for chunk {i+1}")
                
            except Exception as e:
                self.logger.error(f"Error fetching chunk {i+1}: {e}")
                # Continue with next chunk instead of failing completely
                continue
        
        if not all_data:
            raise ValueError(f"No data could be fetched for {symbol} ({timeframe})")
        
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
        with ThreadPoolExecutor(max_workers=3) as executor:  # Limit concurrent requests
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
                    results[symbol][timeframe] = pd.DataFrame()
        
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
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return list(data.values())[0].get('usd_market_cap')
            
        except Exception as e:
            self.logger.debug(f"Could not fetch market cap for {symbol}: {e}")
        
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
                asyncio.run(self.async_exchange.close())
        except Exception as e:
            self.logger.warning(f"Error closing async exchange: {e}")
        
        self.logger.info("Data fetcher closed")