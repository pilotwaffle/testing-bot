"""
File: E:\Trade Chat Bot\G Trading Bot\core\market_data_processor.py
Location: E:\Trade Chat Bot\G Trading Bot\core\market_data_processor.py

Market Data Processor for Trading Bot - Corrected and Enhanced
"""
import time
import random
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio # asyncio is imported but not used directly in current methods, good for future async operations

logger = logging.getLogger(__name__)

class MarketDataProcessor:
    """Processes and manages market data, providing caching and historical data simulation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize market data processor.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary for the processor.
        """
        self.config = config or {}
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.last_update: Dict[str, datetime] = {}
        self.update_interval: int = 60  # seconds
        self.latest_data: Dict[str, Dict[str, Any]] = {} # Initialize latest_data here
        self.historical_cache: Dict[str, List[Dict[str, Any]]] = {} # Initialize historical_cache here
        logger.info("Market Data Processor initialized.")

    def process_data(self, raw_data: Any, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Process raw market data into a standardized format.

        Args:
            raw_data (Any): The raw data to process. Can be a dict, list of dicts, or a numeric value.
            symbol (Optional[str]): The symbol associated with the data, if not present in raw_data.

        Returns:
            Optional[Dict[str, Any]]: The processed data in a standardized dictionary format,
                                       or None if processing fails.
        """
        try:
            if not raw_data:
                logger.warning(f"Attempted to process empty raw_data for symbol: {symbol}")
                return None
            
            processed: Optional[Dict[str, Any]] = None
            
            if isinstance(raw_data, dict):
                processed = {
                    "symbol": symbol or raw_data.get("symbol", "UNKNOWN"),
                    "price": float(raw_data.get("price", 0.0)),
                    "timestamp": datetime.fromtimestamp(raw_data.get("timestamp", time.time())) if isinstance(raw_data.get("timestamp"), (int, float)) else raw_data.get("timestamp", datetime.now()),
                    "volume": float(raw_data.get("volume", 0.0)),
                    "bid": float(raw_data.get("bid", 0.0)),
                    "ask": float(raw_data.get("ask", 0.0)),
                    "high": float(raw_data.get("high", 0.0)),
                    "low": float(raw_data.get("low", 0.0)),
                    "open": float(raw_data.get("open", 0.0)),
                    "close": float(raw_data.get("close", 0.0))
                }
            elif isinstance(raw_data, list):
                # Recursively process list of data points
                processed_list = [self.process_data(item, symbol) for item in raw_data if isinstance(item, dict)]
                return [item for item in processed_list if item is not None] # Filter out None values
            else:
                # Handle simple numeric data (e.g., just a price)
                processed = {
                    "symbol": symbol or "UNKNOWN",
                    "price": float(raw_data),
                    "timestamp": datetime.now(),
                    "volume": 0.0,
                    "status": "processed"
                }
            
            # Store in latest_data cache
            if processed and isinstance(processed, dict) and "symbol" in processed:
                self.latest_data[processed["symbol"]] = processed
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}", exc_info=True)
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical market data for analysis.

        Args:
            symbol (str): The trading symbol (e.g., "BTC/USDT").
            timeframe (str): The data timeframe (e.g., "1h", "1d").
            limit (int): The maximum number of historical data points to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a historical data point.
        """
        try:
            # Return cached historical data if available and matches criteria
            cache_key = f"{symbol}_{timeframe}_{limit}"
            if cache_key in self.historical_cache:
                return self.historical_cache[cache_key]
            
            # Generate placeholder historical data for now
            historical_data = []
            base_price = 50000.0 if symbol.startswith("BTC") else 3000.0 if symbol.startswith("ETH") else 100.0
            
            for i in range(limit):
                timestamp = datetime.now() - timedelta(hours=i) # 1 hour intervals
                price_variation = random.uniform(-0.05, 0.05)  # ±5% variation
                price = base_price * (1 + price_variation)
                
                data_point = {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": price * random.uniform(0.995, 1.005),
                    "high": price * random.uniform(1.001, 1.02),
                    "low": price * random.uniform(0.98, 0.999),
                    "close": price,
                    "volume": random.uniform(1000, 10000),
                    "timeframe": timeframe
                }
                historical_data.append(data_point)
            
            # Cache the data
            self.historical_cache[cache_key] = historical_data
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}", exc_info=True)
            return []

    def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest market data for a symbol.

        Args:
            symbol (str): The trading symbol (e.g., "BTC/USDT").

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the latest data, or None if not available.
        """
        try:
            # Return cached data if available
            if symbol in self.latest_data:
                return self.latest_data[symbol]
            
            # Fallback to basic structure if no data processed yet
            return {
                "symbol": symbol,
                "price": 0.0,
                "timestamp": datetime.now(),
                "volume": 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "status": "no_data"
            }
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}", exc_info=True)
            return None

    async def update_symbol(self, symbol: str):
        """
        Asynchronously update market data for a given symbol.
        This is a placeholder for real data fetching logic.

        Args:
            symbol (str): The trading symbol to update.
        """
        try:
            current_time = datetime.now()
            
            # Mock price data with volatility
            mock_price = 50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0
            if symbol in self.price_cache:
                change = random.uniform(-0.02, 0.02)
                mock_price = self.price_cache[symbol].get('price', mock_price) * (1 + change)
            
            self.price_cache[symbol] = {
                'price': mock_price,
                'timestamp': current_time,
                'volume': 1000000 * random.uniform(0.8, 1.2), # Add volume variation
                'high_24h': mock_price * random.uniform(1.01, 1.05),
                'low_24h': mock_price * random.uniform(0.95, 0.99)
            }
            
            self.last_update[symbol] = current_time
            logger.debug(f"Updated {symbol}: ${mock_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}", exc_info=True)
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        Get the last known price for a symbol from the cache.

        Args:
            symbol (str): The trading symbol.

        Returns:
            Optional[float]: The last price, or None if not found.
        """
        if symbol in self.price_cache:
            return self.price_cache[symbol].get('price')
        return None
    
    def get_recent_prices(self, symbol: str, count: int = 100) -> List[float]:
        """
        Get a list of recent simulated prices for a symbol.

        Args:
            symbol (str): The trading symbol.
            count (int): The number of recent prices to return.

        Returns:
            List[float]: A list of recent prices.
        """
        if symbol in self.price_cache:
            base_price = self.price_cache[symbol]['price']
            prices = []
            current_price = base_price
            for _ in range(count):
                change = random.uniform(-0.01, 0.01)
                current_price *= (1 + change)
                prices.append(current_price)
            return prices
        return []
    
    def get_price_at(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """
        Get an approximate price for a symbol at a specific timestamp.
        This is a placeholder for real historical data lookup.

        Args:
            symbol (str): The trading symbol.
            timestamp (datetime): The specific datetime for which to retrieve the price.

        Returns:
            Optional[float]: The approximate price at the given timestamp, or None if not found.
        """
        if symbol in self.price_cache:
            base_price = self.price_cache[symbol]['price']
            # Add some time-based variation for simulation
            time_diff = datetime.now() - timestamp
            hours_diff = time_diff.total_seconds() / 3600
            variation = hours_diff * 0.001  # 0.1% per hour deviation from current
            return base_price * (1 + variation) * (1 + random.uniform(-0.002, 0.002)) # Add small random noise
        return None
    
    def get_symbol_volatility(self, symbol: str) -> Optional[float]:
        """
        Get simulated volatility for a symbol.
        This is a placeholder for real volatility calculation.

        Args:
            symbol (str): The trading symbol.

        Returns:
            Optional[float]: The simulated volatility (e.g., 0.02 for 2%).
        """
        # Placeholder volatility calculation
        # In a real scenario, this would compute volatility from historical data
        if symbol in self.price_cache: # Only return volatility if symbol is known
            return 0.02 + random.uniform(-0.005, 0.005) # Add small variation
        return None
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get complete market data for a specific symbol from the cache.

        Args:
            symbol (str): The trading symbol.

        Returns:
            Optional[Dict[str, Any]]: A copy of the market data dictionary for the symbol, or None if not found.
        """
        if symbol in self.price_cache:
            return self.price_cache[symbol].copy()
        return None
    
    def get_all_symbols(self) -> List[str]:
        """
        Get all tracked symbols in the cache.

        Returns:
            List[str]: A list of all symbols currently tracked.
        """
        return list(self.price_cache.keys())
    
    def is_data_fresh(self, symbol: str, max_age_seconds: int = 300) -> bool:
        """
        Check if the data for a given symbol is fresh (within max_age_seconds).

        Args:
            symbol (str): The trading symbol.
            max_age_seconds (int): The maximum age in seconds for data to be considered fresh.

        Returns:
            bool: True if data is fresh, False otherwise.
        """
        if symbol not in self.last_update:
            return False
            
        age = (datetime.now() - self.last_update[symbol]).total_seconds()
        return age <= max_age_seconds