#!/usr/bin/env python3
"""
================================================================================
FILE: quick_rate_limit_fix.py  
LOCATION: E:\Trade Chat Bot\G Trading Bot\quick_rate_limit_fix.py
AUTHOR: Claude AI Assistant
CREATED: Sunday, June 29, 2025
PURPOSE: Immediate fix for "Too many requests" errors
================================================================================
"""

import time
import ccxt

# QUICK FIX: Add this to your optimized_model_trainer.py

def create_safe_exchange():
    """Create exchange with conservative rate limiting"""
    
    exchange = ccxt.kraken({
        'enableRateLimit': True,
        'rateLimit': 3000,  # 3 seconds between requests (very conservative)
        'timeout': 30000,   # 30 second timeout
    })
    
    print("🛡️ Created rate-limited Kraken exchange (3s delays)")
    return exchange

def safe_fetch_with_delays(exchange, symbol, timeframe, limit=500):
    """Fetch data with mandatory delays"""
    
    max_retries = 3
    base_delay = 5  # Start with 5 second delay
    
    for attempt in range(max_retries):
        try:
            print(f"📡 Fetching {symbol} {timeframe} (attempt {attempt + 1})")
            
            # Mandatory delay before each request
            if attempt > 0:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"⏳ Waiting {delay}s due to previous errors...")
                time.sleep(delay)
            else:
                print("⏳ Standard 3s delay...")
                time.sleep(3)  # Always wait 3 seconds
            
            # Make request
            data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            print(f"✅ Success: {len(data)} candles fetched")
            return data
            
        except Exception as e:
            if "Too many requests" in str(e):
                delay = 30 * (attempt + 1)  # Progressive longer delays
                print(f"⚠️ Rate limit hit! Waiting {delay}s...")
                time.sleep(delay)
            else:
                print(f"❌ Error: {str(e)[:100]}...")
                time.sleep(10)
    
    print(f"💀 Failed to fetch {symbol} {timeframe} after {max_retries} attempts")
    return None

# IMMEDIATE INTEGRATION:
print("""
🚨 IMMEDIATE FIX FOR YOUR CURRENT TRAINING:

1. Stop your current training (Ctrl+C)

2. Edit your optimized_model_trainer.py:
   
   # Find your exchange initialization and replace with:
   exchange = create_safe_exchange()
   
   # Find your data fetching and replace with:
   data = safe_fetch_with_delays(exchange, symbol, timeframe)

3. Restart training:
   python optimized_model_trainer.py --full-train --enhanced

Expected result:
❌ No more "Too many requests" errors
✅ 3-5 second delays between requests
✅ Training completes successfully
""")
