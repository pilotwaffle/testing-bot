# train_models.py
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import argparse

# Configure basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model-trainer")

# Import the necessary components from your bot's modules
from core.data_fetcher import CryptoDataFetcher
from ml.ml_engine import OctoBotMLEngine
from core.config import settings # To ensure settings are loaded
from core.notification_manager import SimpleNotificationManager # Used for strategy testing
from strategies.ml_strategy import MLStrategy # Import the strategy itself
from strategies.strategy_base import SignalType # For comparing signals

# --- Main Training Pipeline ---
async def run_training_pipeline():
    logger.info("Initializing ML training pipeline...")
    _ = settings # Accessing settings loads them implicitly via dotenv

    # Initialize components (they won't have the full TradingEngine context)
    data_fetcher = CryptoDataFetcher()
    ml_engine = OctoBotMLEngine()

    # Mock trading engine for ML engine's internal use if needed (e.g., balance access)
    class MockTradingEngine:
        def __init__(self):
            self.balance = 10000.0 # Mock balance
    mock_engine = MockTradingEngine()
    if hasattr(ml_engine, '_engine_reference'):
        ml_engine._engine_reference = mock_engine

    # Load existing models first before trying to train
    ml_engine.load_models()
    logger.info(f"Loaded {len(ml_engine.models)} existing models.")

    symbols_to_train = settings.DEFAULT_TRAINING_SYMBOLS
    model_types_to_train = [
        "neural_network",
        "lorentzian",
        # "social_sentiment", # Only train if you have real data/mocks are sufficient
        "risk_assessment",
    ]
    # Define timeframes to train models for
    timeframes_to_train = ["1h", "1d"] # Add "4h" or other timeframes as needed

    for symbol in symbols_to_train:
        logger.info(f"\n--- Starting training for {symbol} ---")

        for timeframe in timeframes_to_train:
            logger.info(f"\n--- Training for {symbol} at {timeframe} timeframe ---")

            # Adjust limit based on timeframe (e.g., more for hourly, less for daily)
            # You might need to refine these limits based on your data and strategy needs
            if timeframe == "1h":
                fetch_limit = 2000 # Example: ~83 days of hourly data
            elif timeframe == "1d":
                fetch_limit = 365  # Example: 1 year of daily data
            elif timeframe == "4h":
                fetch_limit = 500  # Example: ~83 days of 4-hourly data (2000 hours / 4 hours/candle)
            else:
                fetch_limit = 500 # Default fallback

            # 1. Data Acquisition (from CryptoDataFetcher)
            try:
                df = await data_fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit)
                if df.empty:
                    logger.error(f"No data fetched for {symbol} ({timeframe}). Skipping training for this symbol/timeframe.")
                    continue
                logger.info(f"Fetched {len(df)} data points for {symbol} ({timeframe}).")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} ({timeframe}): {e}", exc_info=True)
                continue

            for model_type in model_types_to_train:
                logger.info(f"Training {model_type} for {symbol} ({timeframe})...")
                try:
                    training_df = df.copy() # Ensure a fresh copy for each model type
                    
                    if model_type == "neural_network":
                        result = ml_engine.train_neural_network(symbol, training_df, timeframe=timeframe)
                    elif model_type == "lorentzian":
                        result = ml_engine.train_lorentzian_classifier(symbol, training_df, timeframe=timeframe)
                    elif model_type == "social_sentiment":
                        # This would need a timeframe-specific social sentiment data source
                        logger.warning(f"Social sentiment model training for {timeframe} not implemented/supported by current data for this timeframe.")
                        continue # Skipping for now
                    elif model_type == "risk_assessment":
                        result = ml_engine.train_risk_assessment_model(symbol, training_df, timeframe=timeframe)
                    else:
                        logger.warning(f"Unknown model type: {model_type}. Skipping.")
                        continue

                    if result and result.get("success", True): # Check if result is not an error dict
                        logger.info(f"Successfully trained {model_type} for {symbol} ({timeframe}). Performance: {result.get('accuracy', result.get('r2_score', 'N/A'))}")
                    else:
                        logger.error(f"Training {model_type} for {symbol} ({timeframe}) FAILED: {result.get('error', 'Unknown error message from training result')}")
                except Exception as e:
                    logger.error(f"Unhandled error during {model_type} training for {symbol} ({timeframe}): {e}", exc_info=True)

    logger.info("\n--- ML Training Pipeline Completed ---")
    logger.info("All trained models are saved and can be loaded by the main bot.")


# --- Basic ML Training Test (single timeframe for quick check) ---
async def test_ml_training_pipeline():
    logger.info("\n=== Running basic ML Training Test ===")
    
    data_fetcher = CryptoDataFetcher()
    ml_engine = OctoBotMLEngine()

    class MockTradingEngine:
        def __init__(self):
            self.balance = 10000.0
    mock_engine = MockTradingEngine()
    if hasattr(ml_engine, '_engine_reference'):
        ml_engine._engine_reference = mock_engine

    test_symbol = "ETH/USDT"
    test_timeframe = "1h" # Testing with hourly data
    test_limit = 500
    
    logger.info(f"Fetching data for {test_symbol} ({test_timeframe}) in ML training test...")
    try:
        df = await data_fetcher.fetch_ohlcv(test_symbol, timeframe=test_timeframe, limit=test_limit)
    except Exception as e:
        logger.error(f"Failed to fetch data for {test_symbol} in ML training test: {e}", exc_info=True)
        return False

    if df.empty:
        logger.warning(f"No data for {test_symbol} ({test_timeframe}), cannot run ML training test.")
        return False

    logger.info(f"Attempting to train Neural Network for {test_symbol} ({test_timeframe})...")
    try:
        # Pass the timeframe to the training method
        result = ml_engine.train_neural_network(test_symbol, df.copy(), timeframe=test_timeframe)
        if result and result.get("success", False) is not False:
            logger.info(f"ML Training Test: Neural Network for {test_symbol} ({test_timeframe}) trained successfully. Accuracy: {result.get('accuracy', 'N/A')}")
            return True
        else:
            logger.error(f"ML Training Test: Neural Network training failed. Result: {result}")
            return False
    except Exception as e:
        logger.error(f"ML Training Test: Error during Neural Network training: {e}", exc_info=True)
        return False

# --- Basic Strategy Test (single timeframe for quick check) ---
async def test_ml_strategy():
    logger.info("\n=== Running basic ML Strategy Test ===")

    data_fetcher = CryptoDataFetcher()
    ml_engine = OctoBotMLEngine()
    notification_manager = SimpleNotificationManager()
    
    class MockTradingEngineForStrategy:
        def __init__(self, ml_engine_ref, notif_manager_ref):
            self.balance = 10000.0
            self.current_market_data = {}
            self.positions = {}
            self.ml_engine = ml_engine_ref
            self.notification_manager = notif_manager_ref

        async def place_order(self, symbol, side, qty, order_type="market", limit_price=None):
            logger.info(f"Mocking order: {side.upper()} {qty} of {symbol}")
            price = self.current_market_data.get(symbol, {}).get('price', 0)
            if price == 0:
                logger.warning(f"Mock place_order: Cannot simulate {side} {qty} {symbol}, price is 0.")
                return 

            if side == "buy":
                cost = qty * price
                if self.balance >= cost:
                    self.balance -= cost
                    if symbol not in self.positions: self.positions[symbol] = {'amount': 0.0, 'entry_price': price}
                    current_value = self.positions[symbol]['amount'] * self.positions[symbol]['entry_price']
                    self.positions[symbol]['amount'] += qty
                    self.positions[symbol]['entry_price'] = (current_value + qty * price) / self.positions[symbol]['amount'] if self.positions[symbol]['amount'] > 0 else 0.0
                else:
                    logger.warning(f"Mock place_order: Insufficient balance {self.balance:.2f} to buy {qty} {symbol} at {price:.2f}.")
            elif side == "sell":
                if symbol in self.positions:
                    if self.positions[symbol]['amount'] >= qty:
                        self.balance += qty * price
                        self.positions[symbol]['amount'] -= qty
                        if self.positions[symbol]['amount'] <= 0: del self.positions[symbol]
                    else:
                        logger.warning(f"Mock place_order: Attempted to sell {qty} {symbol} but only {self.positions[symbol]['amount']} available.")
                else:
                    logger.warning(f"Mock place_order: No position for {symbol} to sell.")
                    
            logger.info(f"Mock Balance after order: {self.balance:.2f} for {symbol}.")
            
        def get_performance_metrics(self):
            return {'total_value': self.balance}

    mock_engine = MockTradingEngineForStrategy(ml_engine, notification_manager)
    if hasattr(ml_engine, '_engine_reference'):
        ml_engine._engine_reference = mock_engine


    # First, train a model that the strategy can use
    test_symbol = "BTC/USDT"
    test_timeframe = "1h" # The primary timeframe for this test
    # Ensure a model for both 1h and 1d is present for the multi-timeframe strategy test
    # This pre-training step will ensure the models are available in the engine's cache
    
    logger.info(f"Pre-training Neural Network models for {test_symbol} (1h and 1d) for strategy test...")
    try:
        # Pre-train 1h model
        hist_df_for_training_1h = await data_fetcher.fetch_ohlcv(test_symbol, timeframe="1h", limit=1000)
        train_result_1h = ml_engine.train_neural_network(test_symbol, hist_df_for_training_1h.copy(), timeframe="1h")

        # Pre-train 1d model
        hist_df_for_training_1d = await data_fetcher.fetch_ohlcv(test_symbol, timeframe="1d", limit=90)
        train_result_1d = ml_engine.train_neural_network(test_symbol, hist_df_for_training_1d.copy(), timeframe="1d")

        if not (train_result_1h and train_result_1h.get("success", False) is not False and
                train_result_1d and train_result_1d.get("success", False) is not False):
            logger.error("Failed to pre-train necessary models for strategy test correctly. Skipping strategy test.")
            return False
    except Exception as e:
        logger.error(f"Error during pre-training for strategy test: {e}. Skipping strategy test.", exc_info=True)
        return False
        
    ml_engine.load_models() # Ensure models are loaded into ml_engine instance

    # Now instantiate the MLStrategy
    # Configuration now incorporates settings for multiple timeframes
    ml_strategy_config = {
        "model_type": "neural_network",
        "symbol": test_symbol,
        "timeframes_config": {
            "1h": {
                "required_history": 200, # Candles needed for 1h indicators/features
                "prediction_threshold_buy": 0.55,
                "prediction_threshold_sell": 0.45,
            },
            "1d": {
                "required_history": 60,  # Candles needed for 1d indicators/features
                "prediction_threshold_buy": 0.5, # Daily threshold for bullish bias
                "prediction_threshold_sell": 0.5, # Daily threshold for bearish bias (can be same or different)
            }
        },
        "take_profit_percent": 0.05,
        "stop_loss_percent": 0.02,
        "allocation_percent": 0.1,
        "single_position_only": True
    }
    ml_strategy = MLStrategy(ml_strategy_config, ml_engine, data_fetcher) # Pass data_fetcher

    if not ml_strategy.validate_config():
        logger.error("ML Strategy configuration is invalid. Skipping strategy test.")
        return False

    # Simulate current market data (primarily needed by Strategy.should_buy/sell signature)
    current_market_data = {
        "symbol": test_symbol,
        "price": 60000.0,
        "change_24h": -1.5,
        "timestamp": datetime.now()
    }
    mock_engine.current_market_data[test_symbol] = current_market_data

    # --- Test Buy Signal ---
    # The should_buy method itself will manage fetching and processing both timeframes
    buy_signal = await ml_strategy.should_buy(current_market_data, mock_engine.positions.get(test_symbol, {}))
    logger.info(f"Buy Signal: {buy_signal}")

    if buy_signal.signal_type == SignalType.BUY and buy_signal.confidence > 0.5:
        logger.info(f"ML Strategy Test: Generated a BUY signal. Attempting mock order.")
        quantity = buy_signal.quantity if (buy_signal.quantity is not None and buy_signal.quantity > 0) else 0.001
        await mock_engine.place_order(test_symbol, "buy", quantity)
        logger.info(f"Mock Balance after buy: {mock_engine.balance:.2f}")
    else:
        logger.info(f"ML Strategy Test: No BUY signal generated. (Reason: {buy_signal.reason})")

    # --- Test Sell Signal (if a position was opened) ---
    current_position_after_buy = mock_engine.positions.get(test_symbol, {})

    if test_symbol in mock_engine.positions and mock_engine.positions[test_symbol]['amount'] > 0:
        current_market_data_sell = current_market_data.copy()
        current_market_data_sell['price'] = mock_engine.positions[test_symbol]['entry_price'] * 1.03 # Simulate 3% profit
            
        sell_signal = await ml_strategy.should_sell(current_market_data_sell, current_position_after_buy)
        logger.info(f"Sell Signal: {sell_signal}")

        if sell_signal.signal_type == SignalType.SELL and sell_signal.confidence > 0.5:
            logger.info(f"ML Strategy Test: Generated a SELL signal. Attempting mock order.")
            quantity_to_sell = sell_signal.quantity if (sell_signal.quantity is not None and sell_signal.quantity > 0) else mock_engine.positions[test_symbol]['amount']
            await mock_engine.place_order(test_symbol, "sell", quantity_to_sell)
            logger.info(f"Mock Balance after sell: {mock_engine.balance:.2f}")
        else:
            logger.info(f"ML Strategy Test: No SELL signal generated. (Reason: {sell_signal.reason})")
    else:
        logger.info("ML Strategy Test: No position to test SELL signal.")
        
    logger.info("\n=== ML Strategy Test Completed ===")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML training pipeline or specific tests.")
    parser.add_argument('--full-train', action='store_true', help="Run the full ML training pipeline across multiple timeframes.")
    parser.add_argument('--test-ml', action='store_true', help="Run basic ML training test (single timeframe).")
    parser.add_argument('--test-strategy', action='store_true', help="Run basic ML strategy test (multi-timeframe).")
    
    args = parser.parse_args()

    if args.full_train:
        asyncio.run(run_training_pipeline())
    elif args.test_ml:
        asyncio.run(test_ml_training_pipeline())
    elif args.test_strategy:
        asyncio.run(test_ml_strategy())
    else:
        logger.info("No specific task requested. Use --full-train, --test-ml, or --test-strategy.")
        logger.info("Example: python train_models.py --full-train")