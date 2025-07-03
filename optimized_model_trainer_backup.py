#!/usr/bin/env python3
"""
Enhanced Trading Bot - Production Ready Version
FILE LOCATION: E:\Trade Chat Bot\G Trading Bot\optimized_model_trainer.py

DIRECTORY STRUCTURE:
E:\Trade Chat Bot\G Trading Bot\
├── optimized_model_trainer.py  ← THIS FILE (replaces existing)
├── compatible_model_trainer.py ← Keep existing
├── trading_bot.log            ← Auto-created log file
├── models\                    ← Auto-created directory
│   ├── BTC\
│   │   ├── USD_1h\           ← Auto-created subdirectories
│   │   ├── USD_4h\
│   │   └── USD_1d\
│   ├── ETH\
│   └── ADA\
└── (venv)\                   ← Your existing virtual environment

All critical issues fixed:
1. Unicode logging errors resolved
2. Model save path creation added
3. Kraken rate limiting with smart delays
4. Fixed LSTM attention layer architecture
"""

import os
import sys
import time
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Trading imports
import ccxt

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# FIX 1: UNICODE LOGGING SETUP
# =============================================================================
def setup_logging():
    """Setup logging with proper Unicode support for Windows"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create handlers with UTF-8 encoding
    handlers = []
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Set encoding explicitly for Windows compatibility
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except:
            pass
    
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler with UTF-8 encoding
    log_file = Path('trading_bot.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# FIX 2: DIRECTORY CREATION UTILITY
# =============================================================================
def ensure_directory_exists(path: str) -> bool:
    """Ensure directory exists, create if needed"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False

# =============================================================================
# FIX 3: RATE LIMITED KRAKEN API WRAPPER
# =============================================================================
class RateLimitedKraken:
    """Kraken API wrapper with intelligent rate limiting"""
    
    def __init__(self):
        self.exchange = ccxt.kraken({
            'rateLimit': 1000,  # Base rate limit in ms
            'enableRateLimit': True,
        })
        self.last_request_time = 0
        self.request_count = 0
        self.min_delay = 1.0  # Minimum delay between requests
        self.adaptive_delay = 1.0  # Adaptive delay based on errors
        
    def fetch_ohlcv_with_retry(self, symbol: str, timeframe: str, limit: int = 720, 
                               since: Optional[int] = None, max_retries: int = 3) -> List:
        """Fetch OHLCV data with intelligent retry and rate limiting"""
        for attempt in range(max_retries):
            try:
                # Adaptive delay based on previous errors
                current_delay = max(self.min_delay, self.adaptive_delay)
                
                # Ensure minimum time between requests
                time_since_last = time.time() - self.last_request_time
                if time_since_last < current_delay:
                    sleep_time = current_delay - time_since_last
                    logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                
                # Make the API call
                self.last_request_time = time.time()
                self.request_count += 1
                
                result = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                # Success - reduce adaptive delay
                self.adaptive_delay = max(0.5, self.adaptive_delay * 0.9)
                logger.debug(f"API call successful, adaptive delay: {self.adaptive_delay:.2f}s")
                
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if 'too many requests' in error_msg or 'rate limit' in error_msg:
                    # Exponential backoff for rate limiting
                    self.adaptive_delay = min(10.0, self.adaptive_delay * 2.0)
                    retry_delay = self.adaptive_delay * (attempt + 1)
                    
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), "
                                   f"waiting {retry_delay:.2f}s. Adaptive delay: {self.adaptive_delay:.2f}s")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff for other errors
                        continue
                    else:
                        raise e
        
        raise Exception(f"Failed to fetch data after {max_retries} attempts")

# =============================================================================
# ENHANCED DATA FETCHER
# =============================================================================
class EnhancedDataFetcher:
    """Enhanced data fetcher with rate limiting and caching"""
    
    def __init__(self):
        self.kraken = RateLimitedKraken()
        self.cache = {}
        logger.info("Enhanced Data Fetcher initialized with rate-limited Kraken")
        
    def fetch_real_data(self, symbol: str, timeframe: str, limit: int = 720) -> pd.DataFrame:
        """Fetch real market data with intelligent chunking and rate limiting"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        if cache_key in self.cache:
            logger.info(f"Using cached data for {symbol} {timeframe}")
            return self.cache[cache_key]
        
        logger.info(f"Fetching {limit} candles for {symbol} ({timeframe})")
        
        # Calculate chunk size based on timeframe to respect rate limits
        chunk_sizes = {'1h': 720, '4h': 720, '1d': 720}
        chunk_size = chunk_sizes.get(timeframe, 720)
        
        all_data = []
        chunks_needed = max(1, (limit + chunk_size - 1) // chunk_size)
        
        for chunk_idx in range(chunks_needed):
            try:
                # Calculate since timestamp for this chunk
                since = None
                if chunk_idx > 0:
                    # Calculate offset based on timeframe
                    tf_minutes = {'1h': 60, '4h': 240, '1d': 1440}
                    minutes_offset = tf_minutes.get(timeframe, 60) * chunk_size * chunk_idx
                    since = int((datetime.now() - timedelta(minutes=minutes_offset)).timestamp() * 1000)
                
                chunk_data = self.kraken.fetch_ohlcv_with_retry(
                    symbol, timeframe, chunk_size, since
                )
                
                if chunk_data:
                    all_data.extend(chunk_data)
                    logger.debug(f"Fetched chunk {chunk_idx + 1}/{chunks_needed}: {len(chunk_data)} candles")
                
            except Exception as e:
                logger.error(f"Error fetching chunk {chunk_idx + 1}: {e}")
                continue
        
        if not all_data:
            raise Exception(f"No data retrieved for {symbol} {timeframe}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        
        # Keep only the most recent 'limit' candles
        df = df.tail(limit)
        
        # Data quality checks
        self._validate_data(df, symbol)
        
        # Cache the result
        self.cache[cache_key] = df
        logger.info(f"Cached {len(df)} candles")
        logger.info(f"Successfully fetched {len(df)} candles for {symbol} ({timeframe})")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, symbol: str):
        """Validate data quality and detect anomalies"""
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # Check for extreme price movements (>50% change)
                pct_change = df[col].pct_change().abs()
                extreme_moves = (pct_change > 0.5).sum()
                if extreme_moves > 0:
                    logger.warning(f"Extreme price movements detected in {col} for {symbol}: {extreme_moves} occurrences")

# =============================================================================
# FIX 4: CORRECTED LSTM ATTENTION ARCHITECTURE
# =============================================================================
class FixedLSTMModel:
    """LSTM model with properly implemented attention mechanism"""
    
    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self) -> Sequential:
        """Build LSTM model with corrected attention layer usage"""
        model = Sequential()
        
        # First LSTM layer - return sequences for attention
        model.add(LSTM(64, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        
        # Second LSTM layer - return sequences for attention
        model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.2))
        
        # Attention mechanism - using MultiHeadAttention (more stable)
        # Note: MultiHeadAttention expects (batch, seq_len, features)
        attention_layer = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            name='attention'
        )
        
        # Apply attention (query=value=key for self-attention)
        class AttentionWrapper(tf.keras.layers.Layer):
            def __init__(self, attention_layer, **kwargs):
                super().__init__(**kwargs)
                self.attention = attention_layer
                
            def call(self, inputs):
                # Self-attention: query, value, and key are all the same
                attended = self.attention(inputs, inputs, inputs)
                return attended
        
        model.add(AttentionWrapper(attention_layer))
        
        # Global average pooling to reduce sequence dimension
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        
        # Final dense layers
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train the LSTM model"""
        if self.model is None:
            self.build_model()
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        return {
            'model': self.model,
            'history': history.history,
            'final_val_accuracy': max(history.history['val_accuracy'])
        }

# =============================================================================
# ENHANCED ML ENGINE
# =============================================================================
class EnhancedAdaptiveMLEngine:
    """Enhanced ML engine with fixed LSTM and comprehensive ensemble"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        logger.info("Enhanced Adaptive ML Engine initialized")
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare comprehensive technical features"""
        features = []
        
        # Price features
        features.extend([
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values
        ])
        
        # Technical indicators
        # Moving averages
        for window in [5, 10, 20]:
            ma = df['close'].rolling(window=window).mean()
            features.append(ma.fillna(method='bfill').values)
        
        # Price ratios
        features.append((df['high'] / df['low']).values)
        features.append((df['close'] / df['open']).values)
        
        # Returns
        features.append(df['close'].pct_change().fillna(0).values)
        
        # Volatility (rolling std of returns)
        volatility = df['close'].pct_change().rolling(window=10).std()
        features.append(volatility.fillna(volatility.mean()).values)
        
        return np.column_stack(features)
    
    def create_sequences(self, data: np.ndarray, labels: np.ndarray, 
                        sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(labels[i])
        return np.array(X), np.array(y)
    
    def train_ensemble(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble of models with fixed LSTM"""
        logger.info(f"Training ensemble models for {symbol} ({timeframe})")
        
        # Prepare features and labels
        features = self.prepare_features(df)
        
        # Create binary labels (1 if next close > current close, 0 otherwise)
        labels = (df['close'].shift(-1) > df['close']).astype(int).values[:-1]
        features = features[:-1]  # Remove last row to match labels
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        results = {}
        
        # 1. Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            results['random_forest'] = {'model': rf, 'accuracy': rf_accuracy}
            logger.info(f"Random Forest accuracy: {rf_accuracy:.4f}")
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
        
        # 2. Gradient Boosting
        try:
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb.fit(X_train, y_train)
            gb_pred = gb.predict(X_test)
            gb_accuracy = accuracy_score(y_test, gb_pred)
            results['gradient_boosting'] = {'model': gb, 'accuracy': gb_accuracy}
            logger.info(f"Gradient Boosting accuracy: {gb_accuracy:.4f}")
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")
        
        # 3. Fixed LSTM with Attention
        try:
            # Scale features for LSTM
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Create sequences
            X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val)
            X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test)
            
            if len(X_train_seq) > 0:
                # Build and train LSTM
                lstm_model = FixedLSTMModel(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
                lstm_results = lstm_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
                
                # Evaluate on test set
                lstm_pred = (lstm_model.model.predict(X_test_seq) > 0.5).astype(int)
                lstm_accuracy = accuracy_score(y_test_seq, lstm_pred)
                
                results['lstm_attention'] = {
                    'model': lstm_model.model,
                    'scaler': scaler,
                    'accuracy': lstm_accuracy
                }
                logger.info(f"LSTM with Attention accuracy: {lstm_accuracy:.4f}")
            else:
                logger.warning("Not enough data for LSTM sequence creation")
                
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
        
        return results

# =============================================================================
# OPTIMIZED MODEL TRAINER WITH ALL FIXES
# =============================================================================
class OptimizedModelTrainer:
    """Optimized model trainer with all fixes applied"""
    
    def __init__(self, symbols: List[str], timeframes: List[str]):
        self.symbols = symbols
        self.timeframes = timeframes
        
        # Initialize components
        try:
            self.data_fetcher = EnhancedDataFetcher()
            logger.info("CHECK EnhancedDataFetcher initialized")  # Emoji removed
        except Exception as e:
            logger.error(f"Failed to initialize data fetcher: {e}")
            raise
        
        try:
            self.ml_engine = EnhancedAdaptiveMLEngine()
            logger.info("CHECK AdaptiveMLEngine initialized")  # Emoji removed
        except Exception as e:
            logger.error(f"Failed to initialize ML engine: {e}")
            raise
        
        # Ensure models directory exists
        self.models_dir = Path("models")
        ensure_directory_exists(str(self.models_dir))
        
        logger.info(f"OptimizedModelTrainer initialized for {len(symbols)} symbols, {len(timeframes)} timeframes")
    
    def fetch_real_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch real market data with error handling"""
        try:
            logger.info(f"Fetching real data for {symbol} {timeframe}")
            data = self.data_fetcher.fetch_real_data(symbol, timeframe)
            logger.info(f"CHECK Fetched {len(data)} candles for {symbol} {timeframe}")  # Emoji removed
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol} {timeframe}: {e}")
            raise
    
    def save_models(self, symbol: str, timeframe: str, models: Dict[str, Any]) -> bool:
        """Save trained models with proper directory creation"""
        try:
            # Create symbol-specific directory
            symbol_clean = symbol.replace('/', '')  # Remove slash for directory name
            model_dir = self.models_dir / symbol_clean / f"{timeframe}"
            
            # Ensure directory exists
            if not ensure_directory_exists(str(model_dir)):
                raise Exception(f"Could not create model directory: {model_dir}")
            
            # Save each model
            saved_models = {}
            for model_name, model_data in models.items():
                try:
                    model_file = model_dir / f"{model_name}.pkl"
                    
                    if model_name == 'lstm_attention':
                        # Save TensorFlow model separately with proper extension
                        tf_model_file = model_dir / f"{model_name}_model.keras"
                        model_data['model'].save(str(tf_model_file))
                        
                        # Save scaler and metadata
                        metadata = {
                            'scaler': model_data['scaler'],
                            'accuracy': model_data['accuracy'],
                            'model_path': str(tf_model_file)
                        }
                        with open(model_file, 'wb') as f:
                            pickle.dump(metadata, f)
                    else:
                        # Save sklearn models
                        with open(model_file, 'wb') as f:
                            pickle.dump(model_data, f)
                    
                    saved_models[model_name] = str(model_file)
                    logger.debug(f"Saved {model_name} to {model_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to save {model_name}: {e}")
                    continue
            
            # Save summary
            summary_file = model_dir / "training_summary.json"
            summary = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'models': saved_models,
                'accuracies': {k: v.get('accuracy', 0) for k, v in models.items()}
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Models saved successfully to {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models for {symbol} {timeframe}: {e}")
            return False
    
    def train_symbol_timeframe(self, symbol: str, timeframe: str, verbose: bool = False) -> Dict[str, Any]:
        """Train models for a specific symbol and timeframe"""
        try:
            # Fetch data
            data = self.fetch_real_data(symbol, timeframe)
            
            if verbose:
                date_range = f"{data.index[0].strftime('%Y-%m-%d %H:%M:%S')} to {data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}"
                logger.info(f"   Data range: {date_range}")
                print(f"   📊 Using {len(data)} data points")
                print(f"   📅 Range: {date_range}")
            
            # Train models
            models = self.ml_engine.train_ensemble(symbol, timeframe, data)
            
            if not models:
                raise Exception("No models were successfully trained")
            
            # Save models
            if self.save_models(symbol, timeframe, models):
                return {
                    'success': True,
                    'models': list(models.keys()),
                    'accuracies': {k: v.get('accuracy', 0) for k, v in models.items()},
                    'data_points': len(data)
                }
            else:
                raise Exception("Failed to save models")
                
        except Exception as e:
            error_msg = f"Error training {symbol} {timeframe}: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': str(e)}
    
    def train_symbol_all_timeframes(self, symbol: str, verbose: bool = False) -> Dict[str, Any]:
        """Train models for all timeframes of a symbol"""
        print(f"\n📈 Training {symbol} across {len(self.timeframes)} timeframes...")
        
        results = {}
        successful = 0
        
        for timeframe in self.timeframes:
            print(f"🔄 Training {symbol} {timeframe}...")
            
            result = self.train_symbol_timeframe(symbol, timeframe, verbose)
            results[timeframe] = result
            
            if result.get('success', False):
                successful += 1
                accuracies = result.get('accuracies', {})
                acc_str = ", ".join([f"{k}: {v:.4f}" for k, v in accuracies.items()])
                print(f"   ✅ {timeframe}: Success ({acc_str})")
            else:
                error = result.get('error', 'Unknown error')
                print(f"   ❌ {timeframe}: {error}")
        
        print(f"   📊 {symbol} summary: {successful}/{len(self.timeframes)} timeframes successful")
        
        return {
            'symbol': symbol,
            'successful': successful,
            'total': len(self.timeframes),
            'results': results
        }
    
    def train_all_symbols(self, verbose: bool = False) -> Dict[str, Any]:
        """Train models for all symbols and timeframes"""
        logger.info("ROCKET Starting optimized training for all symbols")  # Emoji removed
        
        print("🚀 Enhanced Trading Bot - Optimized Training")
        print("=" * 55)
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(f"⏰ Timeframes: {', '.join(self.timeframes)}")
        print(f"🎯 Total combinations: {len(self.symbols)} × {len(self.timeframes)} = {len(self.symbols) * len(self.timeframes)}")
        
        all_results = {}
        total_successful = 0
        total_combinations = len(self.symbols) * len(self.timeframes)
        
        for i, symbol in enumerate(self.symbols, 1):
            print(f"\n[{i}/{len(self.symbols)}] Processing {symbol}...")
            
            symbol_result = self.train_symbol_all_timeframes(symbol, verbose)
            all_results[symbol] = symbol_result
            total_successful += symbol_result['successful']
        
        # Final summary
        success_rate = (total_successful / total_combinations) * 100
        
        print("\n" + "=" * 55)
        print("📊 Training Summary:")
        print(f"   🎯 Total successful: {total_successful}/{total_combinations} combinations")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        print()
        print("📋 Per-symbol results:")
        for symbol, result in all_results.items():
            success_pct = (result['successful'] / result['total']) * 100
            print(f"   {symbol}: {result['successful']}/{result['total']} ({success_pct:.0f}%)")
        
        logger.info(f"CHECK Training completed: {total_successful}/{total_combinations} successful")  # Emoji removed
        
        return {
            'total_successful': total_successful,
            'total_combinations': total_combinations,
            'success_rate': success_rate,
            'symbol_results': all_results
        }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function with proper argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Trading Bot - Production Ready')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USD', 'ETH/USD', 'ADA/USD'],
                       help='Trading symbols to train on')
    parser.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'],
                       help='Timeframes to train on')
    parser.add_argument('--full-train', action='store_true',
                       help='Train all symbols and timeframes')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = OptimizedModelTrainer(args.symbols, args.timeframes)
        
        if args.full_train:
            # Train all combinations
            results = trainer.train_all_symbols(args.verbose)
            
            # Show final results
            if results['total_successful'] > 0:
                print(f"\n🎉 Training completed successfully!")
                print(f"📊 {results['total_successful']} out of {results['total_combinations']} models trained")
                print(f"📈 Success rate: {results['success_rate']:.1f}%")
            else:
                print(f"\n⚠️  No models were successfully trained.")
                print("Please check the error logs above for details.")
        else:
            print("Use --full-train to start training all symbols and timeframes")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\n⏹️  Training stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())