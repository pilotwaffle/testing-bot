#!/usr/bin/env python3
"""
compatible_model_trainer.py - Compatible Model Trainer

This trainer works with your existing class structure and doesn't assume 
specific method names. It adapts to whatever methods are available.
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.feature_selection import SelectKBest, f_classif
    import joblib
    sklearn_available = True
except ImportError:
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    sklearn_available = False

# Import from core modules
try:
    from core.enhanced_data_fetcher import EnhancedDataFetcher
    data_fetcher_available = True
except ImportError as e:
    print(f"Warning: Could not import EnhancedDataFetcher: {e}")
    data_fetcher_available = False

try:
    from core.enhanced_ml_engine import AdaptiveMLEngine
    ml_engine_available = True
except ImportError as e:
    print(f"Warning: Could not import AdaptiveMLEngine: {e}")
    ml_engine_available = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/compatible_trainer.log')
    ]
)
logger = logging.getLogger(__name__)

class CompatibleModelTrainer:
    """
    Model trainer that adapts to your existing class structure
    """
    
    def __init__(self, symbols: List[str] = None, model_save_path: str = "models/"):
        """Initialize the compatible trainer"""
        self.symbols = symbols or ['BTC/USD', 'ETH/USD', 'ADA/USD']
        self.model_save_path = model_save_path
        self.models = {}
        self.scalers = {}
        
        # Ensure directories exist
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Try to initialize data fetcher
        self.data_fetcher = None
        if data_fetcher_available:
            try:
                self.data_fetcher = EnhancedDataFetcher()
                logger.info("EnhancedDataFetcher initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize EnhancedDataFetcher: {e}")
        
        # Try to initialize ML engine
        self.ml_engine = None
        if ml_engine_available:
            try:
                self.ml_engine = AdaptiveMLEngine()
                logger.info("AdaptiveMLEngine initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize AdaptiveMLEngine: {e}")
        
        logger.info(f"CompatibleModelTrainer initialized for symbols: {self.symbols}")
    
    def generate_sample_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Generate sample data for testing"""
        logger.info(f"Generating sample data for {symbol}")
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducible results
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1
        
        # Random walk with trend
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Add some noise for OHLCV
            noise = np.random.normal(0, 0.01, 4)
            open_price = price * (1 + noise[0])
            high_price = price * (1 + abs(noise[1]))
            low_price = price * (1 - abs(noise[2]))
            close_price = price * (1 + noise[3])
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': max(open_price, high_price, low_price, close_price),
                'low': min(open_price, high_price, low_price, close_price),
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} rows of sample data for {symbol}")
        return df
    
    def try_fetch_real_data(self, symbol: str) -> pd.DataFrame:
        """Try to fetch real data using various method names"""
        if not self.data_fetcher:
            return pd.DataFrame()
        
        # Try different possible method names
        method_names = [
            'fetch_ohlcv',
            'get_ohlcv', 
            'fetch_data',
            'get_data',
            'fetch_candles',
            'get_candles',
            'fetch_historical_data',
            'get_historical_data'
        ]
        
        for method_name in method_names:
            if hasattr(self.data_fetcher, method_name):
                try:
                    logger.info(f"Trying to fetch data using {method_name}")
                    method = getattr(self.data_fetcher, method_name)
                    
                    # Try different parameter combinations
                    try:
                        data = method(symbol, '1h', 2000)
                    except:
                        try:
                            data = method(symbol, '1h')
                        except:
                            try:
                                data = method(symbol)
                            except:
                                continue
                    
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        logger.info(f"Successfully fetched data using {method_name}")
                        return data
                        
                except Exception as e:
                    logger.debug(f"Method {method_name} failed: {e}")
                    continue
        
        logger.info("No working data fetch method found")
        return pd.DataFrame()
    
    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data with fallback to sample data"""
        try:
            # Try to get real data
            data = self.try_fetch_real_data(symbol)
            
            if not data.empty:
                return data
            else:
                logger.info(f"No real data available for {symbol}, using sample data")
                return self.generate_sample_data(symbol)
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self.generate_sample_data(symbol)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for ML training"""
        try:
            data = df.copy()
            
            # Basic price features
            data['price_change'] = data['close'].pct_change()
            data['high_low_ratio'] = data['high'] / data['low']
            data['volume_change'] = data['volume'].pct_change()
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
                data[f'price_sma_{period}_ratio'] = data['close'] / data[f'sma_{period}']
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            data['macd'] = exp1 - exp2
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            # Volatility
            data['volatility'] = data['close'].rolling(window=20).std()
            
            # Volume indicators
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            
            # Drop original OHLCV columns and NaN values
            feature_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            features_df = data[feature_columns].dropna()
            
            logger.info(f"Created {len(feature_columns)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def create_target(self, df: pd.DataFrame, prediction_horizon: int = 1) -> pd.Series:
        """Create target variable for classification"""
        try:
            # Calculate future returns
            future_returns = df['close'].shift(-prediction_horizon) / df['close'] - 1
            
            # Create binary target: 1 if positive return, 0 otherwise
            target = (future_returns > 0.01).astype(int)  # 1% threshold
            
            return target[:-prediction_horizon]
            
        except Exception as e:
            logger.error(f"Error creating target: {e}")
            return pd.Series()
    
    def train_sklearn_models(self, symbol: str, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """Train scikit-learn models"""
        if not sklearn_available:
            return {'error': 'scikit-learn not available'}
        
        results = {}
        
        try:
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Feature selection
            selector = SelectKBest(f_classif, k=min(20, X_train.shape[1]))
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # Train models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            trained_models = {}
            scores = {}
            
            for name, model in models.items():
                try:
                    model.fit(X_train_selected, y_train)
                    score = model.score(X_test_selected, y_test)
                    
                    trained_models[name] = model
                    scores[name] = score
                    
                    logger.info(f"{name} trained with score: {score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
            
            # Save models
            self.models[symbol] = trained_models
            self.scalers[symbol] = {'scaler': scaler, 'selector': selector}
            
            # Save to disk
            symbol_path = os.path.join(self.model_save_path, symbol.replace('/', '_'))
            os.makedirs(symbol_path, exist_ok=True)
            
            for name, model in trained_models.items():
                joblib.dump(model, os.path.join(symbol_path, f'{name}.pkl'))
            
            joblib.dump(scaler, os.path.join(symbol_path, 'scaler.pkl'))
            joblib.dump(selector, os.path.join(symbol_path, 'selector.pkl'))
            
            best_model = max(scores, key=scores.get) if scores else None
            best_score = scores.get(best_model, 0.0) if best_model else 0.0
            
            results = {
                'symbol': symbol,
                'best_model': best_model,
                'best_score': best_score,
                'all_scores': scores,
                'models_saved': len(trained_models)
            }
            
            logger.info(f"Training completed for {symbol}. Best: {best_model} ({best_score:.3f})")
            return results
            
        except Exception as e:
            logger.error(f"Error in sklearn training for {symbol}: {e}")
            return {'error': str(e)}
    
    def train_symbol(self, symbol: str, verbose: bool = False) -> Dict[str, Any]:
        """Train models for a specific symbol"""
        logger.info(f"Training models for {symbol}")
        
        try:
            # Fetch data
            data = self.fetch_data(symbol)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            if verbose:
                print(f"📊 Training {symbol} with {len(data)} data points")
            
            # Create features and target
            features = self.create_features(data)
            target = self.create_target(data)
            
            if features.empty or target.empty:
                raise ValueError("No features or target data available")
            
            # Align features and target
            min_len = min(len(features), len(target))
            X = features.iloc[:min_len]
            y = target.iloc[:min_len]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train models
            result = self.train_sklearn_models(symbol, X_train, X_test, y_train, y_test)
            
            if verbose and 'error' not in result:
                print(f"✅ {symbol} training completed:")
                print(f"   Best model: {result['best_model']}")
                print(f"   Best score: {result['best_score']:.3f}")
                print(f"   Models saved: {result['models_saved']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error training {symbol}: {e}"
            logger.error(error_msg)
            return {'error': error_msg}
    
    def train_all_symbols(self, verbose: bool = False) -> Dict[str, Any]:
        """Train models for all symbols"""
        logger.info("Starting training for all symbols")
        
        if verbose:
            print("🚀 Compatible Trading Bot - Model Training")
            print("=" * 50)
            print(f"Training symbols: {', '.join(self.symbols)}")
            print(f"Model save path: {self.model_save_path}")
            print(f"Sklearn available: {sklearn_available}")
            print()
        
        results = {}
        
        for i, symbol in enumerate(self.symbols, 1):
            if verbose:
                print(f"[{i}/{len(self.symbols)}] Training {symbol}...")
            
            try:
                result = self.train_symbol(symbol, verbose)
                results[symbol] = result
                
                if verbose and 'error' not in result:
                    print(f"✅ {symbol} completed successfully!")
                elif verbose:
                    print(f"❌ {symbol} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_msg = f"Failed to train {symbol}: {e}"
                logger.error(error_msg)
                results[symbol] = {'error': error_msg}
                
                if verbose:
                    print(f"❌ {symbol} failed: {error_msg}")
            
            if verbose:
                print()
        
        # Summary
        successful = sum(1 for r in results.values() if 'error' not in r)
        total = len(results)
        
        if verbose:
            print("📊 Training Summary:")
            print(f"   Successfully trained: {successful}/{total} symbols")
            print(f"   Models saved to: {self.model_save_path}")
            
            if successful > 0:
                print("\n🎯 Best performing models:")
                for symbol, result in results.items():
                    if 'error' not in result and result.get('best_model'):
                        print(f"   {symbol}: {result['best_model']} (Score: {result['best_score']:.3f})")
        
        logger.info(f"Training completed: {successful}/{total} symbols successful")
        return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compatible Trading Bot Model Trainer')
    parser.add_argument('--symbols', nargs='+', default=['BTC/USD', 'ETH/USD', 'ADA/USD'],
                       help='Trading symbols to train on')
    parser.add_argument('--model-path', default='models/',
                       help='Directory to save trained models')
    parser.add_argument('--full-train', action='store_true',
                       help='Train models for all symbols')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CompatibleModelTrainer(args.symbols, args.model_path)
    
    if args.full_train:
        # Train all symbols
        results = trainer.train_all_symbols(args.verbose)
    else:
        print("Use --full-train to start training, or --help for options")

if __name__ == "__main__":
    main()