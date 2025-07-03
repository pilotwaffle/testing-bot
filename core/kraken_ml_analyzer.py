# Imports added by quick_fix_script.py for Elite Trading Bot V3.0
# Location: E:\Trade Chat Bot\G Trading Bot\core\kraken_ml_analyzer.py
# Added missing type hints and standard imports

from typing import List, Dict, Optional, Union, Any
import asyncio
import logging
import json
import time
from datetime import datetime

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class KrakenMLAnalyzer:
    """
    Advanced ML Analyzer for Kraken Futures market data
    Integrated with EliteTradingEngine ML capabilities
    """
    
    def __init__(self, kraken_client, lookback_period: int = 100):
        self.kraken_client = kraken_client
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(f"{__name__}.KrakenML")
        
        # Model storage
        self.models = {
            'price_prediction': {},
            'volatility_prediction': {},
            'direction_prediction': {},
            'risk_assessment': {}
        }
        
        # Feature engineering parameters
        self.feature_params = {
            'sma_periods': [5, 10, 20, 50],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
        # Scalers for different features
        self.scalers = {
            'price': StandardScaler(),
            'volume': StandardScaler(),
            'technical': MinMaxScaler(),
            'orderbook': StandardScaler()
        }
        
        # Model performance tracking
        self.performance_metrics = {}
        
        # Real-time analysis cache
        self.analysis_cache = {}
        self.cache_ttl = 60  # 1 minute cache
        
    async def initialize(self):
        """Initialize ML analyzer"""
        try:
            self.logger.info("Initializing Kraken ML Analyzer...")
            
            # Load any existing models
            await self._load_existing_models()
            
            # Initialize TensorFlow if available
            if TENSORFLOW_AVAILABLE:
                tf.get_logger().setLevel('ERROR')  # Reduce TF logging
                self.logger.info("TensorFlow initialized for deep learning models")
            else:
                self.logger.warning("TensorFlow not available - using scikit-learn models only")
            
            self.logger.info("Kraken ML Analyzer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML analyzer: {e}")
            return False
    
    # ==================== FEATURE ENGINEERING ====================
    
    async def prepare_features(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Prepare comprehensive features for ML analysis"""
        try:
            # Get raw market data
            df = await self.kraken_client.get_ml_features(symbol, timeframe, self.lookback_period)
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Add advanced technical indicators
            df = self._add_technical_indicators(df)
            
            # Add market microstructure features
            df = await self._add_microstructure_features(df, symbol)
            
            # Add time-based features
            df = self._add_time_features(df)
            
            # Add volatility features
            df = self._add_volatility_features(df)
            
            # Add momentum features
            df = self._add_momentum_features(df)
            
            # Create target variables
            df = self._create_targets(df)
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            self.logger.info(f"Prepared {len(df)} samples with {len(df.columns)} features for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing features for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # Moving averages
            for period in self.feature_params['sma_periods']:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            
            # Exponential moving averages
            for period in self.feature_params['ema_periods']:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
            
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], self.feature_params['rsi_period'])
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(
                df['close'], 
                self.feature_params['macd_fast'],
                self.feature_params['macd_slow'],
                self.feature_params['macd_signal']
            )
            
            # Bollinger Bands
            df['bb_upper'], df['bb_lower'], df['bb_width'], df['bb_position'] = self._calculate_bollinger_bands_extended(
                df['close'], 
                self.feature_params['bb_period'],
                self.feature_params['bb_std']
            )
            
            # Williams %R
            df['williams_r'] = self._calculate_williams_r(df['high'], df['low'], df['close'])
            
            # Stochastic Oscillator
            df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df['high'], df['low'], df['close'])
            
            # Average True Range (ATR)
            df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
            
            # On Balance Volume (OBV)
            df['obv'] = self._calculate_obv(df['close'], df['volume'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    async def _add_microstructure_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add market microstructure features from orderbook"""
        try:
            # Get current orderbook
            orderbook = await self.kraken_client.get_orderbook(symbol)
            
            if orderbook:
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if bids and asks:
                    # Spread features
                    best_bid = float(bids[0][0]) if bids else 0
                    best_ask = float(asks[0][0]) if asks else 0
                    spread = best_ask - best_bid
                    mid_price = (best_ask + best_bid) / 2
                    
                    # Add to all rows (current market state)
                    df['spread'] = spread
                    df['spread_pct'] = (spread / mid_price) * 100 if mid_price > 0 else 0
                    df['mid_price'] = mid_price
                    
                    # Order book depth
                    bid_depth_5 = sum([float(bid[1]) for bid in bids[:5]])
                    ask_depth_5 = sum([float(ask[1]) for ask in asks[:5]])
                    
                    df['bid_depth_5'] = bid_depth_5
                    df['ask_depth_5'] = ask_depth_5
                    df['depth_imbalance'] = (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5) if (bid_depth_5 + ask_depth_5) > 0 else 0
                    
                    # Price impact estimates
                    df['price_impact_buy'] = self._calculate_price_impact(asks, 'buy')
                    df['price_impact_sell'] = self._calculate_price_impact(bids, 'sell')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding microstructure features: {e}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Market session features (assuming UTC times)
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding time features: {e}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
            # Historical volatility (different periods)
            for period in [5, 10, 20, 50]:
                returns = df['close'].pct_change()
                df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(period)
            
            # Parkinson volatility (high-low based)
            df['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(df['high'] / df['low']) ** 2).rolling(20).mean()
            )
            
            # Garman-Klass volatility
            df['gk_vol'] = np.sqrt(
                (np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
                 np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])).rolling(20).mean()
            )
            
            # Volatility rank
            df['vol_rank'] = df['volatility_20'].rolling(252).rank(pct=True)  # 252-period rank
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {e}")
            return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        try:
            # Rate of change for different periods
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = df['close'].pct_change(period)
            
            # Money Flow Index
            df['mfi'] = self._calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
            
            # Commodity Channel Index
            df['cci'] = self._calculate_cci(df['high'], df['low'], df['close'])
            
            # Momentum oscillator
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            
            # Price relative to recent high/low
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding momentum features: {e}")
            return df
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""
        try:
            # Price targets (future returns)
            for horizon in [1, 5, 10, 20]:  # 1h, 5h, 10h, 20h ahead
                df[f'return_{horizon}h'] = df['close'].shift(-horizon) / df['close'] - 1
                df[f'direction_{horizon}h'] = (df[f'return_{horizon}h'] > 0).astype(int)
            
            # Volatility targets
            returns = df['close'].pct_change()
            for horizon in [5, 10, 20]:
                df[f'future_vol_{horizon}h'] = returns.shift(-horizon).rolling(horizon).std()
            
            # High/Low targets
            for horizon in [5, 10, 20]:
                df[f'future_high_{horizon}h'] = df['high'].shift(-horizon).rolling(horizon).max()
                df[f'future_low_{horizon}h'] = df['low'].shift(-horizon).rolling(horizon).min()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating targets: {e}")
            return df
    
    # ==================== TECHNICAL INDICATOR CALCULATIONS ====================
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands_extended(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate extended Bollinger Bands with width and position"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        width = (upper - lower) / sma
        position = (prices - lower) / (upper - lower)
        return upper, lower, width, position
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume"""
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        return (direction * volume).cumsum()
    
    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        money_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + money_ratio))
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _calculate_price_impact(self, orders: List, side: str) -> float:
        """Calculate estimated price impact for order execution"""
        try:
            if not orders:
                return 0
            
            # Simulate 1 BTC order impact
            target_volume = 1.0
            cumulative_volume = 0
            total_cost = 0
            
            for price, volume in orders[:10]:  # Top 10 levels
                price = float(price)
                volume = float(volume)
                
                if cumulative_volume + volume >= target_volume:
                    remaining = target_volume - cumulative_volume
                    total_cost += remaining * price
                    break
                else:
                    total_cost += volume * price
                    cumulative_volume += volume
            
            if cumulative_volume > 0:
                avg_price = total_cost / min(target_volume, cumulative_volume)
                best_price = float(orders[0][0])
                return abs(avg_price - best_price) / best_price
            
            return 0
            
        except Exception:
            return 0
    
    # ==================== MODEL TRAINING ====================
    
    async def train_models(self, symbol: str, model_types: List[str] = None) -> Dict:
        """Train ML models for price prediction"""
        try:
            if model_types is None:
                model_types = ['random_forest', 'gradient_boosting', 'linear']
                if TENSORFLOW_AVAILABLE:
                    model_types.append('lstm')
            
            # Prepare data
            df = await self.prepare_features(symbol)
            
            if df.empty or len(df) < 50:
                raise Exception("Insufficient data for training")
            
            results = {}
            
            # Feature columns (exclude targets and metadata)
            feature_cols = [col for col in df.columns if not any(x in col for x in ['return_', 'direction_', 'future_', 'timestamp', 'symbol'])]
            
            # Train models for different prediction horizons
            for horizon in [1, 5, 10]:
                target_col = f'return_{horizon}h'
                direction_col = f'direction_{horizon}h'
                
                if target_col not in df.columns or direction_col not in df.columns:
                    continue
                
                # Prepare data
                X = df[feature_cols].fillna(0)
                y_reg = df[target_col].fillna(0)
                y_cls = df[direction_col].fillna(0)
                
                # Remove samples with missing targets
                valid_mask = ~(y_reg.isna() | y_cls.isna())
                X = X[valid_mask]
                y_reg = y_reg[valid_mask]
                y_cls = y_cls[valid_mask]
                
                if len(X) < 30:
                    continue
                
                # Scale features
                X_scaled = self.scalers['technical'].fit_transform(X)
                
                # Time series split
                tscv = TimeSeriesSplit(n_splits=3)
                horizon_results = {}
                
                for model_type in model_types:
                    try:
                        if model_type == 'random_forest':
                            model_results = await self._train_random_forest(X_scaled, y_reg, y_cls, tscv)
                        elif model_type == 'gradient_boosting':
                            model_results = await self._train_gradient_boosting(X_scaled, y_reg, y_cls, tscv)
                        elif model_type == 'linear':
                            model_results = await self._train_linear_model(X_scaled, y_reg, y_cls, tscv)
                        elif model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                            model_results = await self._train_lstm_model(X_scaled, y_reg, tscv)
                        else:
                            continue
                        
                        horizon_results[model_type] = model_results
                        
                        # Store best model
                        if symbol not in self.models['price_prediction']:
                            self.models['price_prediction'][symbol] = {}
                        
                        self.models['price_prediction'][symbol][f'{model_type}_{horizon}h'] = {
                            'model': model_results['model'],
                            'scaler': self.scalers['technical'],
                            'features': feature_cols,
                            'performance': model_results['performance'],
                            'trained_at': datetime.now()
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Error training {model_type} for {horizon}h: {e}")
                        continue
                
                results[f'{horizon}h'] = horizon_results
            
            # Save performance metrics
            self.performance_metrics[symbol] = results
            
            self.logger.info(f"Completed model training for {symbol}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {e}")
            return {}
    
    async def _train_random_forest(self, X: np.ndarray, y_reg: pd.Series, y_cls: pd.Series, tscv) -> Dict:
        """Train Random Forest models"""
        # Regression model
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Classification model
        rf_cls = RandomForestRegressor(  # Using regressor for probabilities
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross validation
        reg_scores = []
        cls_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_reg_train, y_reg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
            y_cls_train, y_cls_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
            
            # Train and evaluate regression
            rf_reg.fit(X_train, y_reg_train)
            reg_pred = rf_reg.predict(X_test)
            reg_scores.append(r2_score(y_reg_test, reg_pred))
            
            # Train and evaluate classification
            rf_cls.fit(X_train, y_cls_train)
            cls_pred = rf_cls.predict(X_test)
            cls_scores.append(((cls_pred > 0.5) == y_cls_test).mean())
        
        # Final training on full dataset
        rf_reg.fit(X, y_reg)
        rf_cls.fit(X, y_cls)
        
        return {
            'model': {'regression': rf_reg, 'classification': rf_cls},
            'performance': {
                'regression_r2': np.mean(reg_scores),
                'classification_accuracy': np.mean(cls_scores),
                'feature_importance': rf_reg.feature_importances_
            }
        }
    
    async def _train_gradient_boosting(self, X: np.ndarray, y_reg: pd.Series, y_cls: pd.Series, tscv) -> Dict:
        """Train Gradient Boosting models"""
        # Regression model
        gb_reg = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Classification model (using regressor for simplicity)
        gb_cls = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Cross validation
        reg_scores = []
        cls_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_reg_train, y_reg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
            y_cls_train, y_cls_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
            
            # Train and evaluate regression
            gb_reg.fit(X_train, y_reg_train)
            reg_pred = gb_reg.predict(X_test)
            reg_scores.append(r2_score(y_reg_test, reg_pred))
            
            # Train and evaluate classification
            gb_cls.fit(X_train, y_cls_train)
            cls_pred = gb_cls.predict(X_test)
            cls_scores.append(((cls_pred > 0.5) == y_cls_test).mean())
        
        # Final training
        gb_reg.fit(X, y_reg)
        gb_cls.fit(X, y_cls)
        
        return {
            'model': {'regression': gb_reg, 'classification': gb_cls},
            'performance': {
                'regression_r2': np.mean(reg_scores),
                'classification_accuracy': np.mean(cls_scores),
                'feature_importance': gb_reg.feature_importances_
            }
        }
    
    async def _train_linear_model(self, X: np.ndarray, y_reg: pd.Series, y_cls: pd.Series, tscv) -> Dict:
        """Train Linear models"""
        lr_reg = LinearRegression()
        lr_cls = LinearRegression()
        
        # Cross validation
        reg_scores = []
        cls_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_reg_train, y_reg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
            y_cls_train, y_cls_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
            
            lr_reg.fit(X_train, y_reg_train)
            reg_pred = lr_reg.predict(X_test)
            reg_scores.append(r2_score(y_reg_test, reg_pred))
            
            lr_cls.fit(X_train, y_cls_train)
            cls_pred = lr_cls.predict(X_test)
            cls_scores.append(((cls_pred > 0.5) == y_cls_test).mean())
        
        # Final training
        lr_reg.fit(X, y_reg)
        lr_cls.fit(X, y_cls)
        
        return {
            'model': {'regression': lr_reg, 'classification': lr_cls},
            'performance': {
                'regression_r2': np.mean(reg_scores),
                'classification_accuracy': np.mean(cls_scores)
            }
        }
    
    async def _train_lstm_model(self, X: np.ndarray, y_reg: pd.Series, tscv) -> Dict:
        """Train LSTM model (TensorFlow required)"""
        if not TENSORFLOW_AVAILABLE:
            raise Exception("TensorFlow not available for LSTM training")
        
        # Prepare sequence data
        sequence_length = 20
        X_seq, y_seq = self._prepare_sequences(X, y_reg.values, sequence_length)
        
        if len(X_seq) < 30:
            raise Exception("Insufficient data for LSTM training")
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train with early stopping
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        
        # Simple train/test split for sequences
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': {'regression': model},
            'performance': {
                'regression_r2': r2,
                'training_history': history.history
            }
        }
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    # ==================== PREDICTION ====================
    
    async def predict(self, symbol: str, horizon: str = '1h', model_type: str = 'random_forest') -> Dict:
        """Generate predictions using trained models"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{horizon}_{model_type}"
            if cache_key in self.analysis_cache:
                cache_data = self.analysis_cache[cache_key]
                if (time.time() - cache_data['timestamp']) < self.cache_ttl:
                    return cache_data['prediction']
            
            # Get model
            model_key = f'{model_type}_{horizon}'
            if (symbol not in self.models['price_prediction'] or 
                model_key not in self.models['price_prediction'][symbol]):
                
                # Train model if not exists
                await self.train_models(symbol, [model_type])
                
                if (symbol not in self.models['price_prediction'] or 
                    model_key not in self.models['price_prediction'][symbol]):
                    raise Exception(f"No trained model available for {symbol} {horizon} {model_type}")
            
            model_data = self.models['price_prediction'][symbol][model_key]
            model = model_data['model']
            scaler = model_data['scaler']
            features = model_data['features']
            
            # Prepare current features
            df = await self.prepare_features(symbol)
            
            if df.empty:
                raise Exception("No current data available for prediction")
            
            # Get latest features
            latest_features = df[features].iloc[-1:].fillna(0)
            X_scaled = scaler.transform(latest_features)
            
            # Make predictions
            if model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                # LSTM requires sequence input
                sequence_length = 20
                if len(df) >= sequence_length:
                    sequence_data = df[features].iloc[-sequence_length:].fillna(0)
                    X_seq = scaler.transform(sequence_data).reshape(1, sequence_length, -1)
                    price_pred = model['regression'].predict(X_seq)[0][0]
                    direction_pred = 1 if price_pred > 0 else 0
                else:
                    raise Exception("Insufficient data for LSTM prediction")
            else:
                # Standard ML models
                price_pred = model['regression'].predict(X_scaled)[0]
                direction_pred = model['classification'].predict(X_scaled)[0]
            
            # Get current price for context
            ticker = await self.kraken_client.get_ticker(symbol)
            current_price = ticker.get('last', 0) if ticker else 0
            
            # Calculate target price
            target_price = current_price * (1 + price_pred) if current_price > 0 else 0
            
            prediction = {
                'symbol': symbol,
                'horizon': horizon,
                'model_type': model_type,
                'timestamp': time.time(),
                'current_price': current_price,
                'predicted_return': price_pred,
                'predicted_price': target_price,
                'direction': 'bullish' if direction_pred > 0.5 else 'bearish',
                'direction_confidence': abs(direction_pred - 0.5) * 2,
                'model_performance': model_data['performance'],
                'features_used': len(features)
            }
            
            # Cache prediction
            self.analysis_cache[cache_key] = {
                'prediction': prediction,
                'timestamp': time.time()
            }
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction for {symbol}: {e}")
            return {}
    
    async def get_market_analysis(self, symbols: List[str] = None) -> Dict:
        """Get comprehensive market analysis across multiple symbols"""
        try:
            if not symbols:
                # Get top instruments
                instruments = await self.kraken_client.get_instruments()
                symbols = [inst['symbol'] for inst in instruments[:5]]  # Top 5
            
            analysis = {
                'timestamp': time.time(),
                'symbols_analyzed': len(symbols),
                'predictions': {},
                'market_summary': {
                    'bullish_signals': 0,
                    'bearish_signals': 0,
                    'neutral_signals': 0,
                    'avg_confidence': 0,
                    'high_confidence_signals': []
                }
            }
            
            total_confidence = 0
            valid_predictions = 0
            
            for symbol in symbols:
                try:
                    # Get predictions for different horizons
                    symbol_analysis = {}
                    
                    for horizon in ['1h', '5h', '10h']:
                        prediction = await self.predict(symbol, horizon, 'random_forest')
                        
                        if prediction:
                            symbol_analysis[horizon] = prediction
                            
                            # Update summary
                            direction = prediction.get('direction', 'neutral')
                            confidence = prediction.get('direction_confidence', 0)
                            
                            if direction == 'bullish':
                                analysis['market_summary']['bullish_signals'] += 1
                            elif direction == 'bearish':
                                analysis['market_summary']['bearish_signals'] += 1
                            else:
                                analysis['market_summary']['neutral_signals'] += 1
                            
                            total_confidence += confidence
                            valid_predictions += 1
                            
                            # High confidence signals
                            if confidence > 0.7:
                                analysis['market_summary']['high_confidence_signals'].append({
                                    'symbol': symbol,
                                    'horizon': horizon,
                                    'direction': direction,
                                    'confidence': confidence,
                                    'predicted_return': prediction.get('predicted_return', 0)
                                })
                    
                    if symbol_analysis:
                        analysis['predictions'][symbol] = symbol_analysis
                        
                except Exception as e:
                    self.logger.warning(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Calculate average confidence
            if valid_predictions > 0:
                analysis['market_summary']['avg_confidence'] = total_confidence / valid_predictions
            
            # Sort high confidence signals by confidence
            analysis['market_summary']['high_confidence_signals'].sort(
                key=lambda x: x['confidence'], reverse=True
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return {}
    
    # ==================== UTILITY METHODS ====================
    
    async def _load_existing_models(self):
        """Load any existing trained models"""
        try:
            # In a real implementation, load from disk
            # For now, initialize empty
            self.logger.info("Model storage initialized")
        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")
    
    async def get_model_performance(self, symbol: str) -> Dict:
        """Get performance metrics for trained models"""
        if symbol in self.performance_metrics:
            return self.performance_metrics[symbol]
        return {}
    
    async def retrain_models(self, symbol: str, force: bool = False) -> bool:
        """Retrain models if needed"""
        try:
            # Check if retraining is needed
            if symbol in self.models['price_prediction'] and not force:
                # Check model age
                for model_key, model_data in self.models['price_prediction'][symbol].items():
                    trained_at = model_data.get('trained_at', datetime.min)
                    if (datetime.now() - trained_at).days < 1:  # Retrain daily
                        return False
            
            # Retrain models
            results = await self.train_models(symbol)
            
            if results:
                self.logger.info(f"Successfully retrained models for {symbol}")
                return True
            else:
                self.logger.warning(f"Failed to retrain models for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error retraining models for {symbol}: {e}")
            return False