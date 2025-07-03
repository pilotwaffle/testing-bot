# enhanced_ml_engine.py - Enhanced ML Engine with Self-Learning Capabilities
"""
Enhanced ML Engine with Self-Learning Capabilities
Implements adaptive learning, better architectures, and performance monitoring
"""

import numpy as np
import pandas as pd
import tensorflow as tf
try:
    from tensorflow try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    keras = None
    KERAS_AVAILABLE = False
    print("WARNING: Keras not available - using fallback")
    KERAS_AVAILABLE = True
except ImportError:
    keras = None
    KERAS_AVAILABLE = False
    print("WARNING: Keras not available - using fallback")
from tensorflow.keras import layers
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class AdaptiveMLEngine:
    def __init__(self, model_save_path: str = "models/", performance_log_path: str = "logs/"):
        self.model_save_path = Path(model_save_path)
        self.performance_log_path = Path(performance_log_path)
        self.model_save_path.mkdir(exist_ok=True)
        self.performance_log_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_sets = {}
        self.performance_history = {}
        self.prediction_errors = {}
        
        # Adaptive learning parameters
        self.learning_rate_decay = 0.95
        self.performance_threshold = 0.6
        self.retrain_threshold = 0.05  # Retrain if performance drops by 5%
        self.min_samples_for_retrain = 100
        
        self.logger.info("Enhanced Adaptive ML Engine initialized")
    
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators and features"""
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        
        # Moving averages and trends
        for period in [5, 10, 20, 50, 100]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_ma_{period}_ratio'] = df['close'] / df[f'ma_{period}']
            
        # Volatility features
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']
        
        # RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['rsi'] = calculate_rsi(df['close'])
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_volume'] = df['close'] * df['volume']
            
        # Support/Resistance levels
        df['local_high'] = df['high'].rolling(10, center=True).max()
        df['local_low'] = df['low'].rolling(10, center=True).min()
        df['distance_to_high'] = (df['local_high'] - df['close']) / df['close']
        df['distance_to_low'] = (df['close'] - df['local_low']) / df['close']
        
        # Market structure
        df['higher_high'] = ((df['high'] > df['high'].shift(1)) & 
                            (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['lower_low'] = ((df['low'] < df['low'].shift(1)) & 
                          (df['low'].shift(1) < df['low'].shift(2))).astype(int)
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
            
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
            
        return df
    
    def create_prediction_targets(self, data: pd.DataFrame, prediction_horizon: int = 1) -> pd.DataFrame:
        """Create multiple prediction targets"""
        df = data.copy()
        
        # Price direction (classification)
        df['future_return'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
        df['price_direction'] = (df['future_return'] > 0).astype(int)
        
        # Volatility prediction (regression)
        df['future_volatility'] = df['returns'].shift(-prediction_horizon).rolling(5).std()
        
        # Risk level (classification)
        volatility_threshold = df['volatility_20'].quantile(0.7)
        df['high_risk'] = (df['volatility_20'] > volatility_threshold).astype(int)
        
        # Trend strength
        df['trend_strength'] = abs(df['returns'].rolling(10).mean()) / df['volatility_10']
        df['strong_trend'] = (df['trend_strength'] > df['trend_strength'].quantile(0.7)).astype(int)
        
        return df
    
    def prepare_sequences(self, data: pd.DataFrame, sequence_length: int = 60, 
                         target_column: str = 'price_direction') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training"""
        # Select feature columns (exclude target and future data)
        feature_cols = [col for col in data.columns if not col.startswith('future_') 
                       and col not in ['price_direction', 'high_risk', 'strong_trend']]
        
        # Remove rows with NaN values
        clean_data = data[feature_cols + [target_column]].dropna()
        
        if len(clean_data) < sequence_length + 10:
            raise ValueError(f"Insufficient data: {len(clean_data)} rows, need at least {sequence_length + 10}")
        
        # Normalize features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(clean_data[feature_cols])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(clean_data[target_column].iloc[i])
        
        return np.array(X), np.array(y), scaler, feature_cols
    
    def build_advanced_lstm(self, input_shape: Tuple[int, int], 
                           model_type: str = 'classification') -> keras.Model:
        """Build advanced LSTM architecture with attention"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # LSTM layers with dropout
        model.add(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.BatchNormalization())
        
        model.add(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(layers.BatchNormalization())
        
        # Attention mechanism
        model.add(layers.LSTM(32, return_sequences=True))
        model.add(layers.Attention())
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.2))
        
        # Output layer
        if model_type == 'classification':
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
        else:
            model.add(layers.Dense(1, activation='linear'))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def train_ensemble_model(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble of models with cross-validation"""
        self.logger.info(f"Training ensemble models for {symbol} ({timeframe})")
        
        # Feature engineering
        enhanced_data = self.create_advanced_features(data)
        target_data = self.create_prediction_targets(enhanced_data)
        
        results = {}
        
        # Time series split for validation
        train_size = int(len(target_data) * 0.7)
        val_size = int(len(target_data) * 0.15)
        
        train_data = target_data.iloc[:train_size]
        val_data = target_data.iloc[train_size:train_size + val_size]
        test_data = target_data.iloc[train_size + val_size:]
        
        self.logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Train LSTM for price direction
        try:
            X_train, y_train, scaler, feature_cols = self.prepare_sequences(
                train_data, sequence_length=60, target_column='price_direction'
            )
            X_val, y_val, _, _ = self.prepare_sequences(
                val_data, sequence_length=60, target_column='price_direction'
            )
            
            # Build and train LSTM
            lstm_model = self.build_advanced_lstm(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                model_type='classification'
            )
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10, restore_best_weights=True
            )
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001
            )
            
            # Train model
            history = lstm_model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate
            val_predictions = (lstm_model.predict(X_val) > 0.5).astype(int)
            val_accuracy = accuracy_score(y_val, val_predictions)
            
            results['lstm'] = {
                'model': lstm_model,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'accuracy': val_accuracy,
                'history': history.history
            }
            
            self.logger.info(f"LSTM validation accuracy: {val_accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            results['lstm'] = None
        
        # Train Random Forest ensemble
        try:
            feature_cols = [col for col in train_data.columns if not col.startswith('future_') 
                           and col not in ['price_direction', 'high_risk', 'strong_trend']]
            
            X_train_rf = train_data[feature_cols].fillna(0)
            y_train_rf = train_data['price_direction']
            X_val_rf = val_data[feature_cols].fillna(0)
            y_val_rf = val_data['price_direction']
            
            # Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_rf, y_train_rf)
            rf_pred = rf_model.predict(X_val_rf)
            rf_accuracy = accuracy_score(y_val_rf, rf_pred)
            
            # Gradient Boosting
            gb_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            gb_model.fit(X_train_rf, y_train_rf)
            gb_pred = gb_model.predict(X_val_rf)
            gb_accuracy = accuracy_score(y_val_rf, gb_pred)
            
            results['random_forest'] = {
                'model': rf_model,
                'feature_cols': feature_cols,
                'accuracy': rf_accuracy,
                'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_))
            }
            
            results['gradient_boosting'] = {
                'model': gb_model,
                'feature_cols': feature_cols,
                'accuracy': gb_accuracy,
                'feature_importance': dict(zip(feature_cols, gb_model.feature_importances_))
            }
            
            self.logger.info(f"Random Forest accuracy: {rf_accuracy:.4f}")
            self.logger.info(f"Gradient Boosting accuracy: {gb_accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}")
        
        # Store results
        model_key = f"{symbol}_{timeframe}"
        self.models[model_key] = results
        self.performance_history[model_key] = {
            'training_date': datetime.now().isoformat(),
            'data_points': len(target_data),
            'validation_scores': {k: v['accuracy'] for k, v in results.items() if v and 'accuracy' in v}
        }
        
        # Save models
        self.save_models(symbol, timeframe, results)
        
        return results
    
    def save_models(self, symbol: str, timeframe: str, models: Dict[str, Any]):
        """Save all models and metadata"""
        model_dir = self.model_save_path / f"{symbol}_{timeframe}"
        model_dir.mkdir(exist_ok=True)
        
        for model_name, model_data in models.items():
            if model_data is None:
                continue
                
            if model_name == 'lstm':
                # Save LSTM model
                model_data['model'].save(model_dir / f"lstm_model.keras")
                joblib.dump(model_data['scaler'], model_dir / f"lstm_scaler.joblib")
                
                # Save metadata
                metadata = {
                    'feature_cols': model_data['feature_cols'],
                    'accuracy': model_data['accuracy'],
                    'training_date': datetime.now().isoformat()
                }
                with open(model_dir / f"lstm_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            else:
                # Save sklearn models
                joblib.dump(model_data['model'], model_dir / f"{model_name}_model.joblib")
                
                metadata = {
                    'feature_cols': model_data['feature_cols'],
                    'accuracy': model_data['accuracy'],
                    'feature_importance': model_data.get('feature_importance', {}),
                    'training_date': datetime.now().isoformat()
                }
                with open(model_dir / f"{model_name}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Models saved for {symbol} ({timeframe})")
    
    def predict_ensemble(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using ensemble of models"""
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.models:
            self.logger.warning(f"No models found for {symbol} ({timeframe})")
            return None
        
        # Prepare data
        enhanced_data = self.create_advanced_features(data)
        
        predictions = {}
        confidence_scores = []
        
        models = self.models[model_key]
        
        # LSTM prediction
        if models.get('lstm'):
            try:
                lstm_data = models['lstm']
                X_seq, _, _, _ = self.prepare_sequences(
                    enhanced_data, sequence_length=60, target_column='price_direction'
                )
                
                if len(X_seq) > 0:
                    lstm_pred = lstm_data['model'].predict(X_seq[-1:])
                    predictions['lstm'] = float(lstm_pred[0][0])
                    confidence_scores.append(lstm_data['accuracy'])
                    
            except Exception as e:
                self.logger.error(f"LSTM prediction failed: {e}")
        
        # Random Forest prediction
        if models.get('random_forest'):
            try:
                rf_data = models['random_forest']
                X_rf = enhanced_data[rf_data['feature_cols']].fillna(0).iloc[-1:].values
                rf_pred = rf_data['model'].predict_proba(X_rf)[0][1]
                predictions['random_forest'] = float(rf_pred)
                confidence_scores.append(rf_data['accuracy'])
                
            except Exception as e:
                self.logger.error(f"Random Forest prediction failed: {e}")
        
        # Gradient Boosting prediction
        if models.get('gradient_boosting'):
            try:
                gb_data = models['gradient_boosting']
                X_gb = enhanced_data[gb_data['feature_cols']].fillna(0).iloc[-1:].values
                gb_pred = gb_data['model'].predict_proba(X_gb)[0][1]
                predictions['gradient_boosting'] = float(gb_pred)
                confidence_scores.append(gb_data['accuracy'])
                
            except Exception as e:
                self.logger.error(f"Gradient Boosting prediction failed: {e}")
        
        if not predictions:
            return None
        
        # Weighted ensemble prediction
        weights = np.array(confidence_scores)
        weights = weights / np.sum(weights)
        
        ensemble_prediction = np.average(list(predictions.values()), weights=weights)
        
        return {
            'individual_predictions': predictions,
            'ensemble_prediction': float(ensemble_prediction),
            'confidence': float(np.mean(confidence_scores)),
            'timestamp': datetime.now().isoformat()
        }
    
    def track_prediction_error(self, symbol: str, timeframe: str, prediction: float, 
                              actual: float, timestamp: datetime):
        """Track prediction errors for adaptive learning"""
        model_key = f"{symbol}_{timeframe}"
        
        if model_key not in self.prediction_errors:
            self.prediction_errors[model_key] = []
        
        error = {
            'timestamp': timestamp.isoformat(),
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual),
            'accuracy': 1.0 - abs(prediction - actual)
        }
        
        self.prediction_errors[model_key].append(error)
        
        # Keep only recent errors (last 1000 predictions)
        if len(self.prediction_errors[model_key]) > 1000:
            self.prediction_errors[model_key] = self.prediction_errors[model_key][-1000:]
        
        # Check if retraining is needed
        self._check_retrain_trigger(model_key)
    
    def _check_retrain_trigger(self, model_key: str):
        """Check if model needs retraining based on recent performance"""
        if len(self.prediction_errors[model_key]) < self.min_samples_for_retrain:
            return False
        
        recent_errors = self.prediction_errors[model_key][-50:]  # Last 50 predictions
        recent_accuracy = np.mean([e['accuracy'] for e in recent_errors])
        
        # Get historical performance
        if model_key in self.performance_history:
            historical_accuracy = np.mean(list(self.performance_history[model_key]['validation_scores'].values()))
            
            if recent_accuracy < historical_accuracy - self.retrain_threshold:
                self.logger.warning(f"Performance degradation detected for {model_key}: "
                                  f"Recent: {recent_accuracy:.4f}, Historical: {historical_accuracy:.4f}")
                return True
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {}
        
        for model_key, errors in self.prediction_errors.items():
            if not errors:
                continue
                
            recent_errors = errors[-100:]  # Last 100 predictions
            
            summary[model_key] = {
                'total_predictions': len(errors),
                'recent_accuracy': np.mean([e['accuracy'] for e in recent_errors]),
                'recent_avg_error': np.mean([e['error'] for e in recent_errors]),
                'historical_performance': self.performance_history.get(model_key, {}),
                'last_prediction': errors[-1]['timestamp'] if errors else None
            }
        
        return summary