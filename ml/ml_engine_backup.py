# ml/ml_engine.py
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib # For saving/loading scikit-learn models & scalers

from typing import Dict, Any, Optional, List 

try:
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout, Bidirectional
    from keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow/Keras not installed. Neural Network models will not be available.")

from core.config import settings

logger = logging.getLogger(__name__)

# Define model extensions for different types
MODEL_EXTENSIONS = {
    "neural_network": ".keras",
    "lorentzian": ".joblib",
    "risk_assessment": ".joblib",
    "scaler": "_scaler.joblib",
    "model_metadata": "_metadata.json"
}

class OctoBotMLEngine:
    def __init__(self, engine_reference=None):
        logger.info("Initializing OctoBot ML Engine.")
        self._engine_reference = engine_reference
        self.models = {}
        self.scalers = {}
        self.model_features = {} # NEW: Stores the ordered feature names used for each trained model

        self.model_save_path = settings.DEFAULT_MODEL_SAVE_PATH
        os.makedirs(self.model_save_path, exist_ok=True)
        logger.info(f"Model save path: {self.model_save_path}")

        if not TF_AVAILABLE:
            logger.error("TensorFlow/Keras is not available. Please install it to use Neural Network models.")
        else:
            logger.info("TensorFlow/Keras available for advanced Neural Networks.")

    def _get_model_name(self, symbol: str, model_type: str, timeframe: str = "") -> str:
        """Generates a standardized filename for models."""
        clean_symbol = symbol.replace('/', '_').replace(':', '_')
        timeframe_suffix = f"_{timeframe}" if timeframe else ""
        return f"{model_type}_{clean_symbol}{timeframe_suffix}"

    def _get_model_path(self, symbol: str, model_type: str, timeframe: str = "") -> str:
        """Returns the full path for a model file."""
        model_name = self._get_model_name(symbol, model_type, timeframe)
        extension = MODEL_EXTENSIONS.get(model_type, ".bin")
        return os.path.join(self.model_save_path, f"{model_name}{extension}")
    
    def _get_scaler_path(self, symbol: str, timeframe: str = "") -> str:
        """Returns the full path for a scaler file."""
        clean_symbol = symbol.replace('/', '_').replace(':', '_')
        timeframe_suffix = f"_{timeframe}" if timeframe else ""
        scaler_name = f"{clean_symbol}{timeframe_suffix}{MODEL_EXTENSIONS['scaler']}"
        return os.path.join(self.model_save_path, scaler_name)

    def _get_metadata_path(self, symbol: str, model_type: str, timeframe: str = "") -> str:
        """Returns the full path for a model's metadata file."""
        model_name = self._get_model_name(symbol, model_type, timeframe)
        return os.path.join(self.model_save_path, f"{model_name}{MODEL_EXTENSIONS['model_metadata']}")

    def load_models(self):
        """Loads all models and scalers from the saved directory, and associated feature names."""
        self.models = {}
        self.scalers = {}
        self.model_features = {} # Reset
        logger.info(f"Attempting to load models from: {self.model_save_path}")

        for filename in os.listdir(self.model_save_path):
            file_path = os.path.join(self.model_save_path, filename)
            
            # --- Load Scalers ---
            if filename.endswith(MODEL_EXTENSIONS['scaler']):
                try:
                    parts = filename.replace(MODEL_EXTENSIONS['scaler'], '').split('_')
                    symbol_part = f"{parts[-3]}/{parts[-2]}".upper() if len(parts) >= 3 else "_".join(parts[:-1]).upper()
                    timeframe_part = parts[-1] if len(parts) >= 3 else ""
                    
                    if not symbol_part: 
                        raise ValueError(f"Could not parse symbol from scaler filename: {filename}")

                    scaler = joblib.load(file_path)
                    self.scalers[(symbol_part, timeframe_part)] = scaler
                    logger.debug(f"Loaded scaler for {symbol_part} ({timeframe_part}).")
                except Exception as e:
                    logger.error(f"Failed to load scaler {filename}: {e}")
                continue

            # --- Load Models & Metadata ---
            # Try to load metadata first to get feature names for consistency
            if filename.endswith(MODEL_EXTENSIONS['model_metadata']):
                try:
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                        symbol_part = metadata.get('symbol')
                        timeframe_part = metadata.get('timeframe', '')
                        model_type_part = metadata.get('model_type')
                        features_used = metadata.get('features')

                        if symbol_part and model_type_part and features_used is not None:
                            self.model_features[(symbol_part, timeframe_part, model_type_part)] = features_used
                            logger.debug(f"Loaded metadata (features) for {model_type_part} {symbol_part} ({timeframe_part}).")
                        else:
                            logger.warning(f"Metadata file {filename} is incomplete. Missing symbol, model_type, or features.")
                except Exception as e:
                    logger.error(f"Failed to load metadata {filename}: {e}")
                continue 

        # Now load the actual models, relying on metadata for feature names
        for filename in os.listdir(self.model_save_path):
            file_path = os.path.join(self.model_save_path, filename)

            for model_type, ext in MODEL_EXTENSIONS.items():
                if model_type in ["scaler", "model_metadata"]: continue
                if filename.endswith(ext):
                    try:
                        base_name = filename.replace(ext, '')
                        parts = base_name.split('_')
                        
                        actual_model_type_prefix = ""
                        for known_type in MODEL_EXTENSIONS.keys():
                            if base_name.startswith(known_type) and known_type not in ["scaler", "model_metadata"]:
                                actual_model_type_prefix = known_type
                                break
                        
                        if not actual_model_type_prefix: continue 

                        remaining_parts_str = base_name[len(actual_model_type_prefix) + 1:]
                        remaining_parts = remaining_parts_str.split('_')
                        
                        symbol_part = ""
                        timeframe_part = ""

                        if len(remaining_parts) >= 3:
                            symbol_part = f"{remaining_parts[-3]}/{remaining_parts[-2]}".upper()
                            timeframe_part = remaining_parts[-1]
                        elif len(remaining_parts) == 2:
                            symbol_part = f"{remaining_parts[-2]}/{remaining_parts[-1]}".upper()
                            timeframe_part = ""
                        else:
                             symbol_part = remaining_parts_str.replace('_', '/') 
                             timeframe_part = ""

                        model = None
                        if actual_model_type_prefix == "neural_network" and TF_AVAILABLE:
                            model = load_model(file_path)
                        else:
                            model = joblib.load(file_path)
                        
                        self.models[(symbol_part, timeframe_part, actual_model_type_prefix)] = model
                        logger.info(f"Loaded {actual_model_type_prefix} for {symbol_part} ({timeframe_part}).")
                        break 

                    except Exception as e:
                        logger.error(f"Failed to load model {filename} for type {model_type}: {e}")
        
        logger.info(f"Total models loaded: {len(self.models)}")
        logger.info(f"Total scalers loaded: {len(self.scalers)}")
        logger.info(f"Total model feature sets loaded: {len(self.model_features)}")


    # --- Feature Engineering & Preprocessing ---
    # These functions now define the EXACT features they expect and return that exact list.
    _NN_FEATURES = ['open', 'high', 'low', 'close', 'volume', 
                    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',   
                    'RSI', 'MACD', 'Signal_Line', 'MACD_Hist', 
                    'Middle_Band', 'Upper_Band', 'Lower_Band']

    _LORENTZIAN_FEATURES = ['close', 'volume', 'SMA_20', 'RSI', 'MACD', 'MACD_Hist']

    _RISK_FEATURES = ['close', 'Daily_Return', 'Volatility', 'SMA_20', 'RSI']


    def _prepare_data_for_nn(self, df: pd.DataFrame, symbol: str, timeframe: str, is_training: bool = True):
        # 1. Select the exact features from the incoming DataFrame
        features_df = pd.DataFrame(index=df.index) 
        for feature_name in self._NN_FEATURES:
            if feature_name in df.columns:
                features_df[feature_name] = df[feature_name]
            else:
                features_df[feature_name] = 0.0 
        
        features_df.fillna(0, inplace=True)
        features_df.replace([np.inf, -np.inf], 0, inplace=True)

        target = pd.Series()
        if is_training:
            target = (df['close'].shift(-1) > df['close']).astype(int)
            features_df = features_df.iloc[:-1]
            target = target.iloc[:-1]

        scaler_key = (symbol, timeframe)
        if is_training:
            scaler = StandardScaler() 
            scaled_features = scaler.fit_transform(features_df)
            self.scalers[scaler_key] = scaler
            joblib.dump(scaler, self._get_scaler_path(symbol, timeframe))
            logger.info(f"Scaler trained and saved for {symbol} ({timeframe}).")
            self.model_features[(symbol, timeframe, "neural_network")] = self._NN_FEATURES 
        else:
            scaler = self.scalers.get(scaler_key)
            if scaler:
                loaded_features = self.model_features.get((symbol, timeframe, "neural_network"))
                if loaded_features: 
                    features_to_transform = features_df[loaded_features]
                    features_to_transform.fillna(0, inplace=True)
                    features_to_transform.replace([np.inf, -np.inf], 0, inplace=True)
                    scaled_features = scaler.transform(features_to_transform)
                else:
                    logger.warning(f"No stored feature names for NN {symbol} ({timeframe}). Cannot apply scaler correctly.")
                    scaled_features = features_df.values
            else:
                logger.warning(f"No scaler found for {symbol} ({timeframe}). Skipping scaling for prediction.")
                scaled_features = features_df.values

        num_features = scaled_features.shape[1]
        X = np.array(scaled_features).reshape(-1, 1, num_features)
        y = np.array(target) if is_training else np.array([]) 
        
        return X, y, self._NN_FEATURES 

    def _prepare_data_for_lorentzian(self, df: pd.DataFrame, symbol: str, timeframe: str, is_training: bool = True):
        features_df = pd.DataFrame(index=df.index)
        for feature_name in self._LORENTZIAN_FEATURES:
            if feature_name in df.columns:
                features_df[feature_name] = df[feature_name]
            else:
                features_df[feature_name] = 0.0 

        features_df.fillna(0, inplace=True)
        features_df.replace([np.inf, -np.inf], 0, inplace=True)

        target = pd.Series()
        if is_training:
            target = (df['close'].shift(-1) > df['close']).astype(int)
            features_df = features_df.iloc[:-1]
            target = target.iloc[:-1]

        scaler_key = (symbol, timeframe)
        if is_training:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df)
            self.scalers[scaler_key] = scaler
            joblib.dump(scaler, self._get_scaler_path(symbol, timeframe))
            self.model_features[(symbol, timeframe, "lorentzian")] = self._LORENTZIAN_FEATURES 
        else:
            scaler = self.scalers.get(scaler_key)
            if scaler:
                loaded_features = self.model_features.get((symbol, timeframe, "lorentzian"))
                if loaded_features:
                    features_to_transform = features_df[loaded_features]
                    features_to_transform.fillna(0, inplace=True)
                    features_to_transform.replace([np.inf, -np.inf], 0, inplace=True)
                    scaled_features = scaler.transform(features_to_transform)
                else:
                    logger.warning(f"No stored feature names for Lorentzian {symbol} ({timeframe}). Cannot apply scaler correctly.")
                    scaled_features = features_df.values
            else:
                logger.warning(f"No scaler found for {symbol} ({timeframe}). Skipping scaling for prediction.")
                scaled_features = features_df.values

        y = np.array(target) if is_training else np.array([])
        return scaled_features, y, self._LORENTZIAN_FEATURES

    def _prepare_data_for_risk(self, df: pd.DataFrame, symbol: str, timeframe: str, is_training: bool = True):
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['close'].pct_change()
        if 'Volatility' not in df.columns:
            vol_window = 20 if timeframe == '1d' else 24 * 20 
            df['Volatility'] = df['Daily_Return'].rolling(window=vol_window).std()
            
        features_df = pd.DataFrame(index=df.index)
        for feature_name in self._RISK_FEATURES:
            if feature_name in df.columns:
                features_df[feature_name] = df[feature_name]
            else:
                features_df[feature_name] = 0.0 

        features_df.fillna(0, inplace=True)
        features_df.replace([np.inf, -np.inf], 0, inplace=True)

        target = pd.Series()
        if is_training:
            target = (features_df['Daily_Return'] < -0.01).astype(int).shift(-1) 
            features_df = features_df.iloc[:-1]
            target = target.iloc[:-1]

        scaler_key = (symbol, timeframe)
        if is_training:
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features_df)
            self.scalers[scaler_key] = scaler
            joblib.dump(scaler, self._get_scaler_path(symbol, timeframe))
            self.model_features[(symbol, timeframe, "risk_assessment")] = self._RISK_FEATURES 
        else:
            scaler = self.scalers.get(scaler_key)
            if scaler:
                loaded_features = self.model_features.get((symbol, timeframe, "risk_assessment"))
                if loaded_features:
                    features_to_transform = features_df[loaded_features]
                    features_to_transform.fillna(0, inplace=True)
                    features_to_transform.replace([np.inf, -np.inf], 0, inplace=True)
                    scaled_features = scaler.transform(features_to_transform)
                else:
                    logger.warning(f"No stored feature names for Risk {symbol} ({timeframe}). Cannot apply scaler correctly.")
                    scaled_features = features_df.values
            else:
                logger.warning(f"No scaler found for {symbol} ({timeframe}). Skipping scaling for prediction.")
                scaled_features = features_df.values
        
        y = np.array(target) if is_training else np.array([])
        return scaled_features, y, self._RISK_FEATURES
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def train_neural_network(self, symbol: str, df: pd.DataFrame, timeframe: str):
        if not TF_AVAILABLE:
            return {"success": False, "error": "TensorFlow not available."}
        
        logger.info(f"Starting Neural Network training for {symbol} ({timeframe})...")
        try:
            X, y, feature_names = self._prepare_data_for_nn(df, symbol, timeframe, is_training=True)
            if X.shape[0] == 0 or y.shape[0] == 0:
                logger.error("Insufficient data after preprocessing steps for NN. Check feature requirements.")
                return {"success": False, "error": "Insufficient data after preprocessing."}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)
            
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"NN model for {symbol} ({timeframe}) - Test Accuracy: {accuracy:.4f}")

            model_path = self._get_model_path(symbol, "neural_network", timeframe)
            model.save(model_path)
            self.models[(symbol, timeframe, "neural_network")] = model
            logger.info(f"Neural Network model saved to {model_path}")
            
            metadata = {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": "neural_network",
                "accuracy": accuracy,
                "features": feature_names,
                "trained_date": datetime.now().isoformat()
            }
            with open(self._get_metadata_path(symbol, "neural_network", timeframe), 'w') as f:
                json.dump(metadata, f, indent=4)

            return {"success": True, "accuracy": accuracy, "loss": loss}

        except Exception as e:
            logger.error(f"Error training Neural Network for {symbol} ({timeframe}): {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def train_lorentzian_classifier(self, symbol: str, df: pd.DataFrame, timeframe: str):
        from sklearn.linear_model import LogisticRegression
        logger.info(f"Starting Lorentzian Classifier training for {symbol} ({timeframe})...")
        try:
            X, y, feature_names = self._prepare_data_for_lorentzian(df, symbol, timeframe, is_training=True)
            if X.shape[0] == 0 or y.shape[0] == 0:
                 logger.error("Insufficient data after preprocessing steps for Lorentzian. Check feature requirements.")
                 return {"success": False, "error": "Insufficient data after preprocessing."}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Lorentzian model for {symbol} ({timeframe}) - Test Accuracy: {accuracy:.4f}")

            model_path = self._get_model_path(symbol, "lorentzian", timeframe)
            joblib.dump(model, model_path)
            self.models[(symbol, timeframe, "lorentzian")] = model
            logger.info(f"Lorentzian model saved to {model_path}")

            metadata = {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": "lorentzian",
                "accuracy": accuracy,
                "features": feature_names,
                "trained_date": datetime.now().isoformat()
            }
            with open(self._get_metadata_path(symbol, "lorentzian", timeframe), 'w') as f:
                json.dump(metadata, f, indent=4)

            return {"success": True, "accuracy": accuracy}

        except Exception as e:
            logger.error(f"Error training Lorentzian Classifier for {symbol} ({timeframe}): {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def train_risk_assessment_model(self, symbol: str, df: pd.DataFrame, timeframe: str):
        from sklearn.ensemble import RandomForestClassifier
        logger.info(f"Starting Risk Assessment model training for {symbol} ({timeframe})...")
        try:
            X, y, feature_names = self._prepare_data_for_risk(df, symbol, timeframe, is_training=True)
            if X.shape[0] == 0 or y.shape[0] == 0:
                logger.error("Insufficient data after preprocessing steps for Risk Assessment. Check feature requirements.")
                return {"success": False, "error": "Insufficient data after preprocessing."}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Risk Assessment model for {symbol} ({timeframe}) - Test Accuracy: {accuracy:.4f}")

            model_path = self._get_model_path(symbol, "risk_assessment", timeframe)
            joblib.dump(model, model_path)
            self.models[(symbol, timeframe, "risk_assessment")] = model
            logger.info(f"Risk Assessment model saved to {model_path}")

            metadata = {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": "risk_assessment",
                "accuracy": accuracy,
                "features": feature_names,
                "trained_date": datetime.now().isoformat()
            }
            with open(self._get_metadata_path(symbol, "risk_assessment", timeframe), 'w') as f:
                json.dump(metadata, f, indent=4)

            return {"success": True, "accuracy": accuracy}

        except Exception as e:
            logger.error(f"Error training Risk Assessment model for {symbol} ({timeframe}): {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def predict_neural_network(self, symbol: str, latest_data: dict, timeframe: str):
        if not TF_AVAILABLE:
            return {"action": "NONE", "confidence": 0, "error": "TensorFlow not available."}

        model_key = (symbol, timeframe, "neural_network")
        model = self.models.get(model_key)
        
        if not model:
            logger.warning(f"Neural Network model not found for {symbol} ({timeframe}).")
            return {"action": "NONE", "confidence": 0, "error": "Model not found."}
        
        scaler = self.scalers.get((symbol, timeframe))
        if not scaler:
            logger.warning(f"Scaler not found for {symbol} ({timeframe}). Prediction might be inconsistent or inaccurate.")
        
        try:
            input_df = pd.DataFrame([latest_data]) 
            X_pred, _, _ = self._prepare_data_for_nn(input_df, symbol, timeframe, is_training=False)
           
            if X_pred.shape[0] == 0:
                return {"action": "NONE", "confidence": 0, "error": "Insufficient data for prediction after preprocessing."}

            prediction_raw = model.predict(X_pred, verbose=0)[0][0]
            
            action = "BUY" if prediction_raw > 0.5 else "SELL"
            confidence = abs(prediction_raw - 0.5) * 2
            
            confidence = max(0.0, min(1.0, confidence))

            logger.info(f"NN Prediction for {symbol} ({timeframe}): {action} with confidence {confidence:.4f} (raw: {prediction_raw:.4f})")
            return {"action": action, "confidence": confidence, "raw_prediction": float(prediction_raw)}

        except Exception as e:
            logger.error(f"Error predicting with Neural Network for {symbol} ({timeframe}): {e}", exc_info=True)
            return {"action": "NONE", "confidence": 0, "error": str(e)}

    def predict_lorentzian(self, symbol: str, latest_data: dict, timeframe: str):
        model_key = (symbol, timeframe, "lorentzian")
        model = self.models.get(model_key)
        
        if not model:
            logger.warning(f"Lorentzian model not found for {symbol} ({timeframe}).")
            return {"action": "NONE", "confidence": 0, "error": "Model not found."}
        
        scaler = self.scalers.get((symbol, timeframe))
        if not scaler:
            logger.warning(f"Scaler not found for {symbol} ({timeframe}). Prediction might be inconsistent or inaccurate.")

        try:
            input_df = pd.DataFrame([latest_data]) 
            X_pred, _, _ = self._prepare_data_for_lorentzian(input_df, symbol, timeframe, is_training=False)

            if X_pred.shape[0] == 0:
                return {"action": "NONE", "confidence": 0, "error": "Insufficient data for prediction after preprocessing."}

            prediction_class = model.predict(X_pred)[0]
            prediction_proba = model.predict_proba(X_pred)[0]

            action = "BUY" if prediction_class == 1 else "SELL"
            confidence = prediction_proba.max()

            logger.info(f"Lorentzian Prediction for {symbol} ({timeframe}): {action} with confidence {confidence:.4f}")
            return {"action": action, "confidence": confidence, "raw_prediction": int(prediction_class)}

        except Exception as e:
            logger.error(f"Error predicting with Lorentzian Classifier for {symbol} ({timeframe}): {e}", exc_info=True)
            return {"action": "NONE", "confidence": 0, "error": str(e)}

    def evaluate_risk(self, symbol: str, latest_data: dict, timeframe: str):
        model_key = (symbol, timeframe, "risk_assessment")
        model = self.models.get(model_key)
        
        if not model:
            logger.warning(f"Risk Assessment model not found for {symbol} ({timeframe}).")
            return {"risk_level": "UNKNOWN", "confidence": 0, "error": "Model not found."}
        
        scaler = self.scalers.get((symbol, timeframe))
        if not scaler:
            logger.warning(f"Scaler not found for {symbol} ({timeframe}). prediction might be inconsistent or inaccurate.")

        try:
            input_df = pd.DataFrame([latest_data]) 
            X_pred, _, _ = self._prepare_data_for_risk(input_df, symbol, timeframe, is_training=False)

            if X_pred.shape[0] == 0:
                return {"risk_level": "UNKNOWN", "confidence": 0, "error": "Insufficient data for prediction after preprocessing."}

            risk_class = model.predict(X_pred)[0]
            risk_proba = model.predict_proba(X_pred)[0]

            risk_level = "HIGH" if risk_class == 1 else "LOW"
            confidence = risk_proba.max()

            logger.info(f"Risk Assessment for {symbol} ({timeframe}): {risk_level} with confidence {confidence:.4f}")
            return {"risk_level": risk_level, "confidence": confidence, "raw_prediction": int(risk_class)}

        except Exception as e:
            logger.error(f"Error evaluating risk for {symbol} ({timeframe}): {e}", exc_info=True)
            return {"risk_level": "UNKNOWN", "confidence": 0, "error": str(e)}

    def get_model_status(self) -> Dict[str, Any]:
        status = {}
        for (symbol, timeframe, model_type), model_obj in self.models.items():
            model_id = f"{model_type}_{symbol.replace('/', '_')}_{timeframe}"
            
            metadata_path = self._get_metadata_path(symbol, model_type, timeframe)
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_id}: {e}")

            accuracy = metadata.get('accuracy', 'N/A')
            r2_score = metadata.get('r2_score', 'N/A')
            
            metric_label = "Accuracy" if accuracy != 'N/A' else "R2 Score"
            metric_value = accuracy if accuracy != 'N/A' else r2_score

            scaler_loaded = (symbol, timeframe) in self.scalers
            features_loaded = (symbol, timeframe, model_type) in self.model_features

            status[model_id] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": model_type,
                "loaded": True,
                "metric_label": metric_label,
                "metric_value": metric_value,
                "features_used": self.model_features.get((symbol, timeframe, model_type), []),
                "trained_date": metadata.get('trained_date', 'N/A'),
                "scaler_loaded": scaler_loaded,
                "description": metadata.get('description', 'No description provided.'),
                "raw_metadata": metadata 
            }
        
        if not status:
            logger.info("No ML models currently loaded.")
            
        return status