# strategies/ml_strategy.py
import logging
from typing import Dict, Any, Optional
import pandas as pd
from strategies.strategy_base import StrategyBase, Signal, SignalType

logger = logging.getLogger(__name__)

class MLStrategy(StrategyBase):
    DESCRIPTION = "A Machine Learning-driven strategy using multi-timeframe analysis."
    CONFIG_TEMPLATE = {
        "symbol": "BTC/USD", # CORRECTION: Changed default symbol for dashboard
        "model_type": "neural_network",
        "timeframes_config": {
            "1h": {
                "required_history": 200, 
                "prediction_threshold_buy": 0.55,
                "prediction_threshold_sell": 0.45,
            },
            "1d": {
                "required_history": 365, # More history for 1d model
                "prediction_threshold_buy": 0.5,
                "prediction_threshold_sell": 0.5,
            }
        },
        "take_profit_percent": 0.05,
        "stop_loss_percent": 0.02,
        "allocation_percent": 0.1,
        "single_position_only": True
    }

    def __init__(self, config: Dict[str, Any], ml_engine, data_fetcher):
        super().__init__(config)
        self.ml_engine = ml_engine
        self.data_fetcher = data_fetcher
        
        self.model_type = config.get("model_type", "neural_network")
        self.symbol = config.get("symbol")
        
        self.timeframes_config = config.get("timeframes_config", {})
        if not self.timeframes_config:
            logger.warning("No timeframes_config provided. Strategy might not function correctly.")
            self.timeframes_config = {
                "1h": {
                    "required_history": config.get("required_history", 200),
                    "prediction_threshold_buy": config.get("prediction_threshold_buy", 0.55),
                    "prediction_threshold_sell": config.get("prediction_threshold_sell", 0.45),
                }
            }

        self.take_profit_percent = config.get("take_profit_percent", 0.05)
        self.stop_loss_percent = config.get("stop_loss_percent", 0.02)
        self.allocation_percent = config.get("allocation_percent", 0.1)
        self.single_position_only = config.get("single_position_only", True)

    def validate_config(self) -> bool:
        if not self.symbol or not self.timeframes_config:
            logger.error("MLStrategy: 'symbol' or 'timeframes_config' not found in configuration.")
            return False
        if not self.ml_engine:
            logger.error("MLStrategy: ml_engine not provided.")
            return False
        if not self.data_fetcher:
            logger.error("MLStrategy: data_fetcher not provided.")
            return False

        # Validate ML model types
        supported_ml_model_types = ["neural_network", "lorentzian", "risk_assessment"] # Add "social_sentiment" if implemented
        if self.model_type not in supported_ml_model_types:
            logger.error(f"MLStrategy: Invalid 'model_type' '{self.model_type}'. Must be one of: {supported_ml_model_types}")
            return False

        # Check if a model exists for the selected symbol, timeframe, and model_type
        for timeframe, tf_config in self.timeframes_config.items():
            model_key = (self.symbol, timeframe, self.model_type)
            if model_key not in self.ml_engine.models:
                logger.error(f"MLStrategy: No model found for {self.symbol} in {timeframe} trained with '{self.model_type}'. Please train this model first.")
                return False
            # Check if scaler exists for this model
            scaler_key = (self.symbol, timeframe)
            if scaler_key not in self.ml_engine.scalers:
                logger.warning(f"MLStrategy: No scalar found for {self.symbol} in {timeframe}. Model predictions might be inconsistent.")
                
            # Basic validation of timeframe config numeric values
            for k in ["required_history", "prediction_threshold_buy", "prediction_threshold_sell"]:
                if k not in tf_config or not (isinstance(tf_config[k], (int, float))):
                    logger.error(f"MLStrategy: timeframes_config for '{timeframe}' has invalid or missing value for '{k}'.")
                    return False
                
        logger.info(f"MLStrategy config for {self.symbol} validated successfully.")
        return True

    def get_required_history(self, timeframe: Optional[str] = None) -> int:
        if timeframe and timeframe in self.timeframes_config:
            return self.timeframes_config[timeframe]["required_history"]
        
        max_history = 0
        for tf_config in self.timeframes_config.values():
            max_history = max(max_history, tf_config.get("required_history", 0))
        return max_history


    async def should_buy(self, current_market_data: Dict[str, Any], current_position: Dict[str, Any]) -> Signal:
        symbol = current_market_data["symbol"]
        current_price = current_market_data["price"]

        if self.single_position_only and current_position:
            return Signal(SignalType.HOLD, 0, "Already in position (single_position_only).")

        predictions = {}

        for timeframe, config in self.timeframes_config.items():
            required_hist = config["required_history"]
            
            fetch_limit = required_hist + max(50, self.get_max_indicator_lookback()) 
            hist_df = await self.data_fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit) 
            
            if hist_df.empty or len(hist_df) < required_hist:
                logger.warning(f"Insufficient historical data for {symbol} ({timeframe}). Needed {required_hist}, got {len(hist_df)}.")
                predictions[timeframe] = {"action": "HOLD", "confidence": 0, "error": "Insufficient data"} 
                continue
            
            populated_df = self.populate_indicators(hist_df)
            
            populated_df.dropna(inplace=True) 
            if populated_df.empty:
                logger.warning(f"DataFrame became empty after populating indicators and dropping NaNs for {symbol} ({timeframe}).")
                # Indentation fix applied here: This line needs to align with the 'logger.warning'
                predictions[timeframe] = {"action": "HOLD", "confidence": 0, "error": "Empty after indicators"} 
                continue # This continue also needs to be at the same level as the 'if populated_df.empty:'

            if len(populated_df) < required_hist:
                 logger.warning(f"Not enough data for {symbol} ({timeframe}) after indicator dropna. Needed {required_hist}, got {len(populated_df)}.")
                 predictions[timeframe] = {"action": "HOLD", "confidence": 0, "error": "Insufficient data after indicators"} 
                 continue

            # Prepare latest data for prediction (pass the last row of features)
            latest_features_dict = populated_df.iloc[-1].to_dict()

            # Get prediction from ML engine using the configured model_type
            # Check for specific model_type method directly, or use getattr for flexibility
            pred = {} # Initialize pred to avoid UnboundLocalError
            if hasattr(self.ml_engine, f"predict_{self.model_type}") and self.model_type != "risk_assessment":
                pred_func = getattr(self.ml_engine, f"predict_{self.model_type}")
                pred = pred_func(symbol, latest_features_dict, timeframe)
                # Ensure the prediction format is consistent
                if "action" not in pred or "confidence" not in pred:
                    logger.warning(f"MLStrategy: Prediction from {self.model_type} model for {symbol} ({timeframe}) has unexpected format: {pred}")
                    pred = {"action": "HOLD", "confidence": 0}
            elif self.model_type == "risk_assessment": # Special handling for risk_assessment's output
                risk_pred = self.ml_engine.evaluate_risk(symbol, latest_features_dict, timeframe)
                if risk_pred.get("risk_level") == "LOW" and risk_pred.get("confidence", 0) > 0.7:
                    # If low risk and high confidence in risk model, treat as BUY opportunity
                    pred = {"action": "BUY", "confidence": risk_pred.get("confidence", 0)}
                else:
                    # If high risk or low confidence in risk model, treat as HOLD
                    pred = {"action": "HOLD", "confidence": risk_pred.get("confidence", 0)}
            else:
                logger.error(f"MLStrategy: Model type '{self.model_type}' not supported for direct prediction or missing in ML engine.")
                pred = {"action": "HOLD", "confidence": 0, "error": "Model type not supported"}
            
            predictions[timeframe] = pred
        
        # --- Multi-Timeframe Buy Logic ---
        pred_1h = predictions.get("1h", {"action": "HOLD", "confidence": 0})
        pred_1d = predictions.get("1d", {"action": "HOLD", "confidence": 0})
        
        threshold_buy_1h = self.timeframes_config.get("1h", {}).get("prediction_threshold_buy", 0.55)
        threshold_buy_1d = self.timeframes_config.get("1d", {}).get("prediction_threshold_buy", 0.5) 

        if pred_1h["action"] == "BUY" and pred_1h["confidence"] >= threshold_buy_1h:
            logger.info(f"1h model for {symbol} suggests BUY (Conf: {pred_1h['confidence']:.2f}).")
            if pred_1d["action"] == "BUY" and pred_1d["confidence"] >= threshold_buy_1d: 
                logger.info(f"1d model for {symbol} confirms BUY (Conf: {pred_1d['confidence']:.2f}). Triggering BUY.")
                
                if not hasattr(self, '_engine_reference') or not self._engine_reference:
                    logger.warning("Engine reference not set in MLStrategy. Cannot check balance for buy.")
                    return Signal(SignalType.HOLD, 0, "Engine not linked to strategy.")

                current_balance = self._engine_reference.balances.get("USDT", 0) 
                if current_price > 0:
                    quantity = (current_balance * self.allocation_percent) / current_price
                else:
                    quantity = 0
                
                if quantity > 0:
                    return Signal(SignalType.BUY, pred_1h["confidence"], f"ML Buy signal (1h confirmed by 1d).", quantity)
                else:
                    return Signal(SignalType.HOLD, 0, "Calculated quantity is zero or less. Not buying.") 
            else:
                return Signal(SignalType.HOLD, 0, f"1h Buy signal not confirmed by 1d model ({pred_1d['action']} Conf: {pred_1d['confidence']:.2f}).") 
        else:
            return Signal(SignalType.HOLD, 0, f"1h model for {symbol} not suggesting BUY ({pred_1h['action']} Conf: {pred_1h['confidence']:.2f}).") 


    async def should_sell(self, current_market_data: Dict[str, Any], current_position: Dict[str, Any]) -> Signal:
        symbol = current_market_data["symbol"]
        current_price = current_market_data["price"]

        if not current_position:
            return Signal(SignalType.HOLD, 0, "No position to sell.") 

        entry_price = current_position.get("entry_price", current_price)
        amount = current_position.get("amount", 0)
        
        # Check for Take Profit / Stop Loss
        profit_loss = (current_price - entry_price) / entry_price
        if profit_loss >= self.take_profit_percent:
            return Signal(SignalType.SELL, 1.0, f"Take Profit triggered: {profit_loss*100:.2f}% gain.", amount)
        if profit_loss <= -self.stop_loss_percent:
            return Signal(SignalType.SELL, 1.0, f"Stop Loss triggered: {profit_loss*100:.2f}% loss.", amount)


        predictions = {}

        for timeframe, config in self.timeframes_config.items():
            required_hist = config["required_history"]
            
            fetch_limit = required_hist + max(50, self.get_max_indicator_lookback())
            hist_df = await self.data_fetcher.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit)
            
            if hist_df.empty or len(hist_df) < required_hist:
                logger.warning(f"Insufficient historical data for {symbol} ({timeframe}). Needed {required_hist}, got {len(hist_df)}.")
                predictions[timeframe] = {"action": "HOLD", "confidence": 0, "error": "Insufficient data"} 
                continue
            
            populated_df = self.populate_indicators(hist_df)
            populated_df.dropna(inplace=True)
            if populated_df.empty:
                logger.warning(f"DataFrame became empty after populating indicators and dropping NaNs for {symbol} ({timeframe}).")
                # Indentation fix applied here
                predictions[timeframe] = {"action": "HOLD", "confidence": 0, "error": "Empty after indicators"} 
                continue # This continue also needs to be at the same level as the 'if populated_df.empty:'

            if len(populated_df) < required_hist:
                 logger.warning(f"Not enough data for {symbol} ({timeframe}) after indicator dropna. Needed {required_hist}, got {len(populated_df)}.")
                 predictions[timeframe] = {"action": "HOLD", "confidence": 0, "error": "Insufficient data after indicators"} 
                 continue

            latest_features_dict = populated_df.iloc[-1].to_dict()

            pred = {} # Initialize pred to avoid UnboundLocalError
            if hasattr(self.ml_engine, f"predict_{self.model_type}") and self.model_type != "risk_assessment":
                pred_func = getattr(self.ml_engine, f"predict_{self.model_type}")
                pred = pred_func(symbol, latest_features_dict, timeframe)
                if "action" not in pred or "confidence" not in pred:
                    logger.warning(f"MLStrategy: Prediction from {self.model_type} model for {symbol} ({timeframe}) has unexpected format: {pred}")
                    pred = {"action": "HOLD", "confidence": 0}
            elif self.model_type == "risk_assessment":
                risk_pred = self.ml_engine.evaluate_risk(symbol, latest_features_dict, timeframe)
                if risk_pred.get("risk_level") == "HIGH" and risk_pred.get("confidence", 0) > 0.7:
                    pred = {"action": "SELL", "confidence": risk_pred.get("confidence", 0)}
                else:
                    pred = {"action": "HOLD", "confidence": risk_pred.get("confidence", 0)}
            else:
                logger.error(f"MLStrategy: Model type '{self.model_type}' not supported for direct prediction or missing in ML engine.")
                pred = {"action": "HOLD", "confidence": 0, "error": "Model type not supported"}
            
            predictions[timeframe] = pred

        # --- Multi-Timeframe Sell Logic ---
        pred_1h = predictions.get("1h", {"action": "HOLD", "confidence": 0})
        pred_1d = predictions.get("1d", {"action": "HOLD", "confidence": 0})

        threshold_sell_1h = self.timeframes_config.get("1h", {}).get("prediction_threshold_sell", 0.45)
        threshold_sell_1d = self.timeframes_config.get("1d", {}).get("prediction_threshold_sell", 0.5) 

        if pred_1h["action"] == "SELL" and pred_1h["confidence"] >= threshold_sell_1h:
            logger.info(f"1h model for {symbol} suggests SELL (Conf: {pred_1h['confidence']:.2f}).")
            if pred_1d["action"] == "SELL" and pred_1d["confidence"] >= threshold_sell_1d: 
                logger.info(f"1d model for {symbol} confirms SELL (Conf: {pred_1d['confidence']:.2f}). Triggering SELL.")
                return Signal(SignalType.SELL, pred_1h["confidence"], f"ML Sell signal (1h confirmed by 1d).", amount)
            else:
                return Signal(SignalType.HOLD, 0, f"1h Sell signal not confirmed by 1d model ({pred_1d['action']} Conf: {pred_1d['confidence']:.2f}).") 
        else:
            return Signal(SignalType.HOLD, 0, f"1h model for {symbol} not suggesting SELL ({pred_1h['action']} Conf: {pred_1h['confidence']:.2f}).") 

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Populates the DataFrame with common trading indicators.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to populate_indicators.")
            return df

        df['open'] = pd.to_numeric(df['open'], errors='coerce').ffill().bfill().fillna(0.0)
        df['high'] = pd.to_numeric(df['high'], errors='coerce').ffill().bfill().fillna(0.0)
        df['low'] = pd.to_numeric(df['low'], errors='coerce').ffill().bfill().fillna(0.0)
        df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill().bfill().fillna(0.0)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').ffill().bfill().fillna(0.0)

        df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
        
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(span=14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(span=14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rs.replace([float('inf'), -float('inf')], np.nan, inplace=True) 
        rs.fillna(1, inplace=True) 

        df['RSI'] = 100 - (100 / (1 + rs)).fillna(0)
        df['RSI'].fillna(50, inplace=True)

        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

        df['Middle_Band'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['Std_Dev'] = df['close'].rolling(window=20, min_periods=1).std()
        df['Upper_Band'] = df['Middle_Band'] + (df['Std_Dev'] * 2)
        df['Lower_Band'] = df['Middle_Band'] - (df['Std_Dev'] * 2)

        df['Daily_Return'] = df['close'].pct_change().fillna(0)
        vol_window = 20 if df.index.freq == 'D' else 24 * 20 
        df['Volatility'] = df['Daily_Return'].rolling(window=vol_window, min_periods=1).std().fillna(0.0)


        df.dropna(inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        return df

    def get_max_indicator_lookback(self):
        return max(50, # SMA_50
                   26, # EMA_26
                   14, # RSI_14
                   26, # MACD (depends on EMA_26)
                   20, # Bollinger Bands
                   20 # Volatility
                  )