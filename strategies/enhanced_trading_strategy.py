# enhanced_trading_strategy.py - Enhanced Trading Strategy Engine with ML Integration
"""
Enhanced Trading Strategy Engine with ML Integration
Implements sophisticated trading strategies using ensemble ML predictions
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import asyncio
import warnings
warnings.filterwarnings('ignore')

class SignalStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    NEUTRAL = 3
    STRONG = 4
    VERY_STRONG = 5

class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    """Complete trading signal with all relevant information"""
    symbol: str
    timeframe: str
    direction: TradeDirection
    strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    ml_predictions: Dict[str, float]
    technical_indicators: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    market_conditions: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'timestamp': self.timestamp.isoformat(),
            'ml_predictions': self.ml_predictions,
            'technical_indicators': self.technical_indicators,
            'risk_assessment': self.risk_assessment,
            'market_conditions': self.market_conditions
        }

class EnhancedTradingStrategy:
    """Advanced trading strategy engine with ML integration"""
    
    def __init__(self, config_path: str = "config/strategy_config.json"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Strategy parameters
        self.risk_params = self.config['risk_management']
        self.signal_params = self.config['signal_generation']
        self.position_params = self.config['position_sizing']
        
        # Market analysis
        self.market_regime = "NORMAL"  # BULL, BEAR, NORMAL, VOLATILE
        self.volatility_regime = "MEDIUM"  # LOW, MEDIUM, HIGH
        
        # Signal history for pattern analysis
        self.signal_history = []
        self.performance_tracker = {}
        
        # Multi-timeframe analysis weights
        self.timeframe_weights = self.config['timeframe_weights']
        
        self.logger.info("Enhanced Trading Strategy initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load strategy configuration"""
        default_config = {
            "risk_management": {
                "max_risk_per_trade": 0.02,
                "max_portfolio_risk": 0.10,
                "max_correlation_exposure": 0.6,
                "max_drawdown_threshold": 0.15,
                "stop_loss_multiplier": 2.0,
                "take_profit_multiplier": 3.0,
                "trailing_stop_enabled": True,
                "risk_free_rate": 0.02
            },
            "signal_generation": {
                "min_confidence_threshold": 0.6,
                "ensemble_weight_ml": 0.6,
                "ensemble_weight_technical": 0.3,
                "ensemble_weight_sentiment": 0.1,
                "signal_strength_thresholds": {
                    "very_weak": 0.45,
                    "weak": 0.55,
                    "neutral": 0.65,
                    "strong": 0.75,
                    "very_strong": 0.85
                },
                "confirmation_required": True,
                "multi_timeframe_analysis": True
            },
            "position_sizing": {
                "base_position_size": 0.1,
                "kelly_criterion_enabled": True,
                "volatility_adjustment": True,
                "confidence_scaling": True,
                "max_position_size": 0.25,
                "min_position_size": 0.01
            },
            "timeframe_weights": {
                "1h": 0.2,
                "4h": 0.3,
                "1d": 0.5
            },
            "market_conditions": {
                "volatility_lookback": 20,
                "trend_lookback": 50,
                "volume_lookback": 10
            }
        }
        
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                self._deep_update(default_config, loaded_config)
                self.logger.info(f"Strategy configuration loaded from {config_path}")
            else:
                self.logger.info("Using default strategy configuration")
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error loading strategy config: {e}. Using defaults.")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def generate_trading_signal(self, 
                               symbol: str, 
                               market_data: Dict[str, pd.DataFrame],
                               ml_predictions: Dict[str, Dict[str, float]],
                               current_positions: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Generate comprehensive trading signal using all available information"""
        
        try:
            self.logger.debug(f"Generating trading signal for {symbol}")
            
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(market_data)
            
            # Multi-timeframe analysis
            timeframe_signals = {}
            for timeframe, data in market_data.items():
                if timeframe in self.timeframe_weights and len(data) > 50:
                    tf_signal = self._analyze_timeframe(
                        symbol, timeframe, data, ml_predictions.get(timeframe, {})
                    )
                    timeframe_signals[timeframe] = tf_signal
            
            if not timeframe_signals:
                self.logger.warning(f"No valid timeframe signals for {symbol}")
                return None
            
            # Combine multi-timeframe signals
            combined_signal = self._combine_timeframe_signals(timeframe_signals)
            
            # Apply risk management filters
            if not self._passes_risk_filters(symbol, combined_signal, current_positions):
                self.logger.debug(f"Signal for {symbol} filtered out by risk management")
                return None
            
            # Calculate position sizing
            position_size = self._calculate_position_size(combined_signal, market_conditions)
            
            # Determine entry, stop loss, and take profit levels
            entry_price = self._get_current_price(market_data)
            stop_loss, take_profit = self._calculate_exit_levels(
                entry_price, combined_signal, market_conditions
            )
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                timeframe="MULTI",  # Multi-timeframe analysis
                direction=combined_signal['direction'],
                strength=combined_signal['strength'],
                confidence=combined_signal['confidence'],
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                timestamp=datetime.now(),
                ml_predictions=self._aggregate_ml_predictions(ml_predictions),
                technical_indicators=combined_signal['technical_summary'],
                risk_assessment=combined_signal['risk_assessment'],
                market_conditions=market_conditions
            )
            
            # Store signal for analysis
            self.signal_history.append(signal)
            if len(self.signal_history) > 1000:  # Keep last 1000 signals
                self.signal_history = self.signal_history[-1000:]
            
            self.logger.info(f"Generated {signal.direction.value} signal for {symbol} "
                           f"(Confidence: {signal.confidence:.3f}, Strength: {signal.strength.name})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal for {symbol}: {e}")
            return None
    
    def _analyze_market_conditions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze overall market conditions"""
        conditions = {
            'volatility_regime': 'MEDIUM',
            'trend_regime': 'NEUTRAL',
            'volume_regime': 'NORMAL',
            'market_stress': 0.0,
            'liquidity_score': 1.0
        }
        
        try:
            # Use the longest timeframe for market analysis (usually daily)
            primary_data = None
            for timeframe in ['1d', '4h', '1h']:
                if timeframe in market_data and len(market_data[timeframe]) > 50:
                    primary_data = market_data[timeframe]
                    break
            
            if primary_data is None:
                return conditions
            
            # Volatility analysis
            returns = primary_data['close'].pct_change().dropna()
            current_vol = returns.rolling(self.config['market_conditions']['volatility_lookback']).std().iloc[-1]
            historical_vol = returns.rolling(100).std().mean()
            
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            if vol_ratio > 1.5:
                conditions['volatility_regime'] = 'HIGH'
            elif vol_ratio < 0.7:
                conditions['volatility_regime'] = 'LOW'
            else:
                conditions['volatility_regime'] = 'MEDIUM'
            
            # Trend analysis
            trend_lookback = self.config['market_conditions']['trend_lookback']
            if len(primary_data) >= trend_lookback:
                price_change = (primary_data['close'].iloc[-1] - primary_data['close'].iloc[-trend_lookback]) / primary_data['close'].iloc[-trend_lookback]
                
                if price_change > 0.1:
                    conditions['trend_regime'] = 'BULLISH'
                elif price_change < -0.1:
                    conditions['trend_regime'] = 'BEARISH'
                else:
                    conditions['trend_regime'] = 'NEUTRAL'
            
            # Volume analysis
            if 'volume' in primary_data.columns:
                vol_lookback = self.config['market_conditions']['volume_lookback']
                recent_volume = primary_data['volume'].rolling(vol_lookback).mean().iloc[-1]
                historical_volume = primary_data['volume'].rolling(50).mean().mean()
                
                volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
                
                if volume_ratio > 1.3:
                    conditions['volume_regime'] = 'HIGH'
                elif volume_ratio < 0.7:
                    conditions['volume_regime'] = 'LOW'
                else:
                    conditions['volume_regime'] = 'NORMAL'
            
            # Market stress indicator (based on volatility and price movements)
            stress_components = [
                min(vol_ratio / 2.0, 1.0),  # Volatility stress
                abs(price_change) * 2,      # Price movement stress
            ]
            conditions['market_stress'] = min(np.mean(stress_components), 1.0)
            
            # Liquidity score (simplified)
            conditions['liquidity_score'] = max(0.5, 1.0 - conditions['market_stress'] * 0.5)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing market conditions: {e}")
        
        return conditions
    
    def _analyze_timeframe(self, symbol: str, timeframe: str, data: pd.DataFrame, 
                          ml_prediction: Dict[str, float]) -> Dict[str, Any]:
        """Analyze individual timeframe for signal generation"""
        
        signal = {
            'timeframe': timeframe,
            'ml_score': 0.5,
            'technical_score': 0.5,
            'confidence': 0.5,
            'direction': TradeDirection.HOLD,
            'strength': SignalStrength.NEUTRAL,
            'indicators': {},
            'risk_factors': []
        }
        
        try:
            # ML Model Analysis
            if ml_prediction:
                ensemble_prediction = ml_prediction.get('ensemble_prediction', 0.5)
                model_confidence = ml_prediction.get('confidence', 0.5)
                
                signal['ml_score'] = ensemble_prediction
                signal['ml_confidence'] = model_confidence
            
            # Technical Analysis
            tech_indicators = self._calculate_technical_indicators(data)
            signal['indicators'] = tech_indicators
            
            # Technical scoring
            tech_score = self._calculate_technical_score(tech_indicators)
            signal['technical_score'] = tech_score
            
            # Combined scoring
            ml_weight = self.signal_params['ensemble_weight_ml']
            tech_weight = self.signal_params['ensemble_weight_technical']
            
            combined_score = (signal['ml_score'] * ml_weight + 
                            signal['technical_score'] * tech_weight)
            
            signal['combined_score'] = combined_score
            
            # Determine direction and strength
            if combined_score > 0.5:
                signal['direction'] = TradeDirection.BUY
            elif combined_score < 0.5:
                signal['direction'] = TradeDirection.SELL
            else:
                signal['direction'] = TradeDirection.HOLD
            
            # Calculate confidence
            confidence_factors = [
                signal.get('ml_confidence', 0.5),
                abs(combined_score - 0.5) * 2,  # Distance from neutral
                self._calculate_indicator_agreement(tech_indicators),
            ]
            signal['confidence'] = np.mean(confidence_factors)
            
            # Determine signal strength
            signal['strength'] = self._determine_signal_strength(combined_score, signal['confidence'])
            
            # Risk assessment
            signal['risk_factors'] = self._assess_timeframe_risks(data, tech_indicators)
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe {timeframe}: {e}")
        
        return signal
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data.get('volume', pd.Series(index=data.index, data=1))
            
            # Moving Averages
            indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
            indicators['sma_50'] = close.rolling(50).mean().iloc[-1]
            indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1]
            
            # Price position relative to MAs
            current_price = close.iloc[-1]
            indicators['price_vs_sma20'] = (current_price - indicators['sma_20']) / indicators['sma_20']
            indicators['price_vs_sma50'] = (current_price - indicators['sma_50']) / indicators['sma_50']
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = pd.Series([indicators['macd']]).ewm(span=9).mean().iloc[0]
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = close.rolling(bb_period).mean()
            bb_upper = bb_middle + (close.rolling(bb_period).std() * bb_std)
            bb_lower = bb_middle - (close.rolling(bb_period).std() * bb_std)
            
            indicators['bb_position'] = ((current_price - bb_lower.iloc[-1]) / 
                                       (bb_upper.iloc[-1] - bb_lower.iloc[-1]))
            
            # Stochastic Oscillator
            if len(data) >= 14:
                low_14 = low.rolling(14).min()
                high_14 = high.rolling(14).max()
                indicators['stoch_k'] = ((current_price - low_14.iloc[-1]) / 
                                       (high_14.iloc[-1] - low_14.iloc[-1]) * 100)
            
            # Volume indicators
            if volume.sum() > 0:
                indicators['volume_sma'] = volume.rolling(20).mean().iloc[-1]
                indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma']
            
            # Volatility
            returns = close.pct_change()
            indicators['volatility'] = returns.rolling(20).std().iloc[-1]
            
            # Support/Resistance levels
            recent_high = high.rolling(20).max().iloc[-1]
            recent_low = low.rolling(20).min().iloc[-1]
            indicators['resistance_distance'] = (recent_high - current_price) / current_price
            indicators['support_distance'] = (current_price - recent_low) / current_price
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def _calculate_technical_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate technical analysis score"""
        scores = []
        
        try:
            # RSI scoring
            rsi = indicators.get('rsi', 50)
            if 30 <= rsi <= 70:
                rsi_score = 0.5 + (rsi - 50) / 100  # Neutral zone
            elif rsi < 30:
                rsi_score = 0.3  # Oversold - potential buy
            else:  # rsi > 70
                rsi_score = 0.7  # Overbought - potential sell
            scores.append(rsi_score)
            
            # Moving Average scoring
            price_vs_sma20 = indicators.get('price_vs_sma20', 0)
            ma_score = 0.5 + np.tanh(price_vs_sma20 * 10) * 0.4  # Scale to 0.1-0.9
            scores.append(ma_score)
            
            # MACD scoring
            macd_histogram = indicators.get('macd_histogram', 0)
            macd_score = 0.5 + np.tanh(macd_histogram * 100) * 0.3
            scores.append(macd_score)
            
            # Bollinger Bands scoring
            bb_position = indicators.get('bb_position', 0.5)
            bb_score = max(0.1, min(0.9, bb_position))
            scores.append(bb_score)
            
            # Volume confirmation
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio > 1.2:  # High volume
                volume_weight = 1.2
            elif volume_ratio < 0.8:  # Low volume
                volume_weight = 0.8
            else:
                volume_weight = 1.0
            
            # Weighted average
            base_score = np.mean(scores)
            final_score = base_score * volume_weight
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating technical score: {e}")
            return 0.5
    
    def _calculate_indicator_agreement(self, indicators: Dict[str, Any]) -> float:
        """Calculate how much technical indicators agree with each other"""
        try:
            signals = []
            
            # RSI signal
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                signals.append(0.2)  # Sell signal
            elif rsi < 30:
                signals.append(0.8)  # Buy signal
            else:
                signals.append(0.5)  # Neutral
            
            # MA signal
            price_vs_ma = indicators.get('price_vs_sma20', 0)
            ma_signal = 0.5 + np.tanh(price_vs_ma * 5) * 0.4
            signals.append(ma_signal)
            
            # MACD signal
            macd_histogram = indicators.get('macd_histogram', 0)
            if macd_histogram > 0:
                signals.append(0.7)
            elif macd_histogram < 0:
                signals.append(0.3)
            else:
                signals.append(0.5)
            
            # Calculate agreement (inverse of standard deviation)
            if len(signals) > 1:
                agreement = 1.0 - min(1.0, np.std(signals) * 2)
            else:
                agreement = 0.5
            
            return agreement
            
        except Exception as e:
            self.logger.error(f"Error calculating indicator agreement: {e}")
            return 0.5
    
    def _determine_signal_strength(self, combined_score: float, confidence: float) -> SignalStrength:
        """Determine signal strength based on score and confidence"""
        thresholds = self.signal_params['signal_strength_thresholds']
        
        # Adjust score by confidence
        adjusted_score = abs(combined_score - 0.5) * 2 * confidence
        
        if adjusted_score >= thresholds['very_strong']:
            return SignalStrength.VERY_STRONG
        elif adjusted_score >= thresholds['strong']:
            return SignalStrength.STRONG
        elif adjusted_score >= thresholds['neutral']:
            return SignalStrength.NEUTRAL
        elif adjusted_score >= thresholds['weak']:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _assess_timeframe_risks(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> List[str]:
        """Assess risks for specific timeframe"""
        risks = []
        
        try:
            # High volatility risk
            volatility = indicators.get('volatility', 0)
            if volatility > 0.05:  # 5% daily volatility
                risks.append('HIGH_VOLATILITY')
            
            # Low volume risk
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio < 0.5:
                risks.append('LOW_VOLUME')
            
            # Extreme RSI levels
            rsi = indicators.get('rsi', 50)
            if rsi > 80 or rsi < 20:
                risks.append('EXTREME_RSI')
            
            # Price near support/resistance
            resistance_distance = indicators.get('resistance_distance', 1.0)
            support_distance = indicators.get('support_distance', 1.0)
            
            if resistance_distance < 0.02:  # Within 2% of resistance
                risks.append('NEAR_RESISTANCE')
            if support_distance < 0.02:  # Within 2% of support
                risks.append('NEAR_SUPPORT')
            
            # Gap analysis (if there are significant price gaps)
            if len(data) > 1:
                price_change = abs(data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
                if price_change > 0.05:  # 5% gap
                    risks.append('PRICE_GAP')
            
        except Exception as e:
            self.logger.error(f"Error assessing timeframe risks: {e}")
        
        return risks
    
    def _combine_timeframe_signals(self, timeframe_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine signals from multiple timeframes"""
        
        combined = {
            'direction': TradeDirection.HOLD,
            'strength': SignalStrength.NEUTRAL,
            'confidence': 0.5,
            'combined_score': 0.5,
            'technical_summary': {},
            'risk_assessment': {'risks': [], 'risk_score': 0.0}
        }
        
        try:
            weighted_scores = []
            confidences = []
            all_risks = []
            
            # Combine timeframe signals with weights
            for timeframe, signal in timeframe_signals.items():
                weight = self.timeframe_weights.get(timeframe, 0.33)
                score = signal.get('combined_score', 0.5)
                confidence = signal.get('confidence', 0.5)
                
                weighted_scores.append(score * weight)
                confidences.append(confidence * weight)
                all_risks.extend(signal.get('risk_factors', []))
            
            # Calculate combined metrics
            combined['combined_score'] = sum(weighted_scores)
            combined['confidence'] = sum(confidences)
            
            # Determine final direction
            if combined['combined_score'] > 0.55:
                combined['direction'] = TradeDirection.BUY
            elif combined['combined_score'] < 0.45:
                combined['direction'] = TradeDirection.SELL
            else:
                combined['direction'] = TradeDirection.HOLD
            
            # Calculate signal strength
            combined['strength'] = self._determine_signal_strength(
                combined['combined_score'], combined['confidence']
            )
            
            # Risk assessment
            unique_risks = list(set(all_risks))
            risk_score = min(1.0, len(unique_risks) / 10.0)  # Scale risk score
            
            combined['risk_assessment'] = {
                'risks': unique_risks,
                'risk_score': risk_score,
                'total_risk_factors': len(all_risks)
            }
            
            # Technical summary (average key indicators)
            combined['technical_summary'] = self._summarize_technical_indicators(timeframe_signals)
            
        except Exception as e:
            self.logger.error(f"Error combining timeframe signals: {e}")
        
        return combined
    
    def _summarize_technical_indicators(self, timeframe_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize technical indicators across timeframes"""
        summary = {}
        
        try:
            # Collect all indicators
            all_indicators = []
            for signal in timeframe_signals.values():
                indicators = signal.get('indicators', {})
                if indicators:
                    all_indicators.append(indicators)
            
            if not all_indicators:
                return summary
            
            # Average numerical indicators
            numeric_keys = ['rsi', 'bb_position', 'volatility', 'volume_ratio']
            for key in numeric_keys:
                values = [ind.get(key) for ind in all_indicators if ind.get(key) is not None]
                if values:
                    summary[f'avg_{key}'] = np.mean(values)
            
            # Trend indicators
            ma_trends = []
            for indicators in all_indicators:
                price_vs_ma = indicators.get('price_vs_sma20', 0)
                if price_vs_ma > 0.02:
                    ma_trends.append('BULLISH')
                elif price_vs_ma < -0.02:
                    ma_trends.append('BEARISH')
                else:
                    ma_trends.append('NEUTRAL')
            
            if ma_trends:
                summary['trend_consensus'] = max(set(ma_trends), key=ma_trends.count)
                summary['trend_agreement'] = ma_trends.count(summary['trend_consensus']) / len(ma_trends)
            
        except Exception as e:
            self.logger.error(f"Error summarizing technical indicators: {e}")
        
        return summary
    
    def _passes_risk_filters(self, symbol: str, signal: Dict[str, Any], 
                           current_positions: Dict[str, Any] = None) -> bool:
        """Apply risk management filters to signal"""
        
        try:
            # Minimum confidence filter
            if signal['confidence'] < self.signal_params['min_confidence_threshold']:
                return False
            
            # Risk score filter
            risk_score = signal['risk_assessment']['risk_score']
            if risk_score > 0.7:  # High risk
                return False
            
            # Position correlation filter (if we have current positions)
            if current_positions:
                correlation_risk = self._calculate_correlation_risk(symbol, current_positions)
                if correlation_risk > self.risk_params['max_correlation_exposure']:
                    return False
            
            # Market stress filter
            # if self.market_stress > 0.8:  # Very high market stress
            #     return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in risk filters: {e}")
            return False
    
    def _calculate_correlation_risk(self, symbol: str, current_positions: Dict[str, Any]) -> float:
        """Calculate correlation risk with current positions"""
        # Simplified correlation calculation
        # In practice, you'd use historical correlation data
        
        related_symbols = {
            'BTC/USD': ['ETH/USD', 'ADA/USD'],
            'ETH/USD': ['BTC/USD', 'ADA/USD'],
            'ADA/USD': ['BTC/USD', 'ETH/USD']
        }
        
        correlation_exposure = 0.0
        for pos_symbol, position in current_positions.items():
            if pos_symbol in related_symbols.get(symbol, []):
                correlation_exposure += abs(position.get('size', 0))
        
        return correlation_exposure
    
    def _calculate_position_size(self, signal: Dict[str, Any], 
                               market_conditions: Dict[str, Any]) -> float:
        """Calculate optimal position size using multiple methods"""
        
        try:
            base_size = self.position_params['base_position_size']
            
            # Confidence scaling
            if self.position_params['confidence_scaling']:
                confidence_multiplier = signal['confidence']
                base_size *= confidence_multiplier
            
            # Volatility adjustment
            if self.position_params['volatility_adjustment']:
                market_stress = market_conditions.get('market_stress', 0.5)
                volatility_multiplier = max(0.5, 1.0 - market_stress * 0.5)
                base_size *= volatility_multiplier
            
            # Kelly Criterion adjustment (simplified)
            if self.position_params['kelly_criterion_enabled']:
                win_rate = 0.55  # Historical win rate (would be calculated from performance)
                avg_win = 0.03   # Average win percentage
                avg_loss = 0.02  # Average loss percentage
                
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_multiplier = max(0.1, min(2.0, kelly_fraction))
                base_size *= kelly_multiplier
            
            # Apply limits
            position_size = max(
                self.position_params['min_position_size'],
                min(self.position_params['max_position_size'], base_size)
            )
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self.position_params['base_position_size']
    
    def _get_current_price(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Get current price from market data"""
        # Use the shortest timeframe for most recent price
        for timeframe in ['1h', '4h', '1d']:
            if timeframe in market_data and len(market_data[timeframe]) > 0:
                return float(market_data[timeframe]['close'].iloc[-1])
        
        return 0.0
    
    def _calculate_exit_levels(self, entry_price: float, signal: Dict[str, Any], 
                             market_conditions: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        
        try:
            # Base levels from risk parameters
            stop_multiplier = self.risk_params['stop_loss_multiplier']
            profit_multiplier = self.risk_params['take_profit_multiplier']
            
            # Adjust for volatility
            volatility_factor = market_conditions.get('market_stress', 0.5)
            adjusted_stop_multiplier = stop_multiplier * (1 + volatility_factor * 0.5)
            adjusted_profit_multiplier = profit_multiplier * (1 + volatility_factor * 0.3)
            
            # Calculate based on direction
            if signal['direction'] == TradeDirection.BUY:
                # For buy signals
                risk_amount = entry_price * self.risk_params['max_risk_per_trade']
                stop_loss = entry_price - (risk_amount * adjusted_stop_multiplier)
                take_profit = entry_price + (risk_amount * adjusted_profit_multiplier)
            
            elif signal['direction'] == TradeDirection.SELL:
                # For sell signals
                risk_amount = entry_price * self.risk_params['max_risk_per_trade']
                stop_loss = entry_price + (risk_amount * adjusted_stop_multiplier)
                take_profit = entry_price - (risk_amount * adjusted_profit_multiplier)
            
            else:  # HOLD
                stop_loss = entry_price
                take_profit = entry_price
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating exit levels: {e}")
            return entry_price * 0.98, entry_price * 1.02  # Default 2% levels
    
    def _aggregate_ml_predictions(self, ml_predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate ML predictions across timeframes"""
        aggregated = {
            'ensemble_prediction': 0.5,
            'confidence': 0.5,
            'individual_models': {}
        }
        
        try:
            predictions = []
            confidences = []
            
            for timeframe, pred_data in ml_predictions.items():
                if pred_data and 'ensemble_prediction' in pred_data:
                    weight = self.timeframe_weights.get(timeframe, 0.33)
                    predictions.append(pred_data['ensemble_prediction'] * weight)
                    confidences.append(pred_data.get('confidence', 0.5) * weight)
                    
                    # Store individual model predictions
                    individual_preds = pred_data.get('individual_predictions', {})
                    for model_name, pred in individual_preds.items():
                        key = f"{timeframe}_{model_name}"
                        aggregated['individual_models'][key] = pred
            
            if predictions:
                aggregated['ensemble_prediction'] = sum(predictions)
                aggregated['confidence'] = sum(confidences)
            
        except Exception as e:
            self.logger.error(f"Error aggregating ML predictions: {e}")
        
        return aggregated
    
    def update_strategy_performance(self, trades: List[Dict[str, Any]]):
        """Update strategy performance metrics"""
        try:
            for trade in trades:
                symbol = trade.get('symbol')
                if symbol not in self.performance_tracker:
                    self.performance_tracker[symbol] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_profit': 0.0,
                        'signals_generated': 0
                    }
                
                tracker = self.performance_tracker[symbol]
                tracker['total_trades'] += 1
                
                profit = trade.get('profit_loss', 0)
                if profit > 0:
                    tracker['winning_trades'] += 1
                
                tracker['total_profit'] += profit
            
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {e}")
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get comprehensive strategy performance metrics"""
        try:
            overall_stats = {
                'total_signals': len(self.signal_history),
                'symbols_tracked': len(self.performance_tracker),
                'last_signal_time': self.signal_history[-1].timestamp.isoformat() if self.signal_history else None,
                'performance_by_symbol': {}
            }
            
            for symbol, tracker in self.performance_tracker.items():
                if tracker['total_trades'] > 0:
                    win_rate = tracker['winning_trades'] / tracker['total_trades']
                    avg_profit = tracker['total_profit'] / tracker['total_trades']
                else:
                    win_rate = 0
                    avg_profit = 0
                
                overall_stats['performance_by_symbol'][symbol] = {
                    'total_trades': tracker['total_trades'],
                    'win_rate': win_rate,
                    'avg_profit_per_trade': avg_profit,
                    'total_profit': tracker['total_profit'],
                    'signals_generated': tracker['signals_generated']
                }
            
            return overall_stats
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return {'error': str(e)}
    
    def optimize_parameters(self, performance_data: Dict[str, Any]):
        """Optimize strategy parameters based on performance"""
        try:
            # Simplified parameter optimization
            # In practice, you'd use more sophisticated optimization algorithms
            
            win_rate = performance_data.get('overall_win_rate', 0.5)
            avg_profit = performance_data.get('avg_profit_per_trade', 0.0)
            
            # Adjust confidence threshold based on performance
            if win_rate < 0.5 and avg_profit < 0:
                # Increase confidence threshold to be more selective
                current_threshold = self.signal_params['min_confidence_threshold']
                new_threshold = min(0.8, current_threshold + 0.05)
                self.signal_params['min_confidence_threshold'] = new_threshold
                
                self.logger.info(f"Increased confidence threshold to {new_threshold:.3f}")
            
            elif win_rate > 0.6 and avg_profit > 0.01:
                # Decrease confidence threshold to capture more opportunities
                current_threshold = self.signal_params['min_confidence_threshold']
                new_threshold = max(0.4, current_threshold - 0.05)
                self.signal_params['min_confidence_threshold'] = new_threshold
                
                self.logger.info(f"Decreased confidence threshold to {new_threshold:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")

def main():
    """Main function for testing"""
    import random
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize strategy
    strategy = EnhancedTradingStrategy()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(50000, 55000, 100),
        'high': np.random.uniform(50500, 55500, 100),
        'low': np.random.uniform(49500, 54500, 100),
        'close': np.random.uniform(50000, 55000, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    market_data = {'1h': sample_data}
    
    # Sample ML predictions
    ml_predictions = {
        '1h': {
            'ensemble_prediction': 0.75,
            'confidence': 0.8,
            'individual_predictions': {
                'lstm': 0.7,
                'random_forest': 0.8,
                'gradient_boosting': 0.75
            }
        }
    }
    
    # Generate signal
    signal = strategy.generate_trading_signal(
        symbol='BTC/USD',
        market_data=market_data,
        ml_predictions=ml_predictions
    )
    
    if signal:
        print("Generated Signal:")
        print(json.dumps(signal.to_dict(), indent=2, default=str))
    else:
        print("No signal generated")

if __name__ == "__main__":
    main()