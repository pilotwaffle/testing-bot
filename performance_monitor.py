# performance_monitor.py - Real-time Performance Monitor & Adaptive Learning System
"""
Real-time Performance Monitor and Adaptive Learning System
Continuously monitors trading performance and triggers model improvements
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import time
import threading
from dataclasses import dataclass, asdict
from collections import deque
import sqlite3
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradeRecord:
    """Structure for individual trade records"""
    timestamp: datetime
    symbol: str
    timeframe: str
    prediction: float
    confidence: float
    actual_result: Optional[float]
    profit_loss: Optional[float]
    trade_id: str
    model_used: str
    market_conditions: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'actual_result': self.actual_result,
            'profit_loss': self.profit_loss,
            'trade_id': self.trade_id,
            'model_used': self.model_used,
            'market_conditions': json.dumps(self.market_conditions)
        }

class PerformanceMonitor:
    """Real-time performance monitoring and adaptive learning system"""
    
    def __init__(self, config_path: str = "config/monitor_config.json"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize database
        self.db_path = Path(self.config['database']['path'])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
        # Performance tracking
        self.trade_buffer = deque(maxlen=1000)  # Recent trades buffer
        self.performance_metrics = {}
        self.alert_callbacks = []
        
        # Adaptive learning parameters
        self.learning_config = self.config['adaptive_learning']
        self.performance_thresholds = self.config['performance_thresholds']
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_evaluation = datetime.now()
        
        # Performance tracking windows
        self.performance_windows = {
            'short_term': deque(maxlen=50),    # Last 50 trades
            'medium_term': deque(maxlen=200),  # Last 200 trades
            'long_term': deque(maxlen=500)     # Last 500 trades
        }
        
        self.logger.info("Performance Monitor initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "database": {
                "path": "data/performance.db"
            },
            "monitoring": {
                "evaluation_interval_minutes": 30,
                "min_trades_for_evaluation": 10,
                "alert_on_performance_drop": True,
                "save_detailed_logs": True
            },
            "performance_thresholds": {
                "accuracy_warning": 0.55,
                "accuracy_critical": 0.45,
                "profit_warning": -0.02,
                "profit_critical": -0.05,
                "max_consecutive_losses": 5
            },
            "adaptive_learning": {
                "auto_retrain_enabled": True,
                "retrain_on_accuracy_drop": 0.1,
                "retrain_on_profit_drop": 0.03,
                "feature_adaptation_enabled": True,
                "market_regime_detection": True
            },
            "alerts": {
                "email_enabled": False,
                "webhook_url": None,
                "log_level": "WARNING"
            }
        }
        
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                self._deep_update(default_config, loaded_config)
                self.logger.info(f"Configuration loaded from {config_path}")
            else:
                self.logger.info("Using default monitoring configuration")
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _initialize_database(self):
        """Initialize SQLite database for performance tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        prediction REAL NOT NULL,
                        confidence REAL NOT NULL,
                        actual_result REAL,
                        profit_loss REAL,
                        trade_id TEXT UNIQUE NOT NULL,
                        model_used TEXT NOT NULL,
                        market_conditions TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create performance summary table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        window_type TEXT NOT NULL,
                        accuracy REAL,
                        avg_profit REAL,
                        win_rate REAL,
                        total_trades INTEGER,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        resolved BOOLEAN DEFAULT FALSE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def record_trade(self, trade: TradeRecord):
        """Record a new trade for monitoring"""
        try:
            # Add to buffer
            self.trade_buffer.append(trade)
            
            # Add to performance windows
            for window in self.performance_windows.values():
                window.append(trade)
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                trade_dict = trade.to_dict()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO trades 
                    (timestamp, symbol, timeframe, prediction, confidence, 
                     actual_result, profit_loss, trade_id, model_used, market_conditions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_dict['timestamp'],
                    trade_dict['symbol'],
                    trade_dict['timeframe'],
                    trade_dict['prediction'],
                    trade_dict['confidence'],
                    trade_dict['actual_result'],
                    trade_dict['profit_loss'],
                    trade_dict['trade_id'],
                    trade_dict['model_used'],
                    trade_dict['market_conditions']
                ))
                
                conn.commit()
            
            self.logger.debug(f"Trade recorded: {trade.trade_id}")
            
            # Trigger real-time analysis if trade is complete
            if trade.actual_result is not None and trade.profit_loss is not None:
                self._analyze_trade_outcome(trade)
            
        except Exception as e:
            self.logger.error(f"Failed to record trade: {e}")
    
    def update_trade_outcome(self, trade_id: str, actual_result: float, profit_loss: float):
        """Update trade with actual outcome"""
        try:
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE trades 
                    SET actual_result = ?, profit_loss = ?
                    WHERE trade_id = ?
                ''', (actual_result, profit_loss, trade_id))
                
                conn.commit()
            
            # Update buffer
            for trade in self.trade_buffer:
                if trade.trade_id == trade_id:
                    trade.actual_result = actual_result
                    trade.profit_loss = profit_loss
                    self._analyze_trade_outcome(trade)
                    break
            
            self.logger.debug(f"Trade outcome updated: {trade_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update trade outcome: {e}")
    
    def _analyze_trade_outcome(self, trade: TradeRecord):
        """Analyze individual trade outcome for immediate feedback"""
        if trade.actual_result is None or trade.profit_loss is None:
            return
        
        # Check for prediction accuracy
        predicted_direction = trade.prediction > 0.5
        actual_direction = trade.actual_result > 0.5
        is_correct = predicted_direction == actual_direction
        
        # Check for significant loss
        if trade.profit_loss < self.performance_thresholds['profit_critical']:
            self._trigger_alert(
                alert_type='significant_loss',
                severity='critical',
                message=f"Significant loss on {trade.symbol} ({trade.timeframe}): {trade.profit_loss:.4f}",
                details=trade.to_dict()
            )
        
        # Check for consecutive losses
        self._check_consecutive_losses(trade)
        
        # Update real-time metrics
        self._update_realtime_metrics(trade, is_correct)
    
    def _check_consecutive_losses(self, latest_trade: TradeRecord):
        """Check for consecutive losses pattern"""
        if latest_trade.profit_loss >= 0:
            return  # Not a loss
        
        # Count recent consecutive losses
        consecutive_losses = 0
        for trade in reversed(list(self.performance_windows['short_term'])):
            if trade.profit_loss is not None and trade.profit_loss < 0:
                consecutive_losses += 1
            else:
                break
        
        max_allowed = self.performance_thresholds['max_consecutive_losses']
        if consecutive_losses >= max_allowed:
            self._trigger_alert(
                alert_type='consecutive_losses',
                severity='warning',
                message=f"Consecutive losses detected: {consecutive_losses} in a row",
                details={
                    'consecutive_losses': consecutive_losses,
                    'symbol': latest_trade.symbol,
                    'timeframe': latest_trade.timeframe,
                    'model': latest_trade.model_used
                }
            )
    
    def _update_realtime_metrics(self, trade: TradeRecord, is_correct: bool):
        """Update real-time performance metrics"""
        key = f"{trade.symbol}_{trade.timeframe}"
        
        if key not in self.performance_metrics:
            self.performance_metrics[key] = {
                'recent_accuracy': deque(maxlen=20),
                'recent_profits': deque(maxlen=20),
                'last_update': datetime.now()
            }
        
        metrics = self.performance_metrics[key]
        metrics['recent_accuracy'].append(1.0 if is_correct else 0.0)
        metrics['recent_profits'].append(trade.profit_loss)
        metrics['last_update'] = datetime.now()
        
        # Check for immediate performance issues
        if len(metrics['recent_accuracy']) >= 10:
            recent_accuracy = np.mean(metrics['recent_accuracy'])
            recent_profit = np.mean(metrics['recent_profits'])
            
            if recent_accuracy < self.performance_thresholds['accuracy_warning']:
                self._trigger_alert(
                    alert_type='accuracy_drop',
                    severity='warning',
                    message=f"Low accuracy for {trade.symbol} ({trade.timeframe}): {recent_accuracy:.3f}",
                    details={'accuracy': recent_accuracy, 'trades_analyzed': len(metrics['recent_accuracy'])}
                )
            
            if recent_profit < self.performance_thresholds['profit_warning']:
                self._trigger_alert(
                    alert_type='profit_drop',
                    severity='warning',
                    message=f"Negative returns for {trade.symbol} ({trade.timeframe}): {recent_profit:.4f}",
                    details={'avg_profit': recent_profit, 'trades_analyzed': len(metrics['recent_profits'])}
                )
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Periodic evaluation
                if self._should_run_evaluation():
                    self._run_periodic_evaluation()
                
                # Sleep for a short interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _should_run_evaluation(self) -> bool:
        """Check if periodic evaluation should run"""
        interval = timedelta(minutes=self.config['monitoring']['evaluation_interval_minutes'])
        return datetime.now() - self.last_evaluation > interval
    
    def _run_periodic_evaluation(self):
        """Run comprehensive periodic evaluation"""
        self.logger.info("Running periodic performance evaluation...")
        
        try:
            # Evaluate each symbol/timeframe combination
            evaluation_results = {}
            
            for symbol_timeframe, metrics in self.performance_metrics.items():
                if '_' not in symbol_timeframe:
                    continue
                
                symbol, timeframe = symbol_timeframe.split('_', 1)
                
                # Get recent trades for this combination
                recent_trades = [
                    trade for trade in self.performance_windows['medium_term']
                    if trade.symbol == symbol and trade.timeframe == timeframe
                    and trade.actual_result is not None and trade.profit_loss is not None
                ]
                
                if len(recent_trades) < self.config['monitoring']['min_trades_for_evaluation']:
                    continue
                
                # Calculate comprehensive metrics
                evaluation = self._calculate_comprehensive_metrics(recent_trades)
                evaluation_results[symbol_timeframe] = evaluation
                
                # Check for performance degradation
                self._check_performance_degradation(symbol, timeframe, evaluation)
                
                # Save to database
                self._save_performance_summary(symbol, timeframe, evaluation)
            
            # Generate evaluation report
            self._generate_evaluation_report(evaluation_results)
            
            self.last_evaluation = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Periodic evaluation failed: {e}")
    
    def _calculate_comprehensive_metrics(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {}
        
        # Convert to arrays for calculation
        predictions = np.array([t.prediction for t in trades])
        actuals = np.array([t.actual_result for t in trades])
        profits = np.array([t.profit_loss for t in trades])
        confidences = np.array([t.confidence for t in trades])
        
        # Basic metrics
        accuracy = np.mean((predictions > 0.5) == (actuals > 0.5))
        avg_profit = np.mean(profits)
        win_rate = np.mean(profits > 0)
        total_return = np.sum(profits)
        
        # Risk metrics
        profit_std = np.std(profits)
        sharpe_ratio = avg_profit / profit_std if profit_std > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Confidence analysis
        high_confidence_trades = profits[confidences > 0.7]
        high_confidence_accuracy = np.mean((predictions[confidences > 0.7] > 0.5) == (actuals[confidences > 0.7] > 0.5)) if len(high_confidence_trades) > 0 else 0
        
        # Streak analysis
        win_streak, loss_streak = self._calculate_streaks(profits)
        
        return {
            'total_trades': len(trades),
            'accuracy': accuracy,
            'avg_profit': avg_profit,
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_std': profit_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'high_confidence_accuracy': high_confidence_accuracy,
            'high_confidence_trades': len(high_confidence_trades),
            'current_win_streak': win_streak,
            'current_loss_streak': loss_streak,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_streaks(self, profits: np.ndarray) -> Tuple[int, int]:
        """Calculate current win/loss streaks"""
        if len(profits) == 0:
            return 0, 0
        
        # Current win streak
        win_streak = 0
        for i in range(len(profits) - 1, -1, -1):
            if profits[i] > 0:
                win_streak += 1
            else:
                break
        
        # Current loss streak
        loss_streak = 0
        for i in range(len(profits) - 1, -1, -1):
            if profits[i] <= 0:
                loss_streak += 1
            else:
                break
        
        return win_streak, loss_streak
    
    def _check_performance_degradation(self, symbol: str, timeframe: str, evaluation: Dict[str, Any]):
        """Check for performance degradation and trigger adaptive responses"""
        accuracy = evaluation.get('accuracy', 0)
        avg_profit = evaluation.get('avg_profit', 0)
        
        # Check accuracy degradation
        if accuracy < self.performance_thresholds['accuracy_critical']:
            self._trigger_alert(
                alert_type='critical_accuracy',
                severity='critical',
                message=f"Critical accuracy drop for {symbol} ({timeframe}): {accuracy:.3f}",
                details=evaluation
            )
            
            if self.learning_config['auto_retrain_enabled']:
                self._trigger_adaptive_action('retrain', symbol, timeframe, evaluation)
        
        elif accuracy < self.performance_thresholds['accuracy_warning']:
            self._trigger_alert(
                alert_type='accuracy_warning',
                severity='warning',
                message=f"Accuracy warning for {symbol} ({timeframe}): {accuracy:.3f}",
                details=evaluation
            )
        
        # Check profit degradation
        if avg_profit < self.performance_thresholds['profit_critical']:
            self._trigger_alert(
                alert_type='critical_losses',
                severity='critical',
                message=f"Critical profit drop for {symbol} ({timeframe}): {avg_profit:.4f}",
                details=evaluation
            )
            
            if self.learning_config['auto_retrain_enabled']:
                self._trigger_adaptive_action('retrain', symbol, timeframe, evaluation)
    
    def _trigger_adaptive_action(self, action: str, symbol: str, timeframe: str, evaluation: Dict[str, Any]):
        """Trigger adaptive learning actions"""
        self.logger.info(f"Triggering adaptive action: {action} for {symbol} ({timeframe})")
        
        try:
            if action == 'retrain':
                # Import here to avoid circular imports
                from enhanced_model_trainer import AdaptiveModelTrainer
                
                # Create trainer instance
                trainer = AdaptiveModelTrainer()
                
                # Trigger retraining in a separate thread to avoid blocking
                retraining_thread = threading.Thread(
                    target=self._run_retraining,
                    args=(trainer, symbol, timeframe, evaluation),
                    daemon=True
                )
                retraining_thread.start()
                
                self._trigger_alert(
                    alert_type='adaptive_action',
                    severity='info',
                    message=f"Adaptive retraining triggered for {symbol} ({timeframe})",
                    details={'action': action, 'trigger_evaluation': evaluation}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to trigger adaptive action: {e}")
    
    def _run_retraining(self, trainer, symbol: str, timeframe: str, evaluation: Dict[str, Any]):
        """Run retraining in background thread"""
        try:
            # Trigger specific model retraining
            trainer._trigger_adaptive_retraining([f"{symbol}_{timeframe}"])
            
            self.logger.info(f"Adaptive retraining completed for {symbol} ({timeframe})")
            
            self._trigger_alert(
                alert_type='retraining_complete',
                severity='info',
                message=f"Adaptive retraining completed for {symbol} ({timeframe})",
                details={'evaluation_trigger': evaluation}
            )
            
        except Exception as e:
            self.logger.error(f"Adaptive retraining failed: {e}")
            
            self._trigger_alert(
                alert_type='retraining_failed',
                severity='error',
                message=f"Adaptive retraining failed for {symbol} ({timeframe}): {str(e)}",
                details={'error': str(e), 'evaluation_trigger': evaluation}
            )
    
    def _trigger_alert(self, alert_type: str, severity: str, message: str, details: Any = None):
        """Trigger performance alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'details': json.dumps(details) if details else None
        }
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts (timestamp, alert_type, severity, message, details)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    alert['timestamp'],
                    alert['alert_type'],
                    alert['severity'],
                    alert['message'],
                    alert['details']
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save alert: {e}")
        
        # Log alert
        log_level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(log_level, f"ALERT [{alert_type}]: {message}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _save_performance_summary(self, symbol: str, timeframe: str, evaluation: Dict[str, Any]):
        """Save performance summary to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for window_type in ['short_term', 'medium_term', 'long_term']:
                    cursor.execute('''
                        INSERT INTO performance_summary 
                        (timestamp, symbol, timeframe, window_type, accuracy, avg_profit, 
                         win_rate, total_trades, sharpe_ratio, max_drawdown)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        evaluation['evaluation_timestamp'],
                        symbol,
                        timeframe,
                        window_type,
                        evaluation.get('accuracy'),
                        evaluation.get('avg_profit'),
                        evaluation.get('win_rate'),
                        evaluation.get('total_trades'),
                        evaluation.get('sharpe_ratio'),
                        evaluation.get('max_drawdown')
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save performance summary: {e}")
    
    def _generate_evaluation_report(self, results: Dict[str, Dict[str, Any]]):
        """Generate and save evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_models_evaluated': len(results),
                'overall_performance': {},
                'alerts_generated': 0,
                'adaptive_actions_triggered': 0
            },
            'model_performance': results,
            'recommendations': []
        }
        
        # Calculate overall metrics
        if results:
            all_accuracies = [r.get('accuracy', 0) for r in results.values() if r.get('accuracy')]
            all_profits = [r.get('avg_profit', 0) for r in results.values() if r.get('avg_profit')]
            
            if all_accuracies:
                report['summary']['overall_performance']['avg_accuracy'] = np.mean(all_accuracies)
                report['summary']['overall_performance']['min_accuracy'] = np.min(all_accuracies)
                report['summary']['overall_performance']['max_accuracy'] = np.max(all_accuracies)
            
            if all_profits:
                report['summary']['overall_performance']['avg_profit'] = np.mean(all_profits)
                report['summary']['overall_performance']['total_profit'] = np.sum(all_profits)
        
        # Save report
        try:
            reports_dir = Path("logs/evaluation_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = reports_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Evaluation report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save evaluation report: {e}")
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_recent_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for recent period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent trades
                cursor.execute('''
                    SELECT symbol, timeframe, prediction, actual_result, profit_loss
                    FROM trades 
                    WHERE timestamp > ? AND actual_result IS NOT NULL
                ''', (cutoff_time.isoformat(),))
                
                trades = cursor.fetchall()
                
                if not trades:
                    return {'message': 'No completed trades in the specified period'}
                
                # Calculate metrics by symbol/timeframe
                performance = {}
                for symbol, timeframe, prediction, actual, profit in trades:
                    key = f"{symbol}_{timeframe}"
                    if key not in performance:
                        performance[key] = {
                            'trades': [],
                            'predictions': [],
                            'actuals': [],
                            'profits': []
                        }
                    
                    performance[key]['trades'].append(1)
                    performance[key]['predictions'].append(prediction)
                    performance[key]['actuals'].append(actual)
                    performance[key]['profits'].append(profit)
                
                # Calculate summary metrics
                summary = {}
                for key, data in performance.items():
                    predictions = np.array(data['predictions'])
                    actuals = np.array(data['actuals'])
                    profits = np.array(data['profits'])
                    
                    accuracy = np.mean((predictions > 0.5) == (actuals > 0.5))
                    
                    summary[key] = {
                        'total_trades': len(data['trades']),
                        'accuracy': accuracy,
                        'avg_profit': np.mean(profits),
                        'total_profit': np.sum(profits),
                        'win_rate': np.mean(profits > 0),
                        'sharpe_ratio': np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
                    }
                
                return {
                    'period_hours': hours,
                    'total_trades': len(trades),
                    'performance_by_model': summary,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get recent performance: {e}")
            return {'error': str(e)}
    
    def get_alerts(self, hours: int = 24, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM alerts WHERE timestamp > ?'
                params = [cutoff_time.isoformat()]
                
                if severity:
                    query += ' AND severity = ?'
                    params.append(severity)
                
                query += ' ORDER BY timestamp DESC'
                
                cursor.execute(query, params)
                alerts = cursor.fetchall()
                
                # Convert to dictionaries
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, alert)) for alert in alerts]
                
        except Exception as e:
            self.logger.error(f"Failed to get alerts: {e}")
            return []
    
    def close(self):
        """Close the performance monitor"""
        self.stop_monitoring()
        self.logger.info("Performance monitor closed")

# Example usage and testing
def main():
    """Main entry point for testing"""
    import time
    import random
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate some trades
        symbols = ['BTC/USD', 'ETH/USD']
        timeframes = ['1h', '4h']
        
        for i in range(10):
            trade = TradeRecord(
                timestamp=datetime.now(),
                symbol=random.choice(symbols),
                timeframe=random.choice(timeframes),
                prediction=random.uniform(0.3, 0.8),
                confidence=random.uniform(0.5, 0.9),
                actual_result=random.uniform(0.2, 0.9),
                profit_loss=random.uniform(-0.03, 0.04),
                trade_id=f"test_trade_{i}",
                model_used="test_model",
                market_conditions={'volatility': random.uniform(0.1, 0.5)}
            )
            
            monitor.record_trade(trade)
            time.sleep(1)
        
        # Get performance summary
        performance = monitor.get_recent_performance(1)
        print(json.dumps(performance, indent=2))
        
        # Get alerts
        alerts = monitor.get_alerts(1)
        print(f"Generated {len(alerts)} alerts")
        
        time.sleep(5)  # Let monitoring run for a bit
        
    finally:
        monitor.close()

if __name__ == "__main__":
    main()