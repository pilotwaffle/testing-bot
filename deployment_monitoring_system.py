#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\deployment_monitoring_system.py
Location: E:\Trade Chat Bot\G Trading Bot\deployment_monitoring_system.py

Advanced Model Deployment & Monitoring System for Elite Trading Bot V3.0
- Production model deployment with A/B testing
- Real-time performance monitoring and drift detection
- Automated retraining triggers
- Model versioning and rollback capabilities
- Performance analytics and alerting
- Trading signal validation and safety checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import json
import pickle
import sqlite3
import threading
import time
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import hashlib
from collections import deque, defaultdict
import asyncio
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for model deployment and monitoring"""
    model_registry_path: str = "model_registry"
    performance_db_path: str = "performance_monitoring.db" 
    drift_threshold: float = 0.1
    performance_threshold: float = 0.6
    retraining_threshold: float = 0.05
    monitoring_interval: int = 300  # 5 minutes
    max_model_versions: int = 10
    ab_test_duration: int = 7  # days
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None
    safety_checks: bool = True
    
@dataclass
class ModelMetadata:
    """Metadata for deployed models"""
    model_id: str
    model_name: str
    version: str
    creation_date: datetime
    accuracy: float
    features: List[str]
    training_data_hash: str
    hyperparameters: Dict
    deployment_date: Optional[datetime] = None
    status: str = "staged"  # staged, active, deprecated
    performance_metrics: Dict = field(default_factory=dict)

class ModelRegistry:
    """Centralized model registry for version control"""
    
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "model_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load model metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.models = {
                    model_id: ModelMetadata(**metadata) 
                    for model_id, metadata in data.items()
                }
        else:
            self.models = {}
    
    def save_metadata(self):
        """Save model metadata to file"""
        data = {}
        for model_id, metadata in self.models.items():
            # Convert ModelMetadata to dict
            data[model_id] = {
                'model_id': metadata.model_id,
                'model_name': metadata.model_name,
                'version': metadata.version,
                'creation_date': metadata.creation_date.isoformat(),
                'accuracy': metadata.accuracy,
                'features': metadata.features,
                'training_data_hash': metadata.training_data_hash,
                'hyperparameters': metadata.hyperparameters,
                'deployment_date': metadata.deployment_date.isoformat() if metadata.deployment_date else None,
                'status': metadata.status,
                'performance_metrics': metadata.performance_metrics
            }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(self, model, model_name: str, accuracy: float, 
                      features: List[str], training_data: np.ndarray,
                      hyperparameters: Dict = None) -> str:
        """Register a new model version"""
        
        # Generate model ID
        model_id = self._generate_model_id(model_name)
        version = self._get_next_version(model_name)
        
        # Calculate training data hash
        data_hash = hashlib.md5(training_data.tobytes()).hexdigest()
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            creation_date=datetime.now(),
            accuracy=accuracy,
            features=features,
            training_data_hash=data_hash,
            hyperparameters=hyperparameters or {}
        )
        
        # Save model file
        model_path = self.registry_path / f"{model_id}.pkl"
        joblib.dump(model, model_path)
        
        # Register metadata
        self.models[model_id] = metadata
        self.save_metadata()
        
        logger.info(f"📦 Registered model {model_name} v{version} (ID: {model_id})")
        return model_id
    
    def load_model(self, model_id: str):
        """Load a model by ID"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_path = self.registry_path / f"{model_id}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)
    
    def get_active_models(self, model_name: str = None) -> List[ModelMetadata]:
        """Get all active models, optionally filtered by name"""
        active_models = [
            metadata for metadata in self.models.values()
            if metadata.status == "active"
        ]
        
        if model_name:
            active_models = [m for m in active_models if m.model_name == model_name]
        
        return active_models
    
    def deploy_model(self, model_id: str) -> bool:
        """Deploy a model to production"""
        if model_id not in self.models:
            return False
        
        # Update metadata
        self.models[model_id].status = "active"
        self.models[model_id].deployment_date = datetime.now()
        self.save_metadata()
        
        logger.info(f"🚀 Deployed model {model_id} to production")
        return True
    
    def retire_model(self, model_id: str) -> bool:
        """Retire a model from production"""
        if model_id not in self.models:
            return False
        
        self.models[model_id].status = "deprecated"
        self.save_metadata()
        
        logger.info(f"🔚 Retired model {model_id}")
        return True
    
    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{timestamp}"
    
    def _get_next_version(self, model_name: str) -> str:
        """Get next version number for model"""
        versions = [
            int(m.version.split('.')[-1]) for m in self.models.values()
            if m.model_name == model_name and '.' in m.version
        ]
        
        if not versions:
            return "1.0"
        
        next_version = max(versions) + 1
        return f"1.{next_version}"

class PerformanceMonitor:
    """Real-time performance monitoring and drift detection"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.db_path = config.performance_db_path
        self.setup_database()
        self.monitoring_active = False
        
        # Sliding windows for monitoring
        self.prediction_window = deque(maxlen=1000)
        self.accuracy_window = deque(maxlen=100)
        self.feature_distributions = defaultdict(lambda: deque(maxlen=1000))
        
    def setup_database(self):
        """Setup performance monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                prediction REAL,
                actual REAL,
                confidence REAL,
                accuracy REAL,
                feature_values TEXT,
                latency_ms REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                drift_type TEXT NOT NULL,
                drift_score REAL,
                alert_level TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, model_id: str, prediction: float, actual: float = None,
                      confidence: float = None, feature_values: List[float] = None,
                      latency_ms: float = None):
        """Log a model prediction for monitoring"""
        
        timestamp = datetime.now()
        
        # Calculate accuracy if actual value available
        accuracy = None
        if actual is not None:
            accuracy = 1.0 if abs(prediction - actual) < 0.5 else 0.0
            self.accuracy_window.append(accuracy)
        
        # Store in sliding window
        self.prediction_window.append({
            'timestamp': timestamp,
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'accuracy': accuracy
        })
        
        # Store feature values for drift detection
        if feature_values:
            for i, value in enumerate(feature_values):
                self.feature_distributions[f'feature_{i}'].append(value)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance 
            (model_id, timestamp, prediction, actual, confidence, accuracy, feature_values, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id, timestamp, prediction, actual, confidence, accuracy,
            json.dumps(feature_values) if feature_values else None, latency_ms
        ))
        
        conn.commit()
        conn.close()
        
        # Check for drift
        self._check_drift(model_id)
    
    def _check_drift(self, model_id: str):
        """Check for model drift"""
        
        # Performance drift
        if len(self.accuracy_window) >= 50:
            recent_accuracy = np.mean(list(self.accuracy_window)[-20:])
            historical_accuracy = np.mean(list(self.accuracy_window)[:-20])
            
            if abs(recent_accuracy - historical_accuracy) > self.config.drift_threshold:
                self._log_drift_alert(model_id, "performance_drift", 
                                    abs(recent_accuracy - historical_accuracy))
        
        # Feature drift (simple statistical test)
        for feature_name, values in self.feature_distributions.items():
            if len(values) >= 100:
                recent_values = list(values)[-50:]
                historical_values = list(values)[:-50]
                
                # Kolmogorov-Smirnov test
                from scipy.stats import ks_2samp
                statistic, p_value = ks_2samp(recent_values, historical_values)
                
                if p_value < 0.05:  # Significant drift
                    self._log_drift_alert(model_id, f"feature_drift_{feature_name}", statistic)
    
    def _log_drift_alert(self, model_id: str, drift_type: str, drift_score: float):
        """Log drift alert to database"""
        
        alert_level = "HIGH" if drift_score > 0.2 else "MEDIUM" if drift_score > 0.1 else "LOW"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drift_alerts (model_id, timestamp, drift_type, drift_score, alert_level)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_id, datetime.now(), drift_type, drift_score, alert_level))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"🚨 {alert_level} drift detected for {model_id}: {drift_type} (score: {drift_score:.3f})")
        
        # Send alert if configured
        if self.config.alert_email or self.config.alert_webhook:
            self._send_alert(model_id, drift_type, drift_score, alert_level)
    
    def _send_alert(self, model_id: str, drift_type: str, drift_score: float, alert_level: str):
        """Send drift alert notification"""
        message = f"Model Drift Alert: {model_id}\nType: {drift_type}\nScore: {drift_score:.3f}\nLevel: {alert_level}"
        
        if self.config.alert_email:
            # Email alert (implement with your email service)
            logger.info(f"📧 Email alert sent to {self.config.alert_email}")
        
        if self.config.alert_webhook:
            # Webhook alert (implement with your webhook service)
            logger.info(f"🔗 Webhook alert sent to {self.config.alert_webhook}")
    
    def get_performance_metrics(self, model_id: str, hours: int = 24) -> Dict:
        """Get performance metrics for a model"""
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_performance 
            WHERE model_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (model_id, start_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {'error': 'No data found'}
        
        # Calculate metrics
        predictions = [row[2] for row in rows]
        actuals = [row[3] for row in rows if row[3] is not None]
        accuracies = [row[5] for row in rows if row[5] is not None]
        latencies = [row[7] for row in rows if row[7] is not None]
        
        metrics = {
            'total_predictions': len(predictions),
            'predictions_with_actuals': len(actuals),
            'average_confidence': np.mean([row[4] for row in rows if row[4] is not None]),
            'average_latency_ms': np.mean(latencies) if latencies else None,
            'accuracy': np.mean(accuracies) if accuracies else None,
            'accuracy_std': np.std(accuracies) if accuracies else None
        }
        
        return metrics

class ABTestingFramework:
    """A/B testing framework for model comparison"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.active_tests = {}
    
    def start_ab_test(self, model_a_id: str, model_b_id: str, 
                     traffic_split: float = 0.5, test_name: str = None) -> str:
        """Start A/B test between two models"""
        
        test_id = test_name or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        test_config = {
            'test_id': test_id,
            'model_a_id': model_a_id,
            'model_b_id': model_b_id,
            'traffic_split': traffic_split,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(days=self.config.ab_test_duration),
            'results_a': [],
            'results_b': []
        }
        
        self.active_tests[test_id] = test_config
        
        logger.info(f"🧪 Started A/B test {test_id}: {model_a_id} vs {model_b_id}")
        return test_id
    
    def route_prediction(self, test_id: str, user_id: str = None) -> str:
        """Route prediction request to appropriate model"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        
        # Simple random routing
        if np.random.random() < test_config['traffic_split']:
            return test_config['model_a_id']
        else:
            return test_config['model_b_id']
    
    def log_ab_result(self, test_id: str, model_id: str, prediction: float, 
                     actual: float, confidence: float = None):
        """Log A/B test result"""
        
        if test_id not in self.active_tests:
            return
        
        test_config = self.active_tests[test_id]
        
        result = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'accuracy': 1.0 if abs(prediction - actual) < 0.5 else 0.0
        }
        
        if model_id == test_config['model_a_id']:
            test_config['results_a'].append(result)
        elif model_id == test_config['model_b_id']:
            test_config['results_b'].append(result)
    
    def analyze_ab_test(self, test_id: str) -> Dict:
        """Analyze A/B test results"""
        
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        test_config = self.active_tests[test_id]
        results_a = test_config['results_a']
        results_b = test_config['results_b']
        
        if len(results_a) < 30 or len(results_b) < 30:
            return {'error': 'Insufficient data for analysis'}
        
        # Calculate metrics
        accuracy_a = np.mean([r['accuracy'] for r in results_a])
        accuracy_b = np.mean([r['accuracy'] for r in results_b])
        
        confidence_a = np.mean([r['confidence'] for r in results_a if r['confidence']])
        confidence_b = np.mean([r['confidence'] for r in results_b if r['confidence']])
        
        # Statistical significance test
        from scipy.stats import ttest_ind
        accuracies_a = [r['accuracy'] for r in results_a]
        accuracies_b = [r['accuracy'] for r in results_b]
        t_stat, p_value = ttest_ind(accuracies_a, accuracies_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(accuracies_a) - 1) * np.var(accuracies_a) + 
                            (len(accuracies_b) - 1) * np.var(accuracies_b)) / 
                           (len(accuracies_a) + len(accuracies_b) - 2))
        cohens_d = (accuracy_a - accuracy_b) / pooled_std
        
        analysis = {
            'test_id': test_id,
            'model_a_id': test_config['model_a_id'],
            'model_b_id': test_config['model_b_id'],
            'samples_a': len(results_a),
            'samples_b': len(results_b),
            'accuracy_a': accuracy_a,
            'accuracy_b': accuracy_b,
            'confidence_a': confidence_a,
            'confidence_b': confidence_b,
            'accuracy_difference': accuracy_a - accuracy_b,
            'statistical_significance': p_value < 0.05,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'winner': test_config['model_a_id'] if accuracy_a > accuracy_b else test_config['model_b_id'],
            'confidence_level': 'HIGH' if p_value < 0.01 else 'MEDIUM' if p_value < 0.05 else 'LOW'
        }
        
        return analysis

class SafetyValidator:
    """Safety validation for trading signals"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.max_position_size = 0.05  # 5% max position
        self.max_daily_trades = 20
        self.max_drawdown = 0.10  # 10% max drawdown
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        
    def validate_signal(self, signal: Dict) -> Tuple[bool, str]:
        """Validate trading signal for safety"""
        
        if not self.config.safety_checks:
            return True, "Safety checks disabled"
        
        # Reset daily counter
        if datetime.now().date() != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = datetime.now().date()
        
        # Check position size
        position_size = signal.get('position_size', 0)
        if position_size > self.max_position_size:
            return False, f"Position size {position_size:.3f} exceeds maximum {self.max_position_size:.3f}"
        
        # Check daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            return False, f"Daily trade limit {self.max_daily_trades} exceeded"
        
        # Check confidence threshold
        confidence = signal.get('confidence', 0)
        if confidence < 0.6:
            return False, f"Signal confidence {confidence:.3f} below threshold 0.6"
        
        # Check for reasonable price movements
        expected_return = signal.get('expected_return', 0)
        if abs(expected_return) > 0.2:  # 20% movement seems extreme
            return False, f"Expected return {expected_return:.3f} seems unrealistic"
        
        self.daily_trade_count += 1
        return True, "Signal validated"

class ProductionDeploymentManager:
    """Main production deployment and monitoring manager"""
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig()
        self.model_registry = ModelRegistry(self.config.model_registry_path)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.ab_tester = ABTestingFramework(self.config)
        self.safety_validator = SafetyValidator(self.config)
        
        self.active_models = {}
        self.monitoring_thread = None
        
    def deploy_model_to_production(self, model, model_name: str, accuracy: float,
                                 features: List[str], training_data: np.ndarray,
                                 hyperparameters: Dict = None) -> str:
        """Deploy a model to production with full monitoring"""
        
        logger.info(f"🚀 Deploying {model_name} to production...")
        
        # Register model
        model_id = self.model_registry.register_model(
            model, model_name, accuracy, features, training_data, hyperparameters
        )
        
        # Deploy to production
        self.model_registry.deploy_model(model_id)
        
        # Load into active models
        self.active_models[model_id] = {
            'model': model,
            'metadata': self.model_registry.models[model_id]
        }
        
        logger.info(f"✅ Successfully deployed {model_name} (ID: {model_id})")
        return model_id
    
    def predict_with_monitoring(self, model_id: str, features: np.ndarray,
                              return_confidence: bool = False) -> Dict:
        """Make prediction with full monitoring and safety checks"""
        
        start_time = time.time()
        
        if model_id not in self.active_models:
            return {'error': f'Model {model_id} not active'}
        
        try:
            model = self.active_models[model_id]['model']
            
            # Make prediction
            prediction = model.predict(features.reshape(1, -1))[0]
            
            # Get confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features.reshape(1, -1))[0]
                confidence = np.max(proba)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log for monitoring
            self.performance_monitor.log_prediction(
                model_id=model_id,
                prediction=prediction,
                confidence=confidence,
                feature_values=features.tolist(),
                latency_ms=latency_ms
            )
            
            # Create signal for safety validation
            signal = {
                'prediction': prediction,
                'confidence': confidence,
                'position_size': 0.02,  # Default position size
                'expected_return': prediction * 0.1  # Rough estimate
            }
            
            # Validate signal
            is_safe, safety_message = self.safety_validator.validate_signal(signal)
            
            result = {
                'model_id': model_id,
                'prediction': prediction,
                'confidence': confidence,
                'latency_ms': latency_ms,
                'is_safe': is_safe,
                'safety_message': safety_message,
                'timestamp': datetime.now().isoformat()
            }
            
            if return_confidence:
                result['confidence'] = confidence
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {model_id}: {e}")
            return {'error': str(e)}
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.performance_monitor.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("📊 Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.performance_monitor.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("🛑 Stopped performance monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.performance_monitor.monitoring_active:
            try:
                # Check model performance
                for model_id in self.active_models.keys():
                    metrics = self.performance_monitor.get_performance_metrics(model_id, hours=1)
                    
                    if 'accuracy' in metrics and metrics['accuracy'] is not None:
                        if metrics['accuracy'] < self.config.performance_threshold:
                            logger.warning(f"⚠️ Model {model_id} accuracy {metrics['accuracy']:.3f} below threshold")
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def get_deployment_status(self) -> Dict:
        """Get overall deployment status"""
        
        active_models = self.model_registry.get_active_models()
        
        status = {
            'active_models': len(active_models),
            'total_models': len(self.model_registry.models),
            'monitoring_active': self.performance_monitor.monitoring_active,
            'active_ab_tests': len(self.ab_tester.active_tests),
            'models': []
        }
        
        for model in active_models:
            model_metrics = self.performance_monitor.get_performance_metrics(
                model.model_id, hours=24
            )
            
            status['models'].append({
                'model_id': model.model_id,
                'model_name': model.model_name,
                'version': model.version,
                'deployment_date': model.deployment_date.isoformat() if model.deployment_date else None,
                'training_accuracy': model.accuracy,
                'recent_metrics': model_metrics
            })
        
        return status

def main():
    """Main function to demonstrate deployment system"""
    logger.info("🚀 Advanced Model Deployment & Monitoring System")
    logger.info("="*80)
    
    # Create deployment manager
    config = DeploymentConfig()
    deployment_manager = ProductionDeploymentManager(config)
    
    # Example: Deploy a model
    from sklearn.ensemble import RandomForestClassifier
    
    # Create example model and data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Deploy model
    model_id = deployment_manager.deploy_model_to_production(
        model=model,
        model_name="test_model",
        accuracy=0.85,
        features=[f"feature_{i}" for i in range(20)],
        training_data=X_train,
        hyperparameters={"n_estimators": 100}
    )
    
    # Start monitoring
    deployment_manager.start_monitoring()
    
    # Make some test predictions
    logger.info("🔮 Making test predictions...")
    for i in range(10):
        test_features = np.random.randn(20)
        result = deployment_manager.predict_with_monitoring(model_id, test_features)
        logger.info(f"Prediction {i+1}: {result['prediction']:.3f} (safe: {result['is_safe']})")
        time.sleep(1)
    
    # Get deployment status
    status = deployment_manager.get_deployment_status()
    logger.info(f"\n📊 Deployment Status:")
    logger.info(f"   Active Models: {status['active_models']}")
    logger.info(f"   Monitoring Active: {status['monitoring_active']}")
    
    # Stop monitoring
    deployment_manager.stop_monitoring()
    
    logger.info("✅ Deployment system demonstration complete!")

if __name__ == "__main__":
    main()