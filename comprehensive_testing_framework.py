#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\comprehensive_testing_framework.py
Location: E:\Trade Chat Bot\G Trading Bot\comprehensive_testing_framework.py

Comprehensive ML Testing & Validation Framework for Elite Trading Bot V3.0
- Advanced cross-validation and backtesting
- Statistical significance testing
- Model stability and robustness analysis
- Performance monitoring and drift detection
- A/B testing framework for model comparison
- Monte Carlo simulation for risk assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (TimeSeriesSplit, StratifiedKFold, cross_validate,
                                   validation_curve, learning_curve)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           confusion_matrix, classification_report)
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
import joblib
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestingConfig:
    """Configuration for comprehensive testing"""
    significance_level: float = 0.05
    cv_folds: int = 10
    monte_carlo_runs: int = 1000
    stability_window: int = 100
    drift_threshold: float = 0.1
    min_test_samples: int = 50
    bootstrap_samples: int = 1000
    performance_threshold: float = 0.65
    
class StatisticalTester:
    """Advanced statistical testing for ML models"""
    
    def __init__(self, config: TestingConfig):
        self.config = config
        self.test_results = {}
        
    def comprehensive_model_validation(self, models: Dict, X: np.ndarray, y: np.ndarray,
                                     model_names: List[str] = None) -> Dict:
        """Comprehensive validation of multiple models"""
        logger.info("🔬 Starting comprehensive model validation...")
        
        if model_names is None:
            model_names = list(models.keys())
        
        validation_results = {}
        
        for name in model_names:
            if name not in models:
                continue
                
            logger.info(f"🧪 Validating {name}...")
            model = models[name]
            
            # Time series cross-validation
            cv_results = self._time_series_cross_validation(model, X, y, name)
            
            # Stability analysis
            stability_results = self._model_stability_analysis(model, X, y, name)
            
            # Performance consistency
            consistency_results = self._performance_consistency_test(model, X, y, name)
            
            # Robustness testing
            robustness_results = self._robustness_testing(model, X, y, name)
            
            # Feature importance stability
            feature_importance = self._feature_importance_analysis(model, X, y, name)
            
            validation_results[name] = {
                'cv_results': cv_results,
                'stability': stability_results,
                'consistency': consistency_results,
                'robustness': robustness_results,
                'feature_importance': feature_importance,
                'overall_score': self._calculate_overall_score(
                    cv_results, stability_results, consistency_results, robustness_results
                )
            }
            
        # Model comparison
        comparison_results = self._statistical_model_comparison(validation_results, X, y, models)
        
        return {
            'individual_results': validation_results,
            'comparison': comparison_results,
            'recommendations': self._generate_recommendations(validation_results, comparison_results)
        }
    
    def _time_series_cross_validation(self, model, X: np.ndarray, y: np.ndarray, 
                                    model_name: str) -> Dict:
        """Advanced time series cross-validation"""
        logger.info(f"📊 Time series CV for {model_name}...")
        
        # Multiple CV strategies
        cv_strategies = {
            'time_series_split': TimeSeriesSplit(n_splits=self.config.cv_folds),
            'purged_cv': self._create_purged_cv(X, gap_size=5),
            'blocked_cv': self._create_blocked_cv(X, block_size=20)
        }
        
        cv_results = {}
        
        for cv_name, cv_splitter in cv_strategies.items():
            scores = cross_validate(
                model, X, y, cv=cv_splitter,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                return_train_score=True, n_jobs=-1
            )
            
            cv_results[cv_name] = {
                'test_accuracy': scores['test_accuracy'],
                'test_precision': scores['test_precision'], 
                'test_recall': scores['test_recall'],
                'test_f1': scores['test_f1'],
                'test_roc_auc': scores['test_roc_auc'],
                'train_accuracy': scores['train_accuracy'],
                'fit_time': scores['fit_time'],
                'score_time': scores['score_time']
            }
            
            # Calculate stability metrics
            cv_results[cv_name]['stability'] = {
                'accuracy_std': np.std(scores['test_accuracy']),
                'accuracy_cv': np.std(scores['test_accuracy']) / np.mean(scores['test_accuracy']),
                'overfitting': np.mean(scores['train_accuracy']) - np.mean(scores['test_accuracy'])
            }
        
        return cv_results
    
    def _create_purged_cv(self, X: np.ndarray, gap_size: int = 5):
        """Create purged cross-validation to prevent data leakage"""
        class PurgedTimeSeriesSplit:
            def __init__(self, n_splits=5, gap_size=5):
                self.n_splits = n_splits
                self.gap_size = gap_size
            
            def split(self, X, y=None, groups=None):
                n_samples = len(X)
                fold_size = n_samples // (self.n_splits + 1)
                
                for i in range(self.n_splits):
                    # Training set
                    train_start = 0
                    train_end = (i + 1) * fold_size
                    
                    # Test set (with gap)
                    test_start = train_end + self.gap_size
                    test_end = test_start + fold_size
                    
                    if test_end <= n_samples:
                        train_indices = np.arange(train_start, train_end)
                        test_indices = np.arange(test_start, test_end)
                        yield train_indices, test_indices
        
        return PurgedTimeSeriesSplit(self.config.cv_folds, gap_size)
    
    def _create_blocked_cv(self, X: np.ndarray, block_size: int = 20):
        """Create blocked cross-validation"""
        class BlockedTimeSeriesSplit:
            def __init__(self, n_splits=5, block_size=20):
                self.n_splits = n_splits
                self.block_size = block_size
            
            def split(self, X, y=None, groups=None):
                n_samples = len(X)
                n_blocks = n_samples // self.block_size
                blocks_per_fold = n_blocks // (self.n_splits + 1)
                
                for i in range(self.n_splits):
                    # Training blocks
                    train_blocks = range(0, (i + 1) * blocks_per_fold)
                    train_indices = []
                    for block in train_blocks:
                        start = block * self.block_size
                        end = min((block + 1) * self.block_size, n_samples)
                        train_indices.extend(range(start, end))
                    
                    # Test blocks
                    test_blocks = range((i + 1) * blocks_per_fold, (i + 2) * blocks_per_fold)
                    test_indices = []
                    for block in test_blocks:
                        start = block * self.block_size
                        end = min((block + 1) * self.block_size, n_samples)
                        test_indices.extend(range(start, end))
                    
                    if len(test_indices) > 0:
                        yield np.array(train_indices), np.array(test_indices)
        
        return BlockedTimeSeriesSplit(self.config.cv_folds, block_size)
    
    def _model_stability_analysis(self, model, X: np.ndarray, y: np.ndarray, 
                                model_name: str) -> Dict:
        """Analyze model stability across different data subsets"""
        logger.info(f"📈 Stability analysis for {model_name}...")
        
        n_samples = len(X)
        window_size = min(self.config.stability_window, n_samples // 3)
        step_size = window_size // 4
        
        stability_scores = []
        window_predictions = []
        
        # Rolling window analysis
        for start in range(0, n_samples - window_size, step_size):
            end = start + window_size
            
            # Split window into train/test
            split_point = start + int(window_size * 0.8)
            
            X_train_window = X[start:split_point]
            y_train_window = y[start:split_point]
            X_test_window = X[split_point:end]
            y_test_window = y[split_point:end]
            
            if len(X_train_window) > 10 and len(X_test_window) > 5:
                try:
                    # Clone and fit model
                    from sklearn.base import clone
                    window_model = clone(model)
                    window_model.fit(X_train_window, y_train_window)
                    
                    # Predict and score
                    predictions = window_model.predict(X_test_window)
                    score = accuracy_score(y_test_window, predictions)
                    
                    stability_scores.append(score)
                    window_predictions.append(predictions)
                    
                except Exception as e:
                    logger.warning(f"Window analysis failed: {e}")
                    continue
        
        if len(stability_scores) < 3:
            return {'error': 'Insufficient data for stability analysis'}
        
        # Calculate stability metrics
        stability_metrics = {
            'mean_accuracy': np.mean(stability_scores),
            'std_accuracy': np.std(stability_scores),
            'min_accuracy': np.min(stability_scores),
            'max_accuracy': np.max(stability_scores),
            'coefficient_of_variation': np.std(stability_scores) / np.mean(stability_scores),
            'stability_score': 1 - (np.std(stability_scores) / np.mean(stability_scores)),
            'num_windows': len(stability_scores)
        }
        
        # Trend analysis
        if len(stability_scores) > 5:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(stability_scores)), stability_scores
            )
            stability_metrics['trend_slope'] = slope
            stability_metrics['trend_r_squared'] = r_value ** 2
            stability_metrics['trend_p_value'] = p_value
        
        return stability_metrics
    
    def _performance_consistency_test(self, model, X: np.ndarray, y: np.ndarray,
                                    model_name: str) -> Dict:
        """Test performance consistency across different market conditions"""
        logger.info(f"🎯 Consistency testing for {model_name}...")
        
        # Bootstrap testing
        bootstrap_scores = []
        n_samples = len(X)
        
        for _ in range(self.config.bootstrap_samples):
            # Random sampling with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Split into train/test
            split_point = int(len(X_bootstrap) * 0.8)
            X_train = X_bootstrap[:split_point]
            y_train = y_bootstrap[:split_point]
            X_test = X_bootstrap[split_point:]
            y_test = y_bootstrap[split_point:]
            
            if len(X_test) > 5:
                try:
                    from sklearn.base import clone
                    bootstrap_model = clone(model)
                    bootstrap_model.fit(X_train, y_train)
                    predictions = bootstrap_model.predict(X_test)
                    score = accuracy_score(y_test, predictions)
                    bootstrap_scores.append(score)
                except:
                    continue
        
        if len(bootstrap_scores) < 100:
            return {'error': 'Insufficient bootstrap samples'}
        
        # Calculate confidence intervals
        bootstrap_scores = np.array(bootstrap_scores)
        confidence_level = 1 - self.config.significance_level
        
        lower_percentile = ((1 - confidence_level) / 2) * 100
        upper_percentile = (confidence_level + (1 - confidence_level) / 2) * 100
        
        consistency_metrics = {
            'bootstrap_mean': np.mean(bootstrap_scores),
            'bootstrap_std': np.std(bootstrap_scores),
            'confidence_interval_lower': np.percentile(bootstrap_scores, lower_percentile),
            'confidence_interval_upper': np.percentile(bootstrap_scores, upper_percentile),
            'bootstrap_samples': len(bootstrap_scores),
            'consistency_score': 1 / (1 + np.std(bootstrap_scores))
        }
        
        return consistency_metrics
    
    def _robustness_testing(self, model, X: np.ndarray, y: np.ndarray,
                          model_name: str) -> Dict:
        """Test model robustness to various perturbations"""
        logger.info(f"🛡️ Robustness testing for {model_name}...")
        
        # Baseline performance
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        try:
            from sklearn.base import clone
            baseline_model = clone(model)
            baseline_model.fit(X_train, y_train)
            baseline_score = accuracy_score(y_test, baseline_model.predict(X_test))
        except:
            return {'error': 'Failed to establish baseline'}
        
        robustness_results = {'baseline_accuracy': baseline_score}
        
        # Noise robustness
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        noise_scores = []
        
        for noise_level in noise_levels:
            try:
                # Add Gaussian noise
                X_test_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
                noisy_score = accuracy_score(y_test, baseline_model.predict(X_test_noisy))
                noise_scores.append(noisy_score)
            except:
                noise_scores.append(0)
        
        robustness_results['noise_robustness'] = {
            'noise_levels': noise_levels,
            'scores': noise_scores,
            'robustness_score': np.mean([max(0, score/baseline_score) for score in noise_scores])
        }
        
        # Feature subset robustness
        n_features = X.shape[1]
        subset_sizes = [0.9, 0.8, 0.7, 0.6, 0.5]
        subset_scores = []
        
        for subset_size in subset_sizes:
            try:
                n_select = int(n_features * subset_size)
                selected_features = np.random.choice(n_features, size=n_select, replace=False)
                
                X_train_subset = X_train[:, selected_features]
                X_test_subset = X_test[:, selected_features]
                
                subset_model = clone(model)
                subset_model.fit(X_train_subset, y_train)
                subset_score = accuracy_score(y_test, subset_model.predict(X_test_subset))
                subset_scores.append(subset_score)
            except:
                subset_scores.append(0)
        
        robustness_results['feature_robustness'] = {
            'subset_sizes': subset_sizes,
            'scores': subset_scores,
            'robustness_score': np.mean([max(0, score/baseline_score) for score in subset_scores])
        }
        
        # Calculate overall robustness
        overall_robustness = (
            robustness_results['noise_robustness']['robustness_score'] * 0.6 +
            robustness_results['feature_robustness']['robustness_score'] * 0.4
        )
        robustness_results['overall_robustness'] = overall_robustness
        
        return robustness_results
    
    def _feature_importance_analysis(self, model, X: np.ndarray, y: np.ndarray,
                                   model_name: str) -> Dict:
        """Analyze feature importance stability"""
        logger.info(f"🔍 Feature importance analysis for {model_name}...")
        
        try:
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                base_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                base_importance = np.abs(model.coef_[0])
            else:
                return {'error': 'Model does not support feature importance'}
            
            # Bootstrap feature importance
            n_bootstrap = 50
            importance_matrix = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
                
                try:
                    from sklearn.base import clone
                    boot_model = clone(model)
                    boot_model.fit(X_boot, y_boot)
                    
                    if hasattr(boot_model, 'feature_importances_'):
                        importance = boot_model.feature_importances_
                    elif hasattr(boot_model, 'coef_'):
                        importance = np.abs(boot_model.coef_[0])
                    else:
                        continue
                    
                    importance_matrix.append(importance)
                except:
                    continue
            
            if len(importance_matrix) < 10:
                return {'error': 'Insufficient bootstrap samples for feature importance'}
            
            importance_matrix = np.array(importance_matrix)
            
            # Calculate stability metrics
            feature_importance_results = {
                'mean_importance': np.mean(importance_matrix, axis=0),
                'std_importance': np.std(importance_matrix, axis=0),
                'cv_importance': np.std(importance_matrix, axis=0) / (np.mean(importance_matrix, axis=0) + 1e-8),
                'stability_score': 1 - np.mean(np.std(importance_matrix, axis=0) / (np.mean(importance_matrix, axis=0) + 1e-8)),
                'top_features': np.argsort(np.mean(importance_matrix, axis=0))[-10:].tolist(),
                'bootstrap_samples': len(importance_matrix)
            }
            
            return feature_importance_results
            
        except Exception as e:
            return {'error': f'Feature importance analysis failed: {str(e)}'}
    
    def _calculate_overall_score(self, cv_results: Dict, stability_results: Dict,
                               consistency_results: Dict, robustness_results: Dict) -> float:
        """Calculate overall model quality score"""
        
        scores = []
        weights = []
        
        # CV accuracy (40% weight)
        if 'time_series_split' in cv_results:
            cv_accuracy = np.mean(cv_results['time_series_split']['test_accuracy'])
            scores.append(cv_accuracy)
            weights.append(0.4)
        
        # Stability (25% weight)
        if 'stability_score' in stability_results:
            scores.append(stability_results['stability_score'])
            weights.append(0.25)
        
        # Consistency (20% weight)
        if 'consistency_score' in consistency_results:
            scores.append(consistency_results['consistency_score'])
            weights.append(0.2)
        
        # Robustness (15% weight)
        if 'overall_robustness' in robustness_results:
            scores.append(robustness_results['overall_robustness'])
            weights.append(0.15)
        
        if len(scores) == 0:
            return 0.0
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        overall_score = np.average(scores, weights=weights)
        return float(overall_score)
    
    def _statistical_model_comparison(self, validation_results: Dict, X: np.ndarray, 
                                    y: np.ndarray, models: Dict) -> Dict:
        """Statistical comparison between models"""
        logger.info("📊 Statistical model comparison...")
        
        model_names = list(validation_results.keys())
        comparison_results = {}
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                logger.info(f"🆚 Comparing {model1} vs {model2}...")
                
                # Get cross-validation scores
                try:
                    scores1 = validation_results[model1]['cv_results']['time_series_split']['test_accuracy']
                    scores2 = validation_results[model2]['cv_results']['time_series_split']['test_accuracy']
                    
                    # Statistical tests
                    t_stat, t_p_value = ttest_ind(scores1, scores2)
                    u_stat, u_p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')
                    ks_stat, ks_p_value = ks_2samp(scores1, scores2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1) + 
                                        (len(scores2) - 1) * np.var(scores2)) / 
                                       (len(scores1) + len(scores2) - 2))
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                    
                    comparison_key = f"{model1}_vs_{model2}"
                    comparison_results[comparison_key] = {
                        'mean_diff': np.mean(scores1) - np.mean(scores2),
                        't_test': {'statistic': t_stat, 'p_value': t_p_value},
                        'mann_whitney': {'statistic': u_stat, 'p_value': u_p_value},
                        'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p_value},
                        'cohens_d': cohens_d,
                        'significant': t_p_value < self.config.significance_level,
                        'better_model': model1 if np.mean(scores1) > np.mean(scores2) else model2
                    }
                    
                except Exception as e:
                    logger.warning(f"Comparison failed for {model1} vs {model2}: {e}")
                    continue
        
        return comparison_results
    
    def _generate_recommendations(self, validation_results: Dict, 
                                comparison_results: Dict) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Find best overall model
        best_model = max(validation_results.keys(), 
                        key=lambda x: validation_results[x]['overall_score'])
        best_score = validation_results[best_model]['overall_score']
        
        recommendations.append(f"🏆 Best overall model: {best_model} (score: {best_score:.3f})")
        
        # Performance recommendations
        high_performers = [name for name, results in validation_results.items()
                          if results['overall_score'] > self.config.performance_threshold]
        
        if len(high_performers) > 1:
            recommendations.append(f"✅ High performers: {', '.join(high_performers)}")
            recommendations.append("💡 Consider ensemble of high performers")
        elif len(high_performers) == 0:
            recommendations.append("⚠️ No models meet performance threshold")
            recommendations.append("📈 Consider feature engineering or different algorithms")
        
        # Stability recommendations
        stable_models = [name for name, results in validation_results.items()
                        if results.get('stability', {}).get('stability_score', 0) > 0.8]
        
        if stable_models:
            recommendations.append(f"🎯 Most stable models: {', '.join(stable_models)}")
        
        # Robustness recommendations
        robust_models = [name for name, results in validation_results.items()
                        if results.get('robustness', {}).get('overall_robustness', 0) > 0.8]
        
        if robust_models:
            recommendations.append(f"🛡️ Most robust models: {', '.join(robust_models)}")
        
        return recommendations

class AdvancedBacktester:
    """Advanced backtesting with realistic market simulation"""
    
    def __init__(self, config: TestingConfig):
        self.config = config
        
    def monte_carlo_backtest(self, model, X: np.ndarray, y: np.ndarray,
                           returns: np.ndarray = None) -> Dict:
        """Monte Carlo simulation of trading performance"""
        logger.info("🎰 Running Monte Carlo backtest simulation...")
        
        if returns is None:
            returns = np.random.normal(0.001, 0.02, len(y))  # Simulated returns
        
        simulation_results = []
        
        for run in range(self.config.monte_carlo_runs):
            # Random bootstrap of data
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sim = X[indices]
            y_sim = y[indices]
            returns_sim = returns[indices]
            
            # Split data
            split_point = int(len(X_sim) * 0.8)
            X_train, X_test = X_sim[:split_point], X_sim[split_point:]
            y_train, y_test = y_sim[:split_point], y_sim[split_point:]
            returns_test = returns_sim[split_point:]
            
            try:
                # Train model
                from sklearn.base import clone
                sim_model = clone(model)
                sim_model.fit(X_train, y_train)
                
                # Generate predictions
                predictions = sim_model.predict(X_test)
                probabilities = None
                if hasattr(sim_model, 'predict_proba'):
                    probabilities = sim_model.predict_proba(X_test)[:, 1]
                
                # Simulate trading
                trading_results = self._simulate_trading(
                    predictions, y_test, returns_test, probabilities
                )
                
                simulation_results.append(trading_results)
                
            except Exception as e:
                logger.warning(f"Simulation run {run} failed: {e}")
                continue
        
        # Aggregate results
        if len(simulation_results) == 0:
            return {'error': 'All simulations failed'}
        
        return self._aggregate_simulation_results(simulation_results)
    
    def _simulate_trading(self, predictions: np.ndarray, actual: np.ndarray,
                         returns: np.ndarray, probabilities: np.ndarray = None) -> Dict:
        """Simulate realistic trading based on predictions"""
        
        # Trading parameters
        transaction_cost = 0.001  # 0.1% per trade
        position_size = 0.02  # 2% of portfolio per trade
        max_positions = 10
        stop_loss = -0.05  # 5% stop loss
        take_profit = 0.10  # 10% take profit
        
        portfolio_value = 1.0  # Start with $1
        cash = 1.0
        positions = []
        trades = []
        daily_returns = []
        
        for i in range(len(predictions)):
            current_return = returns[i]
            
            # Close existing positions
            positions_to_close = []
            for pos in positions:
                pos['current_return'] = pos['entry_return'] + current_return
                
                # Check stop loss / take profit
                if (pos['current_return'] <= stop_loss or 
                    pos['current_return'] >= take_profit or
                    i - pos['entry_day'] >= 10):  # Max holding period
                    
                    positions_to_close.append(pos)
            
            # Close positions
            for pos in positions_to_close:
                exit_value = pos['size'] * (1 + pos['current_return']) * (1 - transaction_cost)
                cash += exit_value
                
                trades.append({
                    'entry_day': pos['entry_day'],
                    'exit_day': i,
                    'return': pos['current_return'],
                    'size': pos['size']
                })
                
                positions.remove(pos)
            
            # Open new positions
            if predictions[i] == 1 and len(positions) < max_positions and cash > position_size:
                # Adjust position size based on confidence if available
                if probabilities is not None:
                    confidence = probabilities[i]
                    adjusted_size = position_size * min(2.0, confidence * 2)  # Scale by confidence
                else:
                    adjusted_size = position_size
                
                if cash >= adjusted_size:
                    cash -= adjusted_size * (1 + transaction_cost)  # Deduct transaction cost
                    
                    positions.append({
                        'entry_day': i,
                        'entry_return': current_return,
                        'size': adjusted_size,
                        'current_return': 0
                    })
            
            # Calculate portfolio value
            positions_value = sum(pos['size'] * (1 + pos['current_return']) for pos in positions)
            portfolio_value = cash + positions_value
            
            # Calculate daily return
            if i > 0:
                daily_return = (portfolio_value / prev_portfolio_value) - 1
                daily_returns.append(daily_return)
            
            prev_portfolio_value = portfolio_value
        
        # Calculate performance metrics
        if len(daily_returns) > 0:
            total_return = portfolio_value - 1.0
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = (np.mean(daily_returns) * 252) / (volatility + 1e-8)
            max_drawdown = self._calculate_max_drawdown(daily_returns)
            
            # Win rate
            winning_trades = [t for t in trades if t['return'] > 0]
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': len(trades),
                'final_portfolio_value': portfolio_value,
                'daily_returns': daily_returns
            }
        else:
            return {
                'total_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': 0,
                'final_portfolio_value': 1.0,
                'daily_returns': []
            }
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
    
    def _aggregate_simulation_results(self, simulation_results: List[Dict]) -> Dict:
        """Aggregate Monte Carlo simulation results"""
        
        metrics = ['total_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 
                  'win_rate', 'num_trades', 'final_portfolio_value']
        
        aggregated = {}
        
        for metric in metrics:
            values = [result[metric] for result in simulation_results if metric in result]
            if len(values) > 0:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentile_5': np.percentile(values, 5),
                    'percentile_95': np.percentile(values, 95),
                    'median': np.median(values)
                }
        
        # Risk metrics
        total_returns = [result['total_return'] for result in simulation_results]
        aggregated['risk_metrics'] = {
            'probability_of_loss': sum(1 for r in total_returns if r < 0) / len(total_returns),
            'expected_shortfall': np.mean([r for r in total_returns if r < np.percentile(total_returns, 5)]),
            'value_at_risk_5': np.percentile(total_returns, 5)
        }
        
        return aggregated

def run_comprehensive_testing():
    """Main function to run comprehensive testing"""
    logger.info("🚀 Starting Comprehensive ML Testing Framework")
    logger.info("="*80)
    
    # Load test data (replace with your actual data loading)
    logger.info("📊 Loading test data...")
    
    # Example: Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # Create some example models
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'svm': SVC(probability=True, random_state=42)
    }
    
    # Initialize testing framework
    config = TestingConfig()
    tester = StatisticalTester(config)
    backtester = AdvancedBacktester(config)
    
    # Run comprehensive validation
    logger.info("🔬 Running comprehensive validation...")
    validation_results = tester.comprehensive_model_validation(models, X, y)
    
    # Print results
    logger.info("\n📊 COMPREHENSIVE TESTING RESULTS")
    logger.info("="*80)
    
    for model_name, results in validation_results['individual_results'].items():
        logger.info(f"\n🤖 {model_name.upper()}:")
        logger.info(f"   Overall Score: {results['overall_score']:.3f}")
        
        if 'cv_results' in results:
            cv_acc = np.mean(results['cv_results']['time_series_split']['test_accuracy'])
            logger.info(f"   CV Accuracy: {cv_acc:.3f}")
        
        if 'stability' in results and 'stability_score' in results['stability']:
            logger.info(f"   Stability: {results['stability']['stability_score']:.3f}")
        
        if 'robustness' in results and 'overall_robustness' in results['robustness']:
            logger.info(f"   Robustness: {results['robustness']['overall_robustness']:.3f}")
    
    # Print recommendations
    logger.info(f"\n💡 RECOMMENDATIONS:")
    for rec in validation_results['recommendations']:
        logger.info(f"   {rec}")
    
    # Run Monte Carlo backtest on best model
    best_model_name = max(validation_results['individual_results'].keys(),
                         key=lambda x: validation_results['individual_results'][x]['overall_score'])
    best_model = models[best_model_name]
    
    logger.info(f"\n🎰 Running Monte Carlo backtest for {best_model_name}...")
    backtest_results = backtester.monte_carlo_backtest(best_model, X, y)
    
    if 'error' not in backtest_results:
        logger.info(f"📈 BACKTEST RESULTS (1000 simulations):")
        logger.info(f"   Expected Return: {backtest_results['total_return']['mean']:.3f} ± {backtest_results['total_return']['std']:.3f}")
        logger.info(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']['mean']:.3f}")
        logger.info(f"   Max Drawdown: {backtest_results['max_drawdown']['mean']:.3f}")
        logger.info(f"   Win Rate: {backtest_results['win_rate']['mean']:.3f}")
        logger.info(f"   Probability of Loss: {backtest_results['risk_metrics']['probability_of_loss']:.3f}")
    
    logger.info("\n✅ Comprehensive testing complete!")
    
    return validation_results, backtest_results

if __name__ == "__main__":
    run_comprehensive_testing()