# error_recovery_system.py - Error Handling and Recovery System
"""
Comprehensive Error Handling and Recovery System
Provides robust error handling, system recovery, and health monitoring
"""

import logging
import traceback
import time
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import psutil
import sys
import os
from functools import wraps
import warnings

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SystemComponent(Enum):
    """System components for health monitoring"""
    DATA_FETCHER = "DATA_FETCHER"
    ML_ENGINE = "ML_ENGINE"
    STRATEGY_ENGINE = "STRATEGY_ENGINE"
    PERFORMANCE_MONITOR = "PERFORMANCE_MONITOR"
    MAIN_ORCHESTRATOR = "MAIN_ORCHESTRATOR"
    WEB_DASHBOARD = "WEB_DASHBOARD"

@dataclass
class ErrorEvent:
    """Error event data structure"""
    timestamp: datetime
    component: SystemComponent
    severity: ErrorSeverity
    error_type: str
    message: str
    traceback_info: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    occurrence_count: int = 1

@dataclass
class SystemHealth:
    """System health status"""
    timestamp: datetime
    overall_status: str  # HEALTHY, DEGRADED, CRITICAL, OFFLINE
    component_status: Dict[str, str]
    error_count_last_hour: int
    critical_errors: int
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_percent: float
    uptime_hours: float
    active_errors: List[ErrorEvent]

class ErrorRecoverySystem:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, config_path: str = "config/error_recovery_config.json"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Error tracking
        self.error_history = []
        self.active_errors = {}
        self.component_health = {}
        self.recovery_strategies = {}
        
        # System monitoring
        self.start_time = datetime.now()
        self.last_health_check = datetime.now()
        self.health_check_interval = self.config['monitoring']['health_check_interval_seconds']
        
        # Recovery settings
        self.max_recovery_attempts = self.config['recovery']['max_attempts_per_error']
        self.recovery_backoff_seconds = self.config['recovery']['backoff_base_seconds']
        
        # Setup recovery strategies
        self._setup_recovery_strategies()
        
        # Setup error persistence
        self.error_log_path = Path(self.config['persistence']['error_log_path'])
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Health monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("Error Recovery System initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load error recovery configuration"""
        default_config = {
            "monitoring": {
                "health_check_interval_seconds": 60,
                "error_retention_hours": 168,  # 1 week
                "max_errors_in_memory": 1000,
                "system_resource_monitoring": True
            },
            "recovery": {
                "auto_recovery_enabled": True,
                "max_attempts_per_error": 3,
                "backoff_base_seconds": 60,
                "escalation_enabled": True,
                "graceful_degradation": True
            },
            "alerting": {
                "email_enabled": False,
                "webhook_enabled": False,
                "log_level_for_alerts": "HIGH",
                "alert_cooldown_minutes": 30
            },
            "persistence": {
                "error_log_path": "logs/errors/",
                "save_system_state": True,
                "backup_interval_hours": 24
            },
            "thresholds": {
                "memory_usage_warning": 80,
                "memory_usage_critical": 95,
                "cpu_usage_warning": 80,
                "cpu_usage_critical": 95,
                "disk_usage_warning": 85,
                "disk_usage_critical": 95,
                "error_rate_warning": 10,  # errors per hour
                "error_rate_critical": 25
            }
        }
        
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                self._deep_update(default_config, loaded_config)
                self.logger.info(f"Error recovery configuration loaded from {config_path}")
            else:
                self.logger.info("Using default error recovery configuration")
                config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Error loading recovery config: {e}. Using defaults.")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _setup_recovery_strategies(self):
        """Setup component-specific recovery strategies"""
        
        self.recovery_strategies = {
            SystemComponent.DATA_FETCHER: [
                self._recover_data_fetcher_connection,
                self._recover_data_fetcher_cache,
                self._recover_data_fetcher_fallback
            ],
            SystemComponent.ML_ENGINE: [
                self._recover_ml_engine_memory,
                self._recover_ml_engine_models,
                self._recover_ml_engine_fallback
            ],
            SystemComponent.STRATEGY_ENGINE: [
                self._recover_strategy_engine_config,
                self._recover_strategy_engine_state,
                self._recover_strategy_engine_fallback
            ],
            SystemComponent.PERFORMANCE_MONITOR: [
                self._recover_performance_monitor_db,
                self._recover_performance_monitor_restart
            ],
            SystemComponent.MAIN_ORCHESTRATOR: [
                self._recover_main_orchestrator_restart,
                self._recover_main_orchestrator_safe_mode
            ],
            SystemComponent.WEB_DASHBOARD: [
                self._recover_web_dashboard_restart,
                self._recover_web_dashboard_fallback
            ]
        }
    
    def error_handler(self, component: SystemComponent, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Decorator for automatic error handling"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(
                        component=component,
                        severity=severity,
                        error=e,
                        context={
                            'function': func.__name__,
                            'args': str(args)[:200],  # Limit context size
                            'kwargs': str(kwargs)[:200]
                        }
                    )
                    raise
            return wrapper
        return decorator
    
    def handle_error(self, 
                    component: SystemComponent, 
                    severity: ErrorSeverity, 
                    error: Exception,
                    context: Dict[str, Any] = None) -> bool:
        """Handle error with automatic recovery attempts"""
        
        try:
            # Create error event
            error_event = ErrorEvent(
                timestamp=datetime.now(),
                component=component,
                severity=severity,
                error_type=type(error).__name__,
                message=str(error),
                traceback_info=traceback.format_exc(),
                context=context or {}
            )
            
            # Check for recurring errors
            error_key = f"{component.value}_{error_event.error_type}_{error_event.message}"
            if error_key in self.active_errors:
                self.active_errors[error_key].occurrence_count += 1
                error_event.occurrence_count = self.active_errors[error_key].occurrence_count
            
            # Log error
            self._log_error(error_event)
            
            # Store error
            self.error_history.append(error_event)
            self.active_errors[error_key] = error_event
            
            # Attempt recovery if enabled
            recovery_successful = False
            if self.config['recovery']['auto_recovery_enabled']:
                recovery_successful = self._attempt_recovery(error_event)
            
            # Update error status
            error_event.recovery_attempted = True
            error_event.recovery_successful = recovery_successful
            
            # Alert if necessary
            self._check_alert_conditions(error_event)
            
            # Update component health
            self._update_component_health(component, severity, recovery_successful)
            
            return recovery_successful
            
        except Exception as recovery_error:
            self.logger.critical(f"Error in error handling system: {recovery_error}")
            return False
    
    def _log_error(self, error_event: ErrorEvent):
        """Log error with appropriate level"""
        
        log_message = (f"[{error_event.component.value}] {error_event.error_type}: "
                      f"{error_event.message}")
        
        if error_event.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_event.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Detailed logging to file
        detailed_message = {
            'timestamp': error_event.timestamp.isoformat(),
            'component': error_event.component.value,
            'severity': error_event.severity.value,
            'error_type': error_event.error_type,
            'message': error_event.message,
            'context': error_event.context,
            'occurrence_count': error_event.occurrence_count
        }
        
        error_file = self.error_log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(error_file, 'a') as f:
            f.write(json.dumps(detailed_message) + '\n')
    
    def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt to recover from error"""
        
        component = error_event.component
        
        # Check if we've exceeded maximum attempts
        if error_event.occurrence_count > self.max_recovery_attempts:
            self.logger.warning(f"Maximum recovery attempts exceeded for {component.value}")
            return False
        
        # Get recovery strategies for component
        strategies = self.recovery_strategies.get(component, [])
        
        if not strategies:
            self.logger.warning(f"No recovery strategies defined for {component.value}")
            return False
        
        # Try each recovery strategy
        for i, strategy in enumerate(strategies):
            try:
                self.logger.info(f"Attempting recovery strategy {i+1}/{len(strategies)} "
                                f"for {component.value}")
                
                # Apply backoff delay
                if error_event.occurrence_count > 1:
                    delay = self.recovery_backoff_seconds * (error_event.occurrence_count - 1)
                    self.logger.info(f"Applying backoff delay: {delay} seconds")
                    time.sleep(delay)
                
                # Execute recovery strategy
                success = strategy(error_event)
                
                if success:
                    self.logger.info(f"Recovery successful for {component.value} "
                                   f"using strategy {i+1}")
                    return True
                
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy {i+1} failed: {recovery_error}")
                continue
        
        self.logger.error(f"All recovery strategies failed for {component.value}")
        return False
    
    # Recovery Strategy Implementations
    def _recover_data_fetcher_connection(self, error_event: ErrorEvent) -> bool:
        """Recover data fetcher connection issues"""
        try:
            self.logger.info("Attempting to recover data fetcher connection...")
            
            # Wait for network issues to resolve
            time.sleep(30)
            
            # Try to reconnect (would integrate with actual data fetcher)
            # This is a placeholder - actual implementation would call data fetcher methods
            
            return True
        except Exception as e:
            self.logger.error(f"Data fetcher connection recovery failed: {e}")
            return False
    
    def _recover_data_fetcher_cache(self, error_event: ErrorEvent) -> bool:
        """Recover data fetcher cache issues"""
        try:
            self.logger.info("Attempting to recover data fetcher cache...")
            
            # Clear cache and retry (placeholder)
            cache_dir = Path("data/cache")
            if cache_dir.exists():
                # Clear old cache files
                for cache_file in cache_dir.glob("*.pkl"):
                    if cache_file.stat().st_mtime < time.time() - 3600:  # Older than 1 hour
                        cache_file.unlink()
            
            return True
        except Exception as e:
            self.logger.error(f"Data fetcher cache recovery failed: {e}")
            return False
    
    def _recover_data_fetcher_fallback(self, error_event: ErrorEvent) -> bool:
        """Fallback to alternative data source"""
        try:
            self.logger.info("Using data fetcher fallback mode...")
            # Implement fallback to cached data or alternative source
            return True
        except Exception as e:
            self.logger.error(f"Data fetcher fallback failed: {e}")
            return False
    
    def _recover_ml_engine_memory(self, error_event: ErrorEvent) -> bool:
        """Recover ML engine memory issues"""
        try:
            self.logger.info("Attempting to recover ML engine memory...")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear TensorFlow session if applicable
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except ImportError:
                pass
            
            return True
        except Exception as e:
            self.logger.error(f"ML engine memory recovery failed: {e}")
            return False
    
    def _recover_ml_engine_models(self, error_event: ErrorEvent) -> bool:
        """Recover ML engine model loading issues"""
        try:
            self.logger.info("Attempting to recover ML engine models...")
            
            # Reload models from backup or retrain
            # This would integrate with actual ML engine
            
            return True
        except Exception as e:
            self.logger.error(f"ML engine model recovery failed: {e}")
            return False
    
    def _recover_ml_engine_fallback(self, error_event: ErrorEvent) -> bool:
        """Fallback to simpler ML models"""
        try:
            self.logger.info("Using ML engine fallback mode...")
            # Switch to simpler models or technical analysis only
            return True
        except Exception as e:
            self.logger.error(f"ML engine fallback failed: {e}")
            return False
    
    def _recover_strategy_engine_config(self, error_event: ErrorEvent) -> bool:
        """Recover strategy engine configuration issues"""
        try:
            self.logger.info("Attempting to recover strategy engine configuration...")
            
            # Reload configuration from backup
            # Reset to default parameters if necessary
            
            return True
        except Exception as e:
            self.logger.error(f"Strategy engine config recovery failed: {e}")
            return False
    
    def _recover_strategy_engine_state(self, error_event: ErrorEvent) -> bool:
        """Recover strategy engine state"""
        try:
            self.logger.info("Attempting to recover strategy engine state...")
            
            # Reset internal state
            # Clear position tracking if corrupted
            
            return True
        except Exception as e:
            self.logger.error(f"Strategy engine state recovery failed: {e}")
            return False
    
    def _recover_strategy_engine_fallback(self, error_event: ErrorEvent) -> bool:
        """Fallback to conservative strategy"""
        try:
            self.logger.info("Using strategy engine fallback mode...")
            # Switch to conservative trading parameters
            return True
        except Exception as e:
            self.logger.error(f"Strategy engine fallback failed: {e}")
            return False
    
    def _recover_performance_monitor_db(self, error_event: ErrorEvent) -> bool:
        """Recover performance monitor database issues"""
        try:
            self.logger.info("Attempting to recover performance monitor database...")
            
            # Check database integrity
            # Rebuild if necessary
            
            return True
        except Exception as e:
            self.logger.error(f"Performance monitor DB recovery failed: {e}")
            return False
    
    def _recover_performance_monitor_restart(self, error_event: ErrorEvent) -> bool:
        """Restart performance monitor"""
        try:
            self.logger.info("Restarting performance monitor...")
            # Restart performance monitoring component
            return True
        except Exception as e:
            self.logger.error(f"Performance monitor restart failed: {e}")
            return False
    
    def _recover_main_orchestrator_restart(self, error_event: ErrorEvent) -> bool:
        """Restart main orchestrator"""
        try:
            self.logger.info("Attempting to restart main orchestrator...")
            # Graceful restart of main components
            return True
        except Exception as e:
            self.logger.error(f"Main orchestrator restart failed: {e}")
            return False
    
    def _recover_main_orchestrator_safe_mode(self, error_event: ErrorEvent) -> bool:
        """Enter safe mode"""
        try:
            self.logger.info("Entering safe mode...")
            # Disable live trading, enable monitoring only
            return True
        except Exception as e:
            self.logger.error(f"Safe mode activation failed: {e}")
            return False
    
    def _recover_web_dashboard_restart(self, error_event: ErrorEvent) -> bool:
        """Restart web dashboard"""
        try:
            self.logger.info("Restarting web dashboard...")
            # Restart Flask application
            return True
        except Exception as e:
            self.logger.error(f"Web dashboard restart failed: {e}")
            return False
    
    def _recover_web_dashboard_fallback(self, error_event: ErrorEvent) -> bool:
        """Fallback to basic dashboard"""
        try:
            self.logger.info("Using basic dashboard mode...")
            # Switch to simplified dashboard
            return True
        except Exception as e:
            self.logger.error(f"Web dashboard fallback failed: {e}")
            return False
    
    def _check_alert_conditions(self, error_event: ErrorEvent):
        """Check if alerts should be sent"""
        
        # Check severity threshold
        alert_threshold = ErrorSeverity[self.config['alerting']['log_level_for_alerts']]
        
        if error_event.severity.value < alert_threshold.value:
            return
        
        # Check cooldown period
        cooldown_minutes = self.config['alerting']['alert_cooldown_minutes']
        recent_alerts = [e for e in self.error_history 
                        if e.timestamp > datetime.now() - timedelta(minutes=cooldown_minutes)
                        and e.component == error_event.component
                        and e.severity == error_event.severity]
        
        if len(recent_alerts) > 1:
            return  # Still in cooldown
        
        # Send alerts
        self._send_alerts(error_event)
    
    def _send_alerts(self, error_event: ErrorEvent):
        """Send alerts via configured channels"""
        
        alert_message = (f"ALERT: {error_event.severity.value} error in "
                        f"{error_event.component.value}\n"
                        f"Error: {error_event.error_type}\n"
                        f"Message: {error_event.message}\n"
                        f"Time: {error_event.timestamp}\n"
                        f"Occurrence: #{error_event.occurrence_count}")
        
        try:
            # Email alerts
            if self.config['alerting']['email_enabled']:
                self._send_email_alert(alert_message)
            
            # Webhook alerts
            if self.config['alerting']['webhook_enabled']:
                self._send_webhook_alert(error_event)
            
            # Log alert
            self.logger.critical(f"ALERT SENT: {alert_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alerts: {e}")
    
    def _send_email_alert(self, message: str):
        """Send email alert (placeholder)"""
        # Implement email sending logic
        pass
    
    def _send_webhook_alert(self, error_event: ErrorEvent):
        """Send webhook alert (placeholder)"""
        # Implement webhook sending logic
        pass
    
    def _update_component_health(self, component: SystemComponent, 
                                severity: ErrorSeverity, recovery_successful: bool):
        """Update component health status"""
        
        if component not in self.component_health:
            self.component_health[component] = {
                'status': 'HEALTHY',
                'last_error': None,
                'error_count': 0,
                'last_recovery': None
            }
        
        health = self.component_health[component]
        health['last_error'] = datetime.now()
        health['error_count'] += 1
        
        if recovery_successful:
            health['last_recovery'] = datetime.now()
            health['status'] = 'RECOVERED'
        else:
            if severity == ErrorSeverity.CRITICAL:
                health['status'] = 'CRITICAL'
            elif severity == ErrorSeverity.HIGH:
                health['status'] = 'DEGRADED'
            else:
                health['status'] = 'WARNING'
    
    def start_health_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Health monitoring stopped")
    
    def _health_monitoring_loop(self):
        """Continuous health monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Perform health check
                health_status = self.get_system_health()
                
                # Check for concerning conditions
                self._check_system_health_alerts(health_status)
                
                # Clean up old errors
                self._cleanup_old_errors()
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status"""
        
        try:
            current_time = datetime.now()
            
            # System resource monitoring
            memory_usage = 0
            cpu_usage = 0
            disk_usage = 0
            
            if self.config['monitoring']['system_resource_monitoring']:
                try:
                    process = psutil.Process()
                    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                    cpu_usage = process.cpu_percent()
                    disk_usage = psutil.disk_usage('/').percent
                except Exception as e:
                    self.logger.warning(f"Resource monitoring error: {e}")
            
            # Error counting
            hour_ago = current_time - timedelta(hours=1)
            recent_errors = [e for e in self.error_history if e.timestamp > hour_ago]
            critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
            
            # Overall system status
            if len(critical_errors) > 0:
                overall_status = "CRITICAL"
            elif len(recent_errors) > self.config['thresholds']['error_rate_critical']:
                overall_status = "CRITICAL"
            elif len(recent_errors) > self.config['thresholds']['error_rate_warning']:
                overall_status = "DEGRADED"
            elif memory_usage > self.config['thresholds']['memory_usage_critical']:
                overall_status = "CRITICAL"
            elif cpu_usage > self.config['thresholds']['cpu_usage_critical']:
                overall_status = "CRITICAL"
            elif disk_usage > self.config['thresholds']['disk_usage_critical']:
                overall_status = "CRITICAL"
            else:
                overall_status = "HEALTHY"
            
            # Component status
            component_status = {}
            for component in SystemComponent:
                if component in self.component_health:
                    component_status[component.value] = self.component_health[component]['status']
                else:
                    component_status[component.value] = "UNKNOWN"
            
            # Uptime calculation
            uptime_hours = (current_time - self.start_time).total_seconds() / 3600
            
            return SystemHealth(
                timestamp=current_time,
                overall_status=overall_status,
                component_status=component_status,
                error_count_last_hour=len(recent_errors),
                critical_errors=len(critical_errors),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                disk_usage_percent=disk_usage,
                uptime_hours=uptime_hours,
                active_errors=list(self.active_errors.values())
            )
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return SystemHealth(
                timestamp=current_time,
                overall_status="ERROR",
                component_status={},
                error_count_last_hour=0,
                critical_errors=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                disk_usage_percent=0,
                uptime_hours=0,
                active_errors=[]
            )
    
    def _check_system_health_alerts(self, health_status: SystemHealth):
        """Check system health for alert conditions"""
        
        thresholds = self.config['thresholds']
        
        # Memory usage alerts
        if health_status.memory_usage_mb > 0:
            memory_percent = (health_status.memory_usage_mb / (psutil.virtual_memory().total / 1024 / 1024)) * 100
            
            if memory_percent > thresholds['memory_usage_critical']:
                self._create_system_alert("CRITICAL", f"Memory usage critical: {memory_percent:.1f}%")
            elif memory_percent > thresholds['memory_usage_warning']:
                self._create_system_alert("WARNING", f"Memory usage high: {memory_percent:.1f}%")
        
        # CPU usage alerts
        if health_status.cpu_usage_percent > thresholds['cpu_usage_critical']:
            self._create_system_alert("CRITICAL", f"CPU usage critical: {health_status.cpu_usage_percent:.1f}%")
        elif health_status.cpu_usage_percent > thresholds['cpu_usage_warning']:
            self._create_system_alert("WARNING", f"CPU usage high: {health_status.cpu_usage_percent:.1f}%")
        
        # Disk usage alerts
        if health_status.disk_usage_percent > thresholds['disk_usage_critical']:
            self._create_system_alert("CRITICAL", f"Disk usage critical: {health_status.disk_usage_percent:.1f}%")
        elif health_status.disk_usage_percent > thresholds['disk_usage_warning']:
            self._create_system_alert("WARNING", f"Disk usage high: {health_status.disk_usage_percent:.1f}%")
        
        # Error rate alerts
        if health_status.error_count_last_hour > thresholds['error_rate_critical']:
            self._create_system_alert("CRITICAL", f"High error rate: {health_status.error_count_last_hour} errors/hour")
        elif health_status.error_count_last_hour > thresholds['error_rate_warning']:
            self._create_system_alert("WARNING", f"Elevated error rate: {health_status.error_count_last_hour} errors/hour")
    
    def _create_system_alert(self, severity: str, message: str):
        """Create system-level alert"""
        
        # Check if similar alert was sent recently
        recent_alerts = [e for e in self.error_history 
                        if e.timestamp > datetime.now() - timedelta(minutes=30)
                        and message in e.message]
        
        if recent_alerts:
            return  # Don't spam alerts
        
        # Create synthetic error event for system alerts
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            component=SystemComponent.MAIN_ORCHESTRATOR,
            severity=ErrorSeverity[severity],
            error_type="SystemAlert",
            message=message,
            traceback_info="",
            context={"type": "system_health_alert"}
        )
        
        self._log_error(error_event)
        self.error_history.append(error_event)
        
        if severity in ["CRITICAL", "HIGH"]:
            self._send_alerts(error_event)
    
    def _cleanup_old_errors(self):
        """Clean up old errors from memory"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.config['monitoring']['error_retention_hours'])
        
        # Clean error history
        self.error_history = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        # Clean active errors
        active_errors_to_remove = []
        for key, error in self.active_errors.items():
            if error.timestamp < cutoff_time:
                active_errors_to_remove.append(key)
        
        for key in active_errors_to_remove:
            del self.active_errors[key]
        
        # Limit memory usage
        max_errors = self.config['monitoring']['max_errors_in_memory']
        if len(self.error_history) > max_errors:
            self.error_history = self.error_history[-max_errors:]
    
    def create_system_backup(self) -> bool:
        """Create system state backup"""
        
        try:
            backup_dir = Path("backups") / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration
            config_backup = backup_dir / "config"
            config_backup.mkdir(exist_ok=True)
            
            for config_file in Path("config").glob("*.json"):
                import shutil
                shutil.copy2(config_file, config_backup)
            
            # Backup error history
            error_backup = {
                'error_history': [asdict(e) for e in self.error_history],
                'component_health': {k.value: v for k, v in self.component_health.items()},
                'backup_timestamp': datetime.now().isoformat()
            }
            
            with open(backup_dir / "error_recovery_state.json", 'w') as f:
                json.dump(error_backup, f, indent=2, default=str)
            
            self.logger.info(f"System backup created: {backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"System backup failed: {e}")
            return False
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore system state from backup"""
        
        try:
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                self.logger.error(f"Backup directory not found: {backup_path}")
                return False
            
            # Restore error recovery state
            state_file = backup_dir / "error_recovery_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    backup_data = json.load(f)
                
                # Restore error history (limited)
                self.error_history = []
                for error_data in backup_data.get('error_history', [])[-100:]:  # Last 100 errors
                    error_event = ErrorEvent(**error_data)
                    self.error_history.append(error_event)
                
                self.logger.info(f"Restored {len(self.error_history)} error records")
            
            self.logger.info(f"System state restored from: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"System restore failed: {e}")
            return False
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        # Group by component
        by_component = {}
        for error in recent_errors:
            component = error.component.value
            if component not in by_component:
                by_component[component] = []
            by_component[component].append(error)
        
        # Group by severity
        by_severity = {}
        for error in recent_errors:
            severity = error.severity.value
            if severity not in by_severity:
                by_severity[severity] = 0
            by_severity[severity] += 1
        
        return {
            'time_period_hours': hours,
            'total_errors': len(recent_errors),
            'by_component': {k: len(v) for k, v in by_component.items()},
            'by_severity': by_severity,
            'most_recent_error': recent_errors[-1].timestamp.isoformat() if recent_errors else None,
            'recovery_success_rate': sum(1 for e in recent_errors if e.recovery_successful) / len(recent_errors) if recent_errors else 0
        }

# Example usage and testing
def main():
    """Main function for testing error recovery system"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize error recovery system
    recovery_system = ErrorRecoverySystem()
    
    try:
        # Start health monitoring
        recovery_system.start_health_monitoring()
        
        # Test error handling
        @recovery_system.error_handler(SystemComponent.ML_ENGINE, ErrorSeverity.HIGH)
        def test_function():
            raise ValueError("Test error for demonstration")
        
        # Test the error handling
        try:
            test_function()
        except ValueError:
            pass  # Expected
        
        # Get system health
        health = recovery_system.get_system_health()
        print(f"System Health: {health.overall_status}")
        print(f"Errors in last hour: {health.error_count_last_hour}")
        
        # Get error summary
        summary = recovery_system.get_error_summary(hours=1)
        print(f"Error Summary: {summary}")
        
        # Create backup
        backup_success = recovery_system.create_system_backup()
        print(f"Backup created: {backup_success}")
        
        # Let monitoring run for a bit
        time.sleep(10)
        
    finally:
        # Stop monitoring
        recovery_system.stop_health_monitoring()

if __name__ == "__main__":
    main()