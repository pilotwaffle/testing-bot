# config_and_monitoring.py
"""
Configuration Management and Monitoring Dashboard
Elite Trading Bot V3.0 - Error Handling Configuration
"""

import os
import yaml
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging

# Configuration models
@dataclass
class ExchangeConfig:
    """Configuration for individual exchange"""
    name: str
    api_key: str
    secret: str
    base_url: str
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    priority: int = 1  # 1 = primary, 2 = secondary, etc.
    enabled: bool = True

@dataclass
class ErrorHandlingConfig:
    """Global error handling configuration"""
    log_level: str = "INFO"
    max_error_history: int = 1000
    health_check_interval: int = 60
    enable_fallback: bool = True
    enable_circuit_breaker: bool = True
    enable_rate_limiting: bool = True
    notification_webhooks: List[str] = None
    
    def __post_init__(self):
        if self.notification_webhooks is None:
            self.notification_webhooks = []

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    enable_discord_alerts: bool = False
    email_recipients: List[str] = None
    slack_webhook: str = ""
    discord_webhook: str = ""
    alert_on_error_types: List[str] = None
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []
        if self.alert_on_error_types is None:
            self.alert_on_error_types = ["CRITICAL", "HIGH"]

class TradingBotConfig:
    """Main configuration manager for the trading bot"""
    
    def __init__(self, config_file: str = "bot_config.yaml"):
        self.config_file = config_file
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self.error_handling: ErrorHandlingConfig = ErrorHandlingConfig()
        self.monitoring: MonitoringConfig = MonitoringConfig()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Load exchanges
                for name, exchange_data in config_data.get('exchanges', {}).items():
                    self.exchanges[name] = ExchangeConfig(name=name, **exchange_data)
                
                # Load error handling config
                if 'error_handling' in config_data:
                    self.error_handling = ErrorHandlingConfig(**config_data['error_handling'])
                
                # Load monitoring config
                if 'monitoring' in config_data:
                    self.monitoring = MonitoringConfig(**config_data['monitoring'])
            
            else:
                # Create default config
                self.create_default_config()
        
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration"""
        self.exchanges = {
            "kraken": ExchangeConfig(
                name="kraken",
                api_key=os.getenv("KRAKEN_API_KEY", ""),
                secret=os.getenv("KRAKEN_SECRET", ""),
                base_url="https://api.kraken.com",
                rate_limit_per_minute=60,
                priority=1
            ),
            "binance": ExchangeConfig(
                name="binance",
                api_key=os.getenv("BINANCE_API_KEY", ""),
                secret=os.getenv("BINANCE_SECRET", ""),
                base_url="https://api.binance.com",
                rate_limit_per_minute=1200,
                priority=2,
                enabled=False
            ),
            "coinbase": ExchangeConfig(
                name="coinbase",
                api_key=os.getenv("COINBASE_API_KEY", ""),
                secret=os.getenv("COINBASE_SECRET", ""),
                base_url="https://api.pro.coinbase.com",
                rate_limit_per_minute=100,
                priority=3,
                enabled=False
            )
        }
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            config_data = {
                'exchanges': {name: asdict(config) for name, config in self.exchanges.items()},
                'error_handling': asdict(self.error_handling),
                'monitoring': asdict(self.monitoring)
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def get_primary_exchange(self) -> Optional[ExchangeConfig]:
        """Get the primary (highest priority) enabled exchange"""
        enabled_exchanges = [ex for ex in self.exchanges.values() if ex.enabled]
        if enabled_exchanges:
            return min(enabled_exchanges, key=lambda x: x.priority)
        return None
    
    def get_fallback_exchanges(self) -> List[ExchangeConfig]:
        """Get fallback exchanges sorted by priority"""
        enabled_exchanges = [ex for ex in self.exchanges.values() if ex.enabled]
        return sorted(enabled_exchanges, key=lambda x: x.priority)[1:]  # Skip primary

# Monitoring dashboard HTML template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Elite Trading Bot V3.0 - Error Monitoring Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            color: #e0e0e0; 
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { 
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            padding: 20px; margin-bottom: 30px; border-radius: 10px;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
        }
        .header h1 { font-size: 28px; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 16px; }
        
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { 
            background: rgba(45, 45, 68, 0.8); 
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 10px; padding: 20px; 
            backdrop-filter: blur(10px);
        }
        .card h3 { color: #6366f1; margin-bottom: 15px; font-size: 18px; }
        
        .status-indicator { 
            display: inline-block; width: 12px; height: 12px; 
            border-radius: 50%; margin-right: 8px; 
        }
        .status-healthy { background: #10b981; }
        .status-degraded { background: #f59e0b; }
        .status-unhealthy { background: #ef4444; }
        
        .metric { 
            display: flex; justify-content: space-between; 
            margin-bottom: 10px; padding: 8px 0; 
            border-bottom: 1px solid rgba(99, 102, 241, 0.1);
        }
        .metric:last-child { border-bottom: none; }
        .metric-value { font-weight: bold; color: #6366f1; }
        
        .error-list { max-height: 300px; overflow-y: auto; }
        .error-item { 
            background: rgba(239, 68, 68, 0.1); 
            border-left: 3px solid #ef4444; 
            padding: 10px; margin-bottom: 10px; 
            border-radius: 5px; font-size: 14px; 
        }
        .error-critical { border-left-color: #dc2626; background: rgba(220, 38, 38, 0.15); }
        .error-high { border-left-color: #ea580c; background: rgba(234, 88, 12, 0.15); }
        .error-medium { border-left-color: #d97706; background: rgba(217, 119, 6, 0.15); }
        .error-low { border-left-color: #65a30d; background: rgba(101, 163, 13, 0.15); }
        
        .circuit-breaker { 
            padding: 5px 10px; border-radius: 15px; 
            font-size: 12px; font-weight: bold; 
        }
        .cb-closed { background: #10b981; color: white; }
        .cb-open { background: #ef4444; color: white; }
        .cb-half-open { background: #f59e0b; color: white; }
        
        .refresh-btn { 
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white; border: none; padding: 10px 20px; 
            border-radius: 5px; cursor: pointer; margin: 10px 0;
        }
        .refresh-btn:hover { opacity: 0.9; }
        
        .timestamp { color: #9ca3af; font-size: 12px; }
        
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .loading { animation: pulse 2s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Elite Trading Bot V3.0 - Error Monitoring Dashboard</h1>
            <p>Real-time monitoring of exchange APIs, error rates, and system health</p>
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
        </div>
        
        <div class="dashboard-grid">
            <!-- System Health Card -->
            <div class="card">
                <h3>🏥 System Health</h3>
                <div id="system-health" class="loading">Loading system health...</div>
            </div>
            
            <!-- Exchange Status Card -->
            <div class="card">
                <h3>🏦 Exchange Status</h3>
                <div id="exchange-status" class="loading">Loading exchange status...</div>
            </div>
            
            <!-- Recent Errors Card -->
            <div class="card">
                <h3>⚠️ Recent Errors (Last Hour)</h3>
                <div id="recent-errors" class="loading">Loading recent errors...</div>
            </div>
            
            <!-- Circuit Breakers Card -->
            <div class="card">
                <h3>🔌 Circuit Breakers</h3>
                <div id="circuit-breakers" class="loading">Loading circuit breaker status...</div>
            </div>
            
            <!-- Error Statistics Card -->
            <div class="card">
                <h3>📊 Error Statistics</h3>
                <div id="error-stats" class="loading">Loading error statistics...</div>
            </div>
            
            <!-- Performance Metrics Card -->
            <div class="card">
                <h3>⚡ Performance Metrics</h3>
                <div id="performance-metrics" class="loading">Loading performance metrics...</div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8000/ws/notifications');
            
            ws.onopen = function(event) {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'health_update') {
                    updateHealthDisplay(data.data);
                } else if (data.type === 'error_notification') {
                    showErrorNotification(data);
                }
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket disconnected');
                setTimeout(connectWebSocket, 5000); // Reconnect after 5 seconds
            };
        }
        
        async function refreshData() {
            try {
                // Fetch system health
                const healthResponse = await fetch('/api/health');
                const healthData = await healthResponse.json();
                updateHealthDisplay(healthData);
                
                // Fetch recent errors
                const errorsResponse = await fetch('/api/errors/recent');
                const errorsData = await errorsResponse.json();
                updateErrorsDisplay(errorsData);
                
            } catch (error) {
                console.error('Failed to refresh data:', error);
            }
        }
        
        function updateHealthDisplay(healthData) {
            // Update system health
            const systemHealthDiv = document.getElementById('system-health');
            const statusClass = getStatusClass(healthData.overall_status);
            systemHealthDiv.innerHTML = `
                <div class="metric">
                    <span>Overall Status</span>
                    <span><span class="status-indicator ${statusClass}"></span>${healthData.overall_status}</span>
                </div>
                <div class="metric">
                    <span>Total Errors (1h)</span>
                    <span class="metric-value">${healthData.total_errors_1h}</span>
                </div>
                <div class="timestamp">Last updated: ${new Date(healthData.timestamp).toLocaleString()}</div>
            `;
            
            // Update exchange status
            const exchangeStatusDiv = document.getElementById('exchange-status');
            let exchangeHtml = '';
            for (const [name, status] of Object.entries(healthData.exchanges)) {
                const statusClass = getStatusClass(status.status);
                exchangeHtml += `
                    <div class="metric">
                        <span>${name.toUpperCase()}</span>
                        <span><span class="status-indicator ${statusClass}"></span>${status.status}</span>
                    </div>
                `;
            }
            exchangeStatusDiv.innerHTML = exchangeHtml;
            
            // Update circuit breakers
            const circuitBreakersDiv = document.getElementById('circuit-breakers');
            let cbHtml = '';
            for (const [name, status] of Object.entries(healthData.exchanges)) {
                for (const [endpoint, state] of Object.entries(status.circuit_breakers)) {
                    cbHtml += `
                        <div class="metric">
                            <span>${endpoint}</span>
                            <span class="circuit-breaker cb-${state}">${state}</span>
                        </div>
                    `;
                }
            }
            circuitBreakersDiv.innerHTML = cbHtml || '<p>No circuit breakers active</p>';
        }
        
        function updateErrorsDisplay(errorsData) {
            const errorsDiv = document.getElementById('recent-errors');
            let errorsHtml = '';
            
            if (errorsData.errors.length === 0) {
                errorsHtml = '<p style="color: #10b981;">✅ No errors in the last hour!</p>';
            } else {
                errorsData.errors.slice(0, 10).forEach(error => {
                    errorsHtml += `
                        <div class="error-item error-${error.severity}">
                            <strong>${error.exchange} - ${error.endpoint}</strong><br>
                            ${error.message}<br>
                            <span class="timestamp">${new Date(error.timestamp).toLocaleString()} (Retry: ${error.retry_count})</span>
                        </div>
                    `;
                });
            }
            errorsDiv.innerHTML = errorsHtml;
            
            // Update error statistics
            const errorStats = {};
            errorsData.errors.forEach(error => {
                errorStats[error.severity] = (errorStats[error.severity] || 0) + 1;
            });
            
            const statsDiv = document.getElementById('error-stats');
            let statsHtml = '';
            Object.entries(errorStats).forEach(([severity, count]) => {
                statsHtml += `
                    <div class="metric">
                        <span>${severity.toUpperCase()}</span>
                        <span class="metric-value">${count}</span>
                    </div>
                `;
            });
            statsDiv.innerHTML = statsHtml || '<p>No error statistics available</p>';
        }
        
        function getStatusClass(status) {
            switch(status) {
                case 'healthy': return 'status-healthy';
                case 'degraded': return 'status-degraded';
                case 'unhealthy': return 'status-unhealthy';
                default: return 'status-degraded';
            }
        }
        
        function showErrorNotification(data) {
            // You could implement toast notifications here
            console.log('Error notification:', data);
        }
        
        // Initialize
        connectWebSocket();
        refreshData();
        
        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
    </script>
</body>
</html>
"""

# FastAPI route for monitoring dashboard
def add_monitoring_dashboard(app: FastAPI):
    """Add monitoring dashboard route to FastAPI app"""
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def monitoring_dashboard(request: Request):
        """Serve the monitoring dashboard"""
        return HTMLResponse(content=DASHBOARD_HTML)
    
    @app.get("/api/config")
    async def get_config():
        """Get current bot configuration"""
        config = TradingBotConfig()
        return {
            "exchanges": {name: asdict(ex) for name, ex in config.exchanges.items()},
            "error_handling": asdict(config.error_handling),
            "monitoring": asdict(config.monitoring)
        }
    
    @app.post("/api/config/exchange/{exchange_name}")
    async def update_exchange_config(exchange_name: str, config_data: dict):
        """Update exchange configuration"""
        try:
            config = TradingBotConfig()
            if exchange_name in config.exchanges:
                # Update existing exchange
                for key, value in config_data.items():
                    if hasattr(config.exchanges[exchange_name], key):
                        setattr(config.exchanges[exchange_name], key, value)
            else:
                # Create new exchange
                config.exchanges[exchange_name] = ExchangeConfig(name=exchange_name, **config_data)
            
            config.save_config()
            return {"message": f"Exchange {exchange_name} configuration updated"}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Alerting system
class AlertManager:
    """Manage alerts and notifications for errors"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
    
    async def send_alert(self, error_type: str, message: str, severity: str):
        """Send alert through configured channels"""
        if severity.upper() not in self.config.alert_on_error_types:
            return
        
        alert_message = f"🚨 Elite Trading Bot Alert\n\nSeverity: {severity}\nType: {error_type}\nMessage: {message}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send to Slack
        if self.config.enable_slack_alerts and self.config.slack_webhook:
            await self._send_slack_alert(alert_message)
        
        # Send to Discord
        if self.config.enable_discord_alerts and self.config.discord_webhook:
            await self._send_discord_alert(alert_message)
        
        # Send email (would need email service integration)
        if self.config.enable_email_alerts and self.config.email_recipients:
            await self._send_email_alert(alert_message)
    
    async def _send_slack_alert(self, message: str):
        """Send alert to Slack webhook"""
        try:
            import aiohttp
            payload = {"text": message}
            async with aiohttp.ClientSession() as session:
                await session.post(self.config.slack_webhook, json=payload)
        except Exception as e:
            logging.error(f"Failed to send Slack alert: {e}")
    
    async def _send_discord_alert(self, message: str):
        """Send alert to Discord webhook"""
        try:
            import aiohttp
            payload = {"content": message}
            async with aiohttp.ClientSession() as session:
                await session.post(self.config.discord_webhook, json=payload)
        except Exception as e:
            logging.error(f"Failed to send Discord alert: {e}")
    
    async def _send_email_alert(self, message: str):
        """Send email alert (implement with your email service)"""
        # Implement with your preferred email service (SendGrid, AWS SES, etc.)
        logging.info(f"Email alert would be sent: {message}")

# Example configuration file content
EXAMPLE_CONFIG_YAML = """
# Elite Trading Bot V3.0 Configuration
exchanges:
  kraken:
    api_key: "your_kraken_api_key"
    secret: "your_kraken_secret"
    base_url: "https://api.kraken.com"
    rate_limit_per_minute: 60
    timeout_seconds: 30
    max_retries: 3
    circuit_breaker_threshold: 5
    circuit_breaker_timeout: 60
    priority: 1
    enabled: true
  
  binance:
    api_key: "your_binance_api_key"
    secret: "your_binance_secret"
    base_url: "https://api.binance.com"
    rate_limit_per_minute: 1200
    timeout_seconds: 30
    max_retries: 3
    circuit_breaker_threshold: 5
    circuit_breaker_timeout: 60
    priority: 2
    enabled: false
  
  coinbase:
    api_key: "your_coinbase_api_key"
    secret: "your_coinbase_secret"
    base_url: "https://api.pro.coinbase.com"
    rate_limit_per_minute: 100
    timeout_seconds: 30
    max_retries: 3
    circuit_breaker_threshold: 5
    circuit_breaker_timeout: 60
    priority: 3
    enabled: false

error_handling:
  log_level: "INFO"
  max_error_history: 1000
  health_check_interval: 60
  enable_fallback: true
  enable_circuit_breaker: true
  enable_rate_limiting: true
  notification_webhooks: []

monitoring:
  enable_email_alerts: false
  enable_slack_alerts: true
  enable_discord_alerts: false
  email_recipients: []
  slack_webhook: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  discord_webhook: ""
  alert_on_error_types: ["CRITICAL", "HIGH"]
"""

# Save example config file
def create_example_config():
    """Create example configuration file"""
    with open("bot_config.example.yaml", "w") as f:
        f.write(EXAMPLE_CONFIG_YAML)
    print("Example configuration file created: bot_config.example.yaml")

if __name__ == "__main__":
    create_example_config()