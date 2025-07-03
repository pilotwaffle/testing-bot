# web_dashboard.py - Web Dashboard for Trading Bot Monitoring
"""
Web Dashboard for Enhanced Trading Bot
Real-time monitoring interface with performance metrics and system status
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import sqlite3
import json
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import os

class TradingBotDashboard:
    """Web dashboard for monitoring trading bot performance"""
    
    def __init__(self, config_path: str = "config/dashboard_config.json"):
        self.app = Flask(__name__)
        self.app.secret_key = 'your-secret-key-here'  # Change in production
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Database path
        self.db_path = self.config['data_sources']['performance_db_path']
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("Trading Bot Dashboard initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load dashboard configuration"""
        default_config = {
            "server": {
                "host": "localhost",
                "port": 8050,
                "debug": True
            },
            "display": {
                "refresh_interval_seconds": 30,
                "max_chart_points": 500,
                "default_timeframe": "1h"
            },
            "data_sources": {
                "performance_db_path": "data/performance.db",
                "log_files_path": "logs/",
                "model_path": "models/"
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                self._deep_update(default_config, loaded_config)
        except Exception as e:
            print(f"Error loading dashboard config: {e}. Using defaults.")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html', 
                                 refresh_interval=self.config['display']['refresh_interval_seconds'])
        
        @self.app.route('/api/status')
        def api_status():
            """Get current bot status"""
            try:
                status = self._get_bot_status()
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance')
        def api_performance():
            """Get performance metrics"""
            try:
                hours = request.args.get('hours', 24, type=int)
                performance = self._get_performance_data(hours)
                return jsonify(performance)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/trades')
        def api_trades():
            """Get recent trades"""
            try:
                limit = request.args.get('limit', 50, type=int)
                trades = self._get_recent_trades(limit)
                return jsonify(trades)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """Get recent alerts"""
            try:
                hours = request.args.get('hours', 24, type=int)
                alerts = self._get_recent_alerts(hours)
                return jsonify(alerts)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/charts/performance')
        def api_performance_chart():
            """Get performance chart data"""
            try:
                hours = request.args.get('hours', 24, type=int)
                chart_data = self._generate_performance_chart(hours)
                return jsonify(chart_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/charts/signals')
        def api_signals_chart():
            """Get signals chart data"""
            try:
                hours = request.args.get('hours', 24, type=int)
                chart_data = self._generate_signals_chart(hours)
                return jsonify(chart_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/model_status')
        def api_model_status():
            """Get ML model status"""
            try:
                model_status = self._get_model_status()
                return jsonify(model_status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/logs')
        def logs_page():
            """Logs viewing page"""
            return render_template('logs.html')
        
        @self.app.route('/api/logs')
        def api_logs():
            """Get recent log entries"""
            try:
                lines = request.args.get('lines', 100, type=int)
                log_entries = self._get_recent_logs(lines)
                return jsonify(log_entries)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _get_bot_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        try:
            # Check if database exists
            if not Path(self.db_path).exists():
                return {
                    'status': 'DATABASE_NOT_FOUND',
                    'message': 'Performance database not found. Bot may not be running.',
                    'timestamp': datetime.now().isoformat()
                }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get latest trade
                cursor.execute('''
                    SELECT COUNT(*) as total_trades,
                           MAX(timestamp) as last_trade
                    FROM trades
                ''')
                trade_stats = cursor.fetchone()
                
                # Get recent performance
                recent_time = (datetime.now() - timedelta(hours=1)).isoformat()
                cursor.execute('''
                    SELECT COUNT(*) as recent_trades
                    FROM trades 
                    WHERE timestamp > ?
                ''', (recent_time,))
                recent_trades = cursor.fetchone()[0]
                
                # Get alerts count
                cursor.execute('''
                    SELECT COUNT(*) as total_alerts,
                           COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts
                    FROM alerts
                    WHERE timestamp > ?
                ''', (recent_time,))
                alert_stats = cursor.fetchone()
                
                status = 'RUNNING' if recent_trades > 0 else 'IDLE'
                
                return {
                    'status': status,
                    'total_trades': trade_stats[0],
                    'last_trade_time': trade_stats[1],
                    'recent_trades': recent_trades,
                    'total_alerts': alert_stats[0],
                    'critical_alerts': alert_stats[1],
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            return {
                'status': 'ERROR',
                'message': f'Error getting bot status: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_performance_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for specified time period"""
        try:
            if not Path(self.db_path).exists():
                return {'error': 'Database not found'}
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get completed trades
                cursor.execute('''
                    SELECT symbol, prediction, actual_result, profit_loss, timestamp
                    FROM trades 
                    WHERE timestamp > ? AND actual_result IS NOT NULL
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))
                
                trades = cursor.fetchall()
                
                if not trades:
                    return {
                        'total_trades': 0,
                        'accuracy': 0,
                        'total_profit': 0,
                        'win_rate': 0,
                        'avg_profit_per_trade': 0,
                        'sharpe_ratio': 0,
                        'by_symbol': {}
                    }
                
                # Calculate metrics
                df = pd.DataFrame(trades, columns=['symbol', 'prediction', 'actual_result', 'profit_loss', 'timestamp'])
                
                # Overall metrics
                total_trades = len(df)
                accuracy = ((df['prediction'] > 0.5) == (df['actual_result'] > 0.5)).mean()
                total_profit = df['profit_loss'].sum()
                win_rate = (df['profit_loss'] > 0).mean()
                avg_profit = df['profit_loss'].mean()
                
                # Sharpe ratio (simplified)
                if df['profit_loss'].std() > 0:
                    sharpe_ratio = avg_profit / df['profit_loss'].std()
                else:
                    sharpe_ratio = 0
                
                # By symbol breakdown
                by_symbol = {}
                for symbol in df['symbol'].unique():
                    symbol_df = df[df['symbol'] == symbol]
                    by_symbol[symbol] = {
                        'trades': len(symbol_df),
                        'accuracy': ((symbol_df['prediction'] > 0.5) == (symbol_df['actual_result'] > 0.5)).mean(),
                        'profit': symbol_df['profit_loss'].sum(),
                        'win_rate': (symbol_df['profit_loss'] > 0).mean()
                    }
                
                return {
                    'total_trades': total_trades,
                    'accuracy': accuracy,
                    'total_profit': total_profit,
                    'win_rate': win_rate,
                    'avg_profit_per_trade': avg_profit,
                    'sharpe_ratio': sharpe_ratio,
                    'by_symbol': by_symbol,
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            return {'error': f'Error getting performance data: {str(e)}'}
    
    def _get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades"""
        try:
            if not Path(self.db_path).exists():
                return []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, symbol, timeframe, prediction, confidence, 
                           actual_result, profit_loss, trade_id, model_used
                    FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                trades = cursor.fetchall()
                
                columns = ['timestamp', 'symbol', 'timeframe', 'prediction', 'confidence',
                          'actual_result', 'profit_loss', 'trade_id', 'model_used']
                
                return [dict(zip(columns, trade)) for trade in trades]
        
        except Exception as e:
            self.logger.error(f"Error getting recent trades: {e}")
            return []
    
    def _get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        try:
            if not Path(self.db_path).exists():
                return []
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, alert_type, severity, message, details
                    FROM alerts 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))
                
                alerts = cursor.fetchall()
                
                columns = ['timestamp', 'alert_type', 'severity', 'message', 'details']
                
                return [dict(zip(columns, alert)) for alert in alerts]
        
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {e}")
            return []
    
    def _generate_performance_chart(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance chart data"""
        try:
            if not Path(self.db_path).exists():
                return {'error': 'Database not found'}
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, profit_loss, symbol
                    FROM trades 
                    WHERE timestamp > ? AND profit_loss IS NOT NULL
                    ORDER BY timestamp ASC
                ''', (cutoff_time,))
                
                trades = cursor.fetchall()
                
                if not trades:
                    return {'data': [], 'layout': {}}
                
                df = pd.DataFrame(trades, columns=['timestamp', 'profit_loss', 'symbol'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['cumulative_profit'] = df['profit_loss'].cumsum()
                
                # Create plotly chart
                fig = go.Figure()
                
                # Cumulative profit line
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['cumulative_profit'],
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='green')
                ))
                
                # Individual trades as markers
                colors = ['green' if p > 0 else 'red' for p in df['profit_loss']]
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['cumulative_profit'],
                    mode='markers',
                    name='Trades',
                    marker=dict(color=colors, size=8),
                    text=[f"{row['symbol']}: {row['profit_loss']:.4f}" for _, row in df.iterrows()],
                    hovertemplate='%{text}<br>Time: %{x}<br>Cumulative P&L: %{y:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Trading Performance',
                    xaxis_title='Time',
                    yaxis_title='Cumulative P&L',
                    hovermode='x unified',
                    template='plotly_dark'
                )
                
                return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        except Exception as e:
            return {'error': f'Error generating performance chart: {str(e)}'}
    
    def _generate_signals_chart(self, hours: int = 24) -> Dict[str, Any]:
        """Generate signals chart data"""
        try:
            if not Path(self.db_path).exists():
                return {'error': 'Database not found'}
            
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, symbol, prediction, confidence
                    FROM trades 
                    WHERE timestamp > ?
                    ORDER BY timestamp ASC
                ''', (cutoff_time,))
                
                trades = cursor.fetchall()
                
                if not trades:
                    return {'data': [], 'layout': {}}
                
                df = pd.DataFrame(trades, columns=['timestamp', 'symbol', 'prediction', 'confidence'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Create plotly chart
                fig = go.Figure()
                
                # Signals by symbol
                for symbol in df['symbol'].unique():
                    symbol_df = df[df['symbol'] == symbol]
                    
                    fig.add_trace(go.Scatter(
                        x=symbol_df['timestamp'],
                        y=symbol_df['prediction'],
                        mode='markers',
                        name=symbol,
                        marker=dict(
                            size=symbol_df['confidence'] * 20,  # Size based on confidence
                            opacity=0.7
                        ),
                        text=[f"Confidence: {c:.3f}" for c in symbol_df['confidence']],
                        hovertemplate=f'{symbol}<br>Prediction: %{{y:.3f}}<br>%{{text}}<br>Time: %{{x}}<extra></extra>'
                    ))
                
                # Add horizontal lines for reference
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Neutral")
                fig.add_hline(y=0.7, line_dash="dot", line_color="green", annotation_text="Strong Buy")
                fig.add_hline(y=0.3, line_dash="dot", line_color="red", annotation_text="Strong Sell")
                
                fig.update_layout(
                    title='Trading Signals',
                    xaxis_title='Time',
                    yaxis_title='Prediction Strength',
                    yaxis=dict(range=[0, 1]),
                    hovermode='x unified',
                    template='plotly_dark'
                )
                
                return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        except Exception as e:
            return {'error': f'Error generating signals chart: {str(e)}'}
    
    def _get_model_status(self) -> Dict[str, Any]:
        """Get ML model status"""
        try:
            model_path = Path(self.config['data_sources']['model_path'])
            
            if not model_path.exists():
                return {'error': 'Model directory not found'}
            
            models = {}
            
            # Scan for model files
            for model_file in model_path.glob('*.keras'):
                model_name = model_file.stem
                model_info = {
                    'type': 'neural_network',
                    'file_size': model_file.stat().st_size,
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                }
                models[model_name] = model_info
            
            for model_file in model_path.glob('*.joblib'):
                model_name = model_file.stem
                model_info = {
                    'type': 'sklearn',
                    'file_size': model_file.stat().st_size,
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                }
                models[model_name] = model_info
            
            # Check for metadata files
            for metadata_file in model_path.glob('*_metadata.json'):
                model_name = metadata_file.stem.replace('_metadata', '')
                if model_name in models:
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        models[model_name]['metadata'] = metadata
                    except Exception as e:
                        models[model_name]['metadata_error'] = str(e)
            
            return {
                'total_models': len(models),
                'models': models,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {'error': f'Error getting model status: {str(e)}'}
    
    def _get_recent_logs(self, lines: int = 100) -> List[str]:
        """Get recent log entries"""
        try:
            log_path = Path(self.config['data_sources']['log_files_path'])
            
            if not log_path.exists():
                return ['Log directory not found']
            
            # Find the most recent log file
            log_files = list(log_path.glob('trading_bot_*.log'))
            
            if not log_files:
                return ['No log files found']
            
            # Get the most recent log file
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            
            # Read last N lines
            with open(latest_log, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        except Exception as e:
            return [f'Error reading logs: {str(e)}']
    
    def run(self):
        """Run the dashboard"""
        try:
            # Create templates directory and files if they don't exist
            self._create_template_files()
            
            host = self.config['server']['host']
            port = self.config['server']['port']
            debug = self.config['server'].get('debug', False)
            
            self.logger.info(f"Starting dashboard at http://{host}:{port}")
            self.app.run(host=host, port=port, debug=debug)
        
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {e}")
            raise
    
    def _create_template_files(self):
        """Create HTML template files"""
        templates_dir = Path('templates')
        templates_dir.mkdir(exist_ok=True)
        
        # Dashboard template
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #1e1e1e; color: white; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { background: #2d2d2d; padding: 20px; border-radius: 8px; }
        .status-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .status-label { color: #888; }
        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .chart-container { background: #2d2d2d; padding: 20px; border-radius: 8px; }
        .trades-table { background: #2d2d2d; padding: 20px; border-radius: 8px; overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #555; }
        th { background-color: #444; }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        .refresh-button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .alert-warning { background-color: #ff9800; color: white; }
        .alert-critical { background-color: #f44336; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Bot Dashboard</h1>
        <button class="refresh-button" onclick="refreshAll()">Refresh All</button>
    </div>
    
    <div class="status-grid">
        <div class="status-card">
            <div class="status-label">Bot Status</div>
            <div class="status-value" id="bot-status">Loading...</div>
        </div>
        <div class="status-card">
            <div class="status-label">Total Trades</div>
            <div class="status-value" id="total-trades">-</div>
        </div>
        <div class="status-card">
            <div class="status-label">Total P&L</div>
            <div class="status-value" id="total-pnl">-</div>
        </div>
        <div class="status-card">
            <div class="status-label">Win Rate</div>
            <div class="status-value" id="win-rate">-</div>
        </div>
        <div class="status-card">
            <div class="status-label">Accuracy</div>
            <div class="status-value" id="accuracy">-</div>
        </div>
    </div>
    
    <div class="charts-grid">
        <div class="chart-container">
            <div id="performance-chart"></div>
        </div>
        <div class="chart-container">
            <div id="signals-chart"></div>
        </div>
    </div>
    
    <div class="trades-table">
        <h3>Recent Trades</h3>
        <table id="trades-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Actual</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody id="trades-tbody">
            </tbody>
        </table>
    </div>
    
    <div id="alerts-container"></div>
    
    <script>
        function refreshAll() {
            loadStatus();
            loadPerformance();
            loadCharts();
            loadTrades();
            loadAlerts();
        }
        
        function loadStatus() {
            $.get('/api/status', function(data) {
                $('#bot-status').text(data.status);
                $('#total-trades').text(data.total_trades || 0);
            });
        }
        
        function loadPerformance() {
            $.get('/api/performance', function(data) {
                $('#total-pnl').text((data.total_profit || 0).toFixed(4)).removeClass('positive negative')
                    .addClass(data.total_profit > 0 ? 'positive' : 'negative');
                $('#win-rate').text(((data.win_rate || 0) * 100).toFixed(1) + '%');
                $('#accuracy').text(((data.accuracy || 0) * 100).toFixed(1) + '%');
            });
        }
        
        function loadCharts() {
            $.get('/api/charts/performance', function(data) {
                if (data.data) {
                    Plotly.newPlot('performance-chart', data.data, data.layout);
                }
            });
            
            $.get('/api/charts/signals', function(data) {
                if (data.data) {
                    Plotly.newPlot('signals-chart', data.data, data.layout);
                }
            });
        }
        
        function loadTrades() {
            $.get('/api/trades', function(data) {
                const tbody = $('#trades-tbody');
                tbody.empty();
                
                data.forEach(trade => {
                    const row = `<tr>
                        <td>${new Date(trade.timestamp).toLocaleString()}</td>
                        <td>${trade.symbol}</td>
                        <td>${trade.prediction.toFixed(3)}</td>
                        <td>${trade.confidence.toFixed(3)}</td>
                        <td>${trade.actual_result ? trade.actual_result.toFixed(3) : '-'}</td>
                        <td class="${trade.profit_loss > 0 ? 'positive' : 'negative'}">
                            ${trade.profit_loss ? trade.profit_loss.toFixed(4) : '-'}
                        </td>
                    </tr>`;
                    tbody.append(row);
                });
            });
        }
        
        function loadAlerts() {
            $.get('/api/alerts', function(data) {
                const container = $('#alerts-container');
                container.empty();
                
                if (data.length > 0) {
                    container.append('<h3>Recent Alerts</h3>');
                    data.forEach(alert => {
                        const alertDiv = `<div class="alert alert-${alert.severity}">
                            <strong>${alert.alert_type}</strong>: ${alert.message}
                            <small> - ${new Date(alert.timestamp).toLocaleString()}</small>
                        </div>`;
                        container.append(alertDiv);
                    });
                }
            });
        }
        
        // Auto-refresh
        setInterval(refreshAll, {{ refresh_interval * 1000 }});
        
        // Initial load
        $(document).ready(function() {
            refreshAll();
        });
    </script>
</body>
</html>
"""
        
        with open(templates_dir / 'dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        # Logs template
        logs_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Logs</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body { font-family: 'Courier New', monospace; margin: 0; padding: 20px; background-color: #1e1e1e; color: white; }
        .header { text-align: center; margin-bottom: 20px; }
        .logs-container { background: #2d2d2d; padding: 20px; border-radius: 8px; height: 80vh; overflow-y: auto; }
        .log-line { margin: 2px 0; white-space: pre-wrap; }
        .log-error { color: #f44336; }
        .log-warning { color: #ff9800; }
        .log-info { color: #4CAF50; }
        .refresh-button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Bot Logs</h1>
        <button class="refresh-button" onclick="loadLogs()">Refresh</button>
        <a href="/" style="color: white; margin-left: 20px;">Back to Dashboard</a>
    </div>
    
    <div class="logs-container" id="logs-container">
        Loading logs...
    </div>
    
    <script>
        function loadLogs() {
            $.get('/api/logs', function(data) {
                const container = $('#logs-container');
                container.empty();
                
                data.forEach(line => {
                    let className = 'log-line';
                    if (line.includes('ERROR')) className += ' log-error';
                    else if (line.includes('WARNING')) className += ' log-warning';
                    else if (line.includes('INFO')) className += ' log-info';
                    
                    container.append(`<div class="${className}">${line}</div>`);
                });
                
                // Scroll to bottom
                container.scrollTop(container[0].scrollHeight);
            });
        }
        
        // Auto-refresh logs every 30 seconds
        setInterval(loadLogs, 30000);
        
        // Initial load
        $(document).ready(function() {
            loadLogs();
        });
    </script>
</body>
</html>
"""
        
        with open(templates_dir / 'logs.html', 'w') as f:
            f.write(logs_html)

def main():
    """Main entry point for dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Bot Dashboard')
    parser.add_argument('--config', default='config/dashboard_config.json',
                       help='Path to dashboard configuration file')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='Port to bind to')
    
    args = parser.parse_args()
    
    # Initialize dashboard
    dashboard = TradingBotDashboard(config_path=args.config)
    
    # Override config with command line arguments
    dashboard.config['server']['host'] = args.host
    dashboard.config['server']['port'] = args.port
    
    try:
        # Run dashboard
        dashboard.run()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")

if __name__ == "__main__":
    main()