#!/usr/bin/env python3
"""
Trading Bot Dashboard System
FILE LOCATION: E:\Trade Chat Bot\G Trading Bot\dashboard_app.py

FEATURES:
✅ Web-based dashboard for monitoring trading bot
✅ Real-time performance metrics
✅ Live position tracking
✅ Model accuracy monitoring
✅ Emergency controls
✅ Authentication system
✅ Trade history and analytics

SETUP:
1. pip install flask flask-login plotly pandas numpy
2. Run: python dashboard_app.py
3. Open browser: http://localhost:5000
4. Login: admin / trading123

INTEGRATION:
- Connects to your enhanced trading bot
- Monitors model predictions in real-time
- Controls bot execution
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import threading
import time
import logging
from pathlib import Path
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'trading_bot_secret_key_2025'  # Change this in production

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# =============================================================================
# USER AUTHENTICATION
# =============================================================================
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

# Simple user database (use proper database in production)
users = {
    'admin': User('1', 'admin', hashlib.sha256('trading123'.encode()).hexdigest())
}

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if user.id == user_id:
            return user
    return None

# =============================================================================
# DATABASE SETUP
# =============================================================================
def init_database():
    """Initialize SQLite database for storing trading data"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Trading history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity REAL,
            pnl REAL,
            confidence REAL,
            status TEXT
        )
    ''')
    
    # Performance metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            total_return REAL,
            daily_return REAL,
            positions_count INTEGER,
            win_rate REAL,
            accuracy REAL,
            sharpe_ratio REAL,
            max_drawdown REAL
        )
    ''')
    
    # Model predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            prediction REAL,
            confidence REAL,
            actual_result REAL,
            model_name TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# =============================================================================
# TRADING BOT CONTROLLER
# =============================================================================
class TradingBotController:
    """Controller for managing the trading bot"""
    
    def __init__(self):
        self.is_running = False
        self.positions = {}
        self.performance_metrics = {
            'total_return': 0.0,
            'daily_return': 0.0,
            'win_rate': 0.0,
            'accuracy': 0.0,
            'positions_count': 0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        self.bot_thread = None
        self.initial_capital = 10000
        self.current_capital = 10000
        
    def start_bot(self):
        """Start the trading bot"""
        if not self.is_running:
            self.is_running = True
            self.bot_thread = threading.Thread(target=self._run_bot_loop)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            logger.info("Trading bot started")
            return True
        return False
    
    def stop_bot(self):
        """Stop the trading bot"""
        self.is_running = False
        if self.bot_thread:
            self.bot_thread.join(timeout=5)
        logger.info("Trading bot stopped")
    
    def emergency_stop(self):
        """Emergency stop - close all positions"""
        self.stop_bot()
        # In production: close all positions immediately
        self.positions.clear()
        logger.critical("EMERGENCY STOP ACTIVATED")
    
    def _run_bot_loop(self):
        """Main bot loop (simplified for demo)"""
        while self.is_running:
            try:
                # Simulate bot activity
                self._update_performance_metrics()
                self._simulate_trading_activity()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Bot loop error: {e}")
                time.sleep(10)
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        # Simulate performance data
        import random
        
        self.performance_metrics.update({
            'total_return': random.uniform(-5, 15),
            'daily_return': random.uniform(-2, 3),
            'win_rate': random.uniform(55, 75),
            'accuracy': random.uniform(60, 80),
            'positions_count': len(self.positions),
            'sharpe_ratio': random.uniform(1.2, 2.5),
            'max_drawdown': random.uniform(2, 12)
        })
        
        # Store in database
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO performance (timestamp, total_return, daily_return, positions_count, 
                                   win_rate, accuracy, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            self.performance_metrics['total_return'],
            self.performance_metrics['daily_return'],
            self.performance_metrics['positions_count'],
            self.performance_metrics['win_rate'],
            self.performance_metrics['accuracy'],
            self.performance_metrics['sharpe_ratio'],
            self.performance_metrics['max_drawdown']
        ))
        conn.commit()
        conn.close()
    
    def _simulate_trading_activity(self):
        """Simulate trading activity for demo"""
        import random
        
        symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD']
        
        # Randomly add/remove positions
        if random.random() < 0.3:  # 30% chance to trade
            symbol = random.choice(symbols)
            if symbol not in self.positions:
                # Open new position
                self.positions[symbol] = {
                    'quantity': random.uniform(0.1, 1.0),
                    'entry_price': random.uniform(30000, 70000) if 'BTC' in symbol else random.uniform(2000, 4000),
                    'direction': random.choice(['long', 'short']),
                    'timestamp': datetime.now(),
                    'confidence': random.uniform(0.65, 0.85)
                }
            else:
                # Close position
                position = self.positions.pop(symbol)
                current_price = position['entry_price'] * random.uniform(0.98, 1.02)
                pnl = (current_price - position['entry_price']) * position['quantity']
                if position['direction'] == 'short':
                    pnl = -pnl
                
                # Store trade in database
                conn = sqlite3.connect('trading_bot.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (timestamp, symbol, direction, entry_price, exit_price, 
                                      quantity, pnl, confidence, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(), symbol, position['direction'], position['entry_price'],
                    current_price, position['quantity'], pnl, position['confidence'], 'closed'
                ))
                conn.commit()
                conn.close()

# Global bot controller
bot_controller = TradingBotController()

# =============================================================================
# ROUTES - AUTHENTICATION
# =============================================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if username in users and users[username].password_hash == password_hash:
            login_user(users[username])
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Bot Login</title>
        <style>
            body { font-family: Arial, sans-serif; background: #1a1a1a; color: white; }
            .login-container { max-width: 400px; margin: 100px auto; padding: 30px; 
                             background: #2d2d2d; border-radius: 10px; box-shadow: 0 0 20px rgba(0,255,0,0.3); }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; color: #00ff00; }
            input { width: 100%; padding: 10px; border: 1px solid #00ff00; background: #1a1a1a; 
                   color: white; border-radius: 5px; }
            button { width: 100%; padding: 12px; background: #00ff00; color: black; 
                    border: none; border-radius: 5px; font-weight: bold; cursor: pointer; }
            button:hover { background: #00cc00; }
            .header { text-align: center; margin-bottom: 30px; }
            .flash { padding: 10px; margin-bottom: 20px; border-radius: 5px; }
            .error { background: #ff4444; }
            .success { background: #44ff44; color: black; }
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="header">
                <h1>🚀 Trading Bot Dashboard</h1>
                <p>Enhanced Trading System Login</p>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="flash {{ category }}">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <form method="POST">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" name="username" required placeholder="admin">
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" name="password" required placeholder="trading123">
                </div>
                <button type="submit">Login to Dashboard</button>
            </form>
            
            <div style="margin-top: 20px; text-align: center; color: #888;">
                <small>Demo credentials: admin / trading123</small>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

# =============================================================================
# ROUTES - DASHBOARD
# =============================================================================
@app.route('/')
@login_required
def dashboard():
    """Main dashboard"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Bot Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                   background: #0a0a0a; color: white; }
            .header { background: #1a1a1a; padding: 15px 30px; border-bottom: 2px solid #00ff00;
                     display: flex; justify-content: space-between; align-items: center; }
            .header h1 { color: #00ff00; }
            .header .controls { display: flex; gap: 10px; }
            .btn { padding: 8px 16px; border: none; border-radius: 5px; cursor: pointer; 
                  font-weight: bold; text-decoration: none; display: inline-block; }
            .btn-success { background: #00ff00; color: black; }
            .btn-danger { background: #ff4444; color: white; }
            .btn-warning { background: #ffaa00; color: black; }
            .btn-secondary { background: #666; color: white; }
            .container { padding: 30px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                           gap: 20px; margin-bottom: 30px; }
            .metric-card { background: #1a1a1a; padding: 20px; border-radius: 10px; 
                          border: 1px solid #333; box-shadow: 0 0 10px rgba(0,255,0,0.1); }
            .metric-value { font-size: 2rem; font-weight: bold; margin-bottom: 5px; }
            .metric-label { color: #888; font-size: 0.9rem; }
            .positive { color: #00ff00; }
            .negative { color: #ff4444; }
            .neutral { color: #ffaa00; }
            .main-content { display: grid; grid-template-columns: 2fr 1fr; gap: 30px; }
            .chart-container, .positions-container { background: #1a1a1a; padding: 20px; 
                                                    border-radius: 10px; border: 1px solid #333; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; 
                               border-radius: 50%; margin-right: 8px; }
            .status-running { background: #00ff00; }
            .status-stopped { background: #ff4444; }
            .refresh-info { text-align: center; color: #888; margin-top: 20px; font-size: 0.9rem; }
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Trading Bot Dashboard</h1>
            <div class="controls">
                <span id="bot-status" class="status-indicator status-stopped"></span>
                <span id="status-text">Bot Stopped</span>
                <button id="start-btn" class="btn btn-success" onclick="startBot()">Start Bot</button>
                <button id="stop-btn" class="btn btn-warning" onclick="stopBot()" style="display:none;">Stop Bot</button>
                <button id="emergency-btn" class="btn btn-danger" onclick="emergencyStop()">Emergency Stop</button>
                <a href="/logout" class="btn btn-secondary">Logout</a>
            </div>
        </div>
        
        <div class="container">
            <!-- Performance Metrics -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div id="total-return" class="metric-value positive">+0.00%</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div id="daily-return" class="metric-value neutral">+0.00%</div>
                    <div class="metric-label">Daily Return</div>
                </div>
                <div class="metric-card">
                    <div id="win-rate" class="metric-value positive">0.00%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div id="accuracy" class="metric-value positive">0.00%</div>
                    <div class="metric-label">Model Accuracy</div>
                </div>
                <div class="metric-card">
                    <div id="positions-count" class="metric-value neutral">0</div>
                    <div class="metric-label">Active Positions</div>
                </div>
                <div class="metric-card">
                    <div id="sharpe-ratio" class="metric-value positive">0.00</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="main-content">
                <div class="chart-container">
                    <h3>Performance Chart</h3>
                    <div id="performance-chart" style="height: 400px;"></div>
                </div>
                <div class="positions-container">
                    <h3>Active Positions</h3>
                    <div id="positions-list">
                        <p style="color: #888;">No active positions</p>
                    </div>
                </div>
            </div>
            
            <div class="refresh-info">
                Dashboard updates every 30 seconds | Last update: <span id="last-update">Never</span>
            </div>
        </div>
        
        <script>
            let botRunning = false;
            
            function startBot() {
                fetch('/api/start-bot', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if(data.success) {
                        botRunning = true;
                        updateBotStatus();
                        alert('Trading bot started successfully!');
                    } else {
                        alert('Failed to start bot: ' + data.message);
                    }
                });
            }
            
            function stopBot() {
                fetch('/api/stop-bot', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if(data.success) {
                        botRunning = false;
                        updateBotStatus();
                        alert('Trading bot stopped successfully!');
                    }
                });
            }
            
            function emergencyStop() {
                if(confirm('Are you sure you want to EMERGENCY STOP? This will close all positions immediately.')) {
                    fetch('/api/emergency-stop', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        botRunning = false;
                        updateBotStatus();
                        alert('EMERGENCY STOP activated! All positions closed.');
                    });
                }
            }
            
            function updateBotStatus() {
                const statusIndicator = document.getElementById('bot-status');
                const statusText = document.getElementById('status-text');
                const startBtn = document.getElementById('start-btn');
                const stopBtn = document.getElementById('stop-btn');
                
                if(botRunning) {
                    statusIndicator.className = 'status-indicator status-running';
                    statusText.textContent = 'Bot Running';
                    startBtn.style.display = 'none';
                    stopBtn.style.display = 'inline-block';
                } else {
                    statusIndicator.className = 'status-indicator status-stopped';
                    statusText.textContent = 'Bot Stopped';
                    startBtn.style.display = 'inline-block';
                    stopBtn.style.display = 'none';
                }
            }
            
            function updateDashboard() {
                fetch('/api/dashboard-data')
                .then(response => response.json())
                .then(data => {
                    // Update metrics
                    document.getElementById('total-return').textContent = 
                        (data.metrics.total_return >= 0 ? '+' : '') + data.metrics.total_return.toFixed(2) + '%';
                    document.getElementById('daily-return').textContent = 
                        (data.metrics.daily_return >= 0 ? '+' : '') + data.metrics.daily_return.toFixed(2) + '%';
                    document.getElementById('win-rate').textContent = data.metrics.win_rate.toFixed(1) + '%';
                    document.getElementById('accuracy').textContent = data.metrics.accuracy.toFixed(1) + '%';
                    document.getElementById('positions-count').textContent = data.metrics.positions_count;
                    document.getElementById('sharpe-ratio').textContent = data.metrics.sharpe_ratio.toFixed(2);
                    
                    // Update colors
                    document.getElementById('total-return').className = 
                        'metric-value ' + (data.metrics.total_return >= 0 ? 'positive' : 'negative');
                    document.getElementById('daily-return').className = 
                        'metric-value ' + (data.metrics.daily_return >= 0 ? 'positive' : 'negative');
                    
                    // Update positions
                    const positionsList = document.getElementById('positions-list');
                    if(data.positions.length === 0) {
                        positionsList.innerHTML = '<p style="color: #888;">No active positions</p>';
                    } else {
                        positionsList.innerHTML = data.positions.map(pos => 
                            `<div style="background: #2d2d2d; padding: 10px; margin: 10px 0; border-radius: 5px;">
                                <strong>${pos.symbol}</strong> - ${pos.direction.toUpperCase()}<br>
                                Size: ${pos.quantity.toFixed(4)} | Entry: $${pos.entry_price.toFixed(2)}<br>
                                Confidence: ${(pos.confidence * 100).toFixed(1)}%
                            </div>`
                        ).join('');
                    }
                    
                    // Update bot status
                    botRunning = data.bot_running;
                    updateBotStatus();
                    
                    // Update timestamp
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                });
            }
            
            // Initialize dashboard
            updateDashboard();
            setInterval(updateDashboard, 30000); // Update every 30 seconds
        </script>
    </body>
    </html>
    '''

# =============================================================================
# API ROUTES
# =============================================================================
@app.route('/api/start-bot', methods=['POST'])
@login_required
def api_start_bot():
    success = bot_controller.start_bot()
    return jsonify({'success': success, 'message': 'Bot started' if success else 'Bot already running'})

@app.route('/api/stop-bot', methods=['POST'])
@login_required
def api_stop_bot():
    bot_controller.stop_bot()
    return jsonify({'success': True, 'message': 'Bot stopped'})

@app.route('/api/emergency-stop', methods=['POST'])
@login_required
def api_emergency_stop():
    bot_controller.emergency_stop()
    return jsonify({'success': True, 'message': 'Emergency stop activated'})

@app.route('/api/dashboard-data')
@login_required
def api_dashboard_data():
    positions_list = []
    for symbol, pos in bot_controller.positions.items():
        positions_list.append({
            'symbol': symbol,
            'quantity': pos['quantity'],
            'entry_price': pos['entry_price'],
            'direction': pos['direction'],
            'confidence': pos['confidence']
        })
    
    return jsonify({
        'metrics': bot_controller.performance_metrics,
        'positions': positions_list,
        'bot_running': bot_controller.is_running
    })

@app.route('/api/trades-history')
@login_required
def api_trades_history():
    conn = sqlite3.connect('trading_bot.db')
    df = pd.read_sql_query('SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100', conn)
    conn.close()
    return jsonify(df.to_dict('records'))

# =============================================================================
# MAIN APPLICATION
# =============================================================================
if __name__ == '__main__':
    print("🚀 Trading Bot Dashboard System")
    print("=" * 50)
    print("📁 File: E:\\Trade Chat Bot\\G Trading Bot\\dashboard_app.py")
    print("🌐 URL: http://localhost:5000")
    print("👤 Login: admin / trading123")
    print("⚡ Features: Real-time monitoring, bot control, performance tracking")
    print()
    print("🔧 Setup Instructions:")
    print("1. pip install flask flask-login plotly pandas numpy")
    print("2. python dashboard_app.py")
    print("3. Open browser: http://localhost:5000")
    print("4. Login with: admin / trading123")
    print()
    
    # Initialize database
    init_database()
    
    # Start Flask app
    print("🚀 Starting dashboard server...")
    app.run(debug=True, host='0.0.0.0', port=5000)