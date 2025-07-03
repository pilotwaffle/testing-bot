# Imports added by quick_fix_script.py for Elite Trading Bot V3.0
# Location: E:\Trade Chat Bot\G Trading Bot\core\kraken_dashboard_routes.py
# Added missing type hints and standard imports

from typing import List, Dict, Optional, Union, Any
import asyncio
import logging
import json
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime
import logging

# Add these routes to your existing main.py FastAPI application

def create_kraken_routes(app, trading_engine):
    """
    Create Kraken-specific routes for the dashboard
    Add this to your main.py after creating the FastAPI app
    """
    
    router = APIRouter(prefix="/kraken", tags=["kraken"])
    
    @router.get("/status")
    async def get_kraken_status():
        """Get Kraken integration status"""
        try:
            if not hasattr(trading_engine, 'kraken_integration'):
                return {"status": "not_integrated", "message": "Kraken integration not available"}
            
            dashboard_data = await trading_engine.get_kraken_dashboard_data()
            return {"status": "success", "data": dashboard_data}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting Kraken status: {str(e)}")
    
    @router.get("/market-data")
    async def get_kraken_market_data(symbols: Optional[str] = None):
        """Get Kraken market data"""
        try:
            if not hasattr(trading_engine, 'kraken_integration'):
                raise HTTPException(status_code=404, detail="Kraken integration not available")
            
            symbol_list = symbols.split(',') if symbols else None
            
            if hasattr(trading_engine.kraken_integration, 'kraken_client'):
                market_data = await trading_engine.kraken_integration.kraken_client.get_market_data(symbol_list)
                return {"status": "success", "data": market_data}
            else:
                raise HTTPException(status_code=503, detail="Kraken client not initialized")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting market data: {str(e)}")
    
    @router.get("/positions")
    async def get_kraken_positions():
        """Get Kraken positions"""
        try:
            if not hasattr(trading_engine, 'kraken_integration'):
                raise HTTPException(status_code=404, detail="Kraken integration not available")
            
            if hasattr(trading_engine.kraken_integration, 'kraken_client'):
                positions = await trading_engine.kraken_integration.kraken_client.get_positions()
                account_info = await trading_engine.kraken_integration.kraken_client.get_account_info()
                
                return {
                    "status": "success", 
                    "data": {
                        "positions": positions,
                        "account": account_info
                    }
                }
            else:
                raise HTTPException(status_code=503, detail="Kraken client not initialized")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting positions: {str(e)}")
    
    @router.get("/ml-analysis")
    async def get_kraken_ml_analysis(symbols: Optional[str] = None):
        """Get ML analysis for Kraken markets"""
        try:
            if not hasattr(trading_engine, 'kraken_integration'):
                raise HTTPException(status_code=404, detail="Kraken integration not available")
            
            symbol_list = symbols.split(',') if symbols else None
            analysis = await trading_engine.get_kraken_analysis(symbol_list)
            
            return {"status": "success", "data": analysis}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting ML analysis: {str(e)}")
    
    @router.post("/place-order")
    async def place_kraken_order(order_data: Dict):
        """Place order through Kraken"""
        try:
            if not hasattr(trading_engine, 'kraken_integration'):
                raise HTTPException(status_code=404, detail="Kraken integration not available")
            
            required_fields = ['symbol', 'side', 'size']
            for field in required_fields:
                if field not in order_data:
                    raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
            
            result = await trading_engine.place_kraken_order(
                symbol=order_data['symbol'],
                side=order_data['side'],
                size=float(order_data['size']),
                order_type=order_data.get('type', 'market'),
                price=float(order_data['price']) if order_data.get('price') else None
            )
            
            if result.get('success'):
                return {"status": "success", "data": result}
            else:
                raise HTTPException(status_code=400, detail=result.get('error', 'Order failed'))
                
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")
    
    @router.get("/instruments")
    async def get_kraken_instruments():
        """Get available Kraken instruments"""
        try:
            if not hasattr(trading_engine, 'kraken_integration'):
                raise HTTPException(status_code=404, detail="Kraken integration not available")
            
            if hasattr(trading_engine.kraken_integration, 'kraken_client'):
                instruments = await trading_engine.kraken_integration.kraken_client.get_instruments()
                return {"status": "success", "data": instruments}
            else:
                raise HTTPException(status_code=503, detail="Kraken client not initialized")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting instruments: {str(e)}")
    
    @router.get("/trading-summary")
    async def get_kraken_trading_summary():
        """Get comprehensive trading summary"""
        try:
            if not hasattr(trading_engine, 'kraken_integration'):
                raise HTTPException(status_code=404, detail="Kraken integration not available")
            
            if hasattr(trading_engine.kraken_integration, 'kraken_client'):
                summary = await trading_engine.kraken_integration.kraken_client.get_trading_summary()
                return {"status": "success", "data": summary}
            else:
                raise HTTPException(status_code=503, detail="Kraken client not initialized")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting trading summary: {str(e)}")
    
    @router.post("/retrain-models")
    async def retrain_kraken_models(background_tasks: BackgroundTasks, symbols: Optional[str] = None):
        """Retrain ML models for Kraken markets"""
        try:
            if not hasattr(trading_engine, 'kraken_integration'):
                raise HTTPException(status_code=404, detail="Kraken integration not available")
            
            symbol_list = symbols.split(',') if symbols else ['BTC/USD', 'ETH/USD']
            
            # Add retraining task to background
            async def retrain_task():
                if hasattr(trading_engine.kraken_integration, 'kraken_ml_analyzer'):
                    for symbol in symbol_list:
                        try:
                            await trading_engine.kraken_integration.kraken_ml_analyzer.retrain_models(symbol, force=True)
                        except Exception as e:
                            logging.error(f"Error retraining models for {symbol}: {e}")
            
            background_tasks.add_task(retrain_task)
            
            return {
                "status": "success", 
                "message": f"Model retraining started for {len(symbol_list)} symbols",
                "symbols": symbol_list
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error starting model retraining: {str(e)}")
    
    # Include router in main app
    app.include_router(router)
    
    return router


# Enhanced Dashboard HTML with Kraken Integration
KRAKEN_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Elite Trading Bot - Kraken Futures Integration</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .status-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .status-card h3 {
            margin-bottom: 15px;
            color: #ffd700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }

        .status-healthy { background-color: #4CAF50; }
        .status-warning { background-color: #FF9800; }
        .status-error { background-color: #F44336; }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .metric-item {
            text-align: center;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }

        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
        }

        .metric-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }

        .chart-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }

        .trading-controls {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }

        .control-group {
            display: flex;
            gap: 15px;
            align-items: center;
            margin-bottom: 15px;
        }

        .control-group label {
            min-width: 80px;
            font-weight: bold;
        }

        .control-group select,
        .control-group input {
            padding: 8px 12px;
            border: none;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.2);
            color: #fff;
            backdrop-filter: blur(5px);
        }

        .control-group select option {
            background: #2a5298;
            color: #fff;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }

        .btn-primary {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
        }

        .btn-secondary {
            background: linear-gradient(45deg, #2196F3, #1976D2);
            color: white;
        }

        .btn-danger {
            background: linear-gradient(45deg, #F44336, #D32F2F);
            color: white;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        .predictions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .prediction-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            backdrop-filter: blur(10px);
            border-left: 4px solid #ffd700;
        }

        .prediction-symbol {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .prediction-direction {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }

        .bullish {
            background-color: rgba(76, 175, 80, 0.3);
            color: #4CAF50;
        }

        .bearish {
            background-color: rgba(244, 67, 54, 0.3);
            color: #F44336;
        }

        .log-container {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            backdrop-filter: blur(10px);
            max-height: 300px;
            overflow-y: auto;
        }

        .log-entry {
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .log-timestamp {
            color: #888;
            margin-right: 10px;
        }

        .refresh-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.2);
            padding: 10px 15px;
            border-radius: 25px;
            backdrop-filter: blur(10px);
            font-size: 0.9em;
        }

        .loading {
            opacity: 0.7;
            pointer-events: none;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .pulse {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Elite Trading Bot v3.0</h1>
            <h2>Kraken Futures Integration Dashboard</h2>
            <p>Real-time monitoring with ML-powered predictions</p>
        </div>

        <div class="refresh-indicator" id="refreshIndicator">
            🔄 Auto-refresh: ON
        </div>

        <div class="status-grid">
            <div class="status-card">
                <h3>
                    <span class="status-indicator" id="krakenStatus"></span>
                    Kraken Integration Status
                </h3>
                <div id="krakenStatusDetails">Loading...</div>
            </div>

            <div class="status-card">
                <h3>
                    <span class="status-indicator status-healthy"></span>
                    Portfolio Performance
                </h3>
                <div class="metrics-grid" id="portfolioMetrics">
                    <div class="metric-item">
                        <div class="metric-value" id="totalValue">$0</div>
                        <div class="metric-label">Total Value</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="unrealizedPnL">$0</div>
                        <div class="metric-label">Unrealized P&L</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="returnPct">0%</div>
                        <div class="metric-label">Return %</div>
                    </div>
                </div>
            </div>

            <div class="status-card">
                <h3>
                    <span class="status-indicator status-healthy"></span>
                    Trading Activity
                </h3>
                <div class="metrics-grid" id="tradingMetrics">
                    <div class="metric-item">
                        <div class="metric-value" id="activePositions">0</div>
                        <div class="metric-label">Positions</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="activeOrders">0</div>
                        <div class="metric-label">Orders</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="tradeCount">0</div>
                        <div class="metric-label">Trades</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="trading-controls">
            <h3>🎯 Quick Trading Controls</h3>
            <div class="control-group">
                <label>Symbol:</label>
                <select id="symbolSelect">
                    <option value="BTC/USD">BTC/USD</option>
                    <option value="ETH/USD">ETH/USD</option>
                    <option value="LTC/USD">LTC/USD</option>
                    <option value="XRP/USD">XRP/USD</option>
                </select>
                
                <label>Side:</label>
                <select id="sideSelect">
                    <option value="buy">Buy</option>
                    <option value="sell">Sell</option>
                </select>
                
                <label>Size:</label>
                <input type="number" id="sizeInput" value="0.1" step="0.01" min="0.01">
                
                <button class="btn btn-primary" onclick="placeMarketOrder()">Place Market Order</button>
                <button class="btn btn-secondary" onclick="refreshData()">Refresh Data</button>
            </div>
        </div>

        <div class="chart-container">
            <h3>📊 Market Data & Predictions</h3>
            <canvas id="marketChart" width="400" height="200"></canvas>
        </div>

        <div class="status-card">
            <h3>🤖 ML Predictions</h3>
            <div class="predictions-grid" id="predictionsGrid">
                Loading ML predictions...
            </div>
        </div>

        <div class="log-container">
            <h3>📝 System Logs</h3>
            <div id="systemLogs"></div>
        </div>
    </div>

    <script>
        let refreshInterval;
        let chart;
        
        // Initialize dashboard
        async function initDashboard() {
            await refreshData();
            setupAutoRefresh();
            initChart();
            addLog('Dashboard initialized', 'INFO');
        }

        // Setup auto-refresh
        function setupAutoRefresh() {
            refreshInterval = setInterval(refreshData, 30000); // 30 seconds
        }

        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('marketChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'BTC/USD Price',
                        data: [],
                        borderColor: '#FFD700',
                        backgroundColor: 'rgba(255, 215, 0, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            labels: { color: '#fff' }
                        }
                    },
                    scales: {
                        x: { 
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        y: { 
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        }

        // Refresh all data
        async function refreshData() {
            document.getElementById('refreshIndicator').textContent = '🔄 Refreshing...';
            
            try {
                await Promise.all([
                    updateKrakenStatus(),
                    updatePortfolioData(),
                    updateMarketData(),
                    updateMLPredictions()
                ]);
                
                addLog('Data refresh completed', 'SUCCESS');
            } catch (error) {
                addLog(`Error refreshing data: ${error.message}`, 'ERROR');
            }
            
            document.getElementById('refreshIndicator').textContent = '🔄 Auto-refresh: ON';
        }

        // Update Kraken status
        async function updateKrakenStatus() {
            try {
                const response = await fetch('/kraken/status');
                const result = await response.json();
                
                const statusElement = document.getElementById('krakenStatus');
                const detailsElement = document.getElementById('krakenStatusDetails');
                
                if (result.status === 'success' && result.data.status === 'active') {
                    statusElement.className = 'status-indicator status-healthy';
                    detailsElement.innerHTML = `
                        <div><strong>Status:</strong> Active</div>
                        <div><strong>Health:</strong> ${result.data.health?.status || 'Unknown'}</div>
                        <div><strong>Trading:</strong> ${result.data.trading_enabled ? 'Enabled' : 'Disabled'}</div>
                        <div><strong>Last Update:</strong> ${new Date(result.data.last_update).toLocaleTimeString()}</div>
                    `;
                } else {
                    statusElement.className = 'status-indicator status-error';
                    detailsElement.innerHTML = `<div>Status: ${result.data?.status || 'Error'}</div>`;
                }
            } catch (error) {
                document.getElementById('krakenStatus').className = 'status-indicator status-error';
                document.getElementById('krakenStatusDetails').innerHTML = '<div>Connection Error</div>';
            }
        }

        // Update portfolio data
        async function updatePortfolioData() {
            try {
                const response = await fetch('/kraken/positions');
                const result = await response.json();
                
                if (result.status === 'success') {
                    const account = result.data.account;
                    const positions = result.data.positions;
                    
                    document.getElementById('totalValue').textContent = `$${(account.total_value || 0).toLocaleString()}`;
                    document.getElementById('returnPct').textContent = `${(account.return_pct || 0).toFixed(2)}%`;
                    document.getElementById('activePositions').textContent = positions.length;
                    
                    // Calculate unrealized P&L
                    const unrealizedPnL = positions.reduce((sum, pos) => sum + (pos.unrealized_pnl || 0), 0);
                    document.getElementById('unrealizedPnL').textContent = `$${unrealizedPnL.toFixed(2)}`;
                    document.getElementById('unrealizedPnL').style.color = unrealizedPnL >= 0 ? '#4CAF50' : '#F44336';
                }
            } catch (error) {
                addLog(`Error updating portfolio: ${error.message}`, 'ERROR');
            }
        }

        // Update market data
        async function updateMarketData() {
            try {
                const response = await fetch('/kraken/market-data?symbols=BTC/USD,ETH/USD,LTC/USD');
                const result = await response.json();
                
                if (result.status === 'success' && result.data.instruments) {
                    // Update trading metrics
                    const summary = result.data.summary;
                    document.getElementById('tradeCount').textContent = summary.active_symbols?.length || 0;
                    
                    // Update chart with BTC data
                    const btcData = result.data.instruments['BTC/USD'];
                    if (btcData && chart) {
                        const now = new Date().toLocaleTimeString();
                        chart.data.labels.push(now);
                        chart.data.datasets[0].data.push(btcData.last_price);
                        
                        // Keep only last 20 data points
                        if (chart.data.labels.length > 20) {
                            chart.data.labels.shift();
                            chart.data.datasets[0].data.shift();
                        }
                        
                        chart.update('none');
                    }
                }
            } catch (error) {
                addLog(`Error updating market data: ${error.message}`, 'ERROR');
            }
        }

        // Update ML predictions
        async function updateMLPredictions() {
            try {
                const response = await fetch('/kraken/ml-analysis?symbols=BTC/USD,ETH/USD,LTC/USD');
                const result = await response.json();
                
                if (result.status === 'success' && result.data.predictions) {
                    const predictionsGrid = document.getElementById('predictionsGrid');
                    predictionsGrid.innerHTML = '';
                    
                    Object.entries(result.data.predictions).forEach(([symbol, predictions]) => {
                        const prediction1h = predictions['1h'];
                        if (prediction1h) {
                            const card = document.createElement('div');
                            card.className = 'prediction-card';
                            
                            const direction = prediction1h.direction || 'neutral';
                            const confidence = (prediction1h.direction_confidence || 0) * 100;
                            const returnPct = (prediction1h.predicted_return || 0) * 100;
                            
                            card.innerHTML = `
                                <div class="prediction-symbol">${symbol}</div>
                                <div>Direction: <span class="prediction-direction ${direction}">${direction.toUpperCase()}</span></div>
                                <div>Confidence: ${confidence.toFixed(1)}%</div>
                                <div>Predicted Return: ${returnPct.toFixed(2)}%</div>
                                <div>Current Price: $${prediction1h.current_price?.toFixed(2) || 'N/A'}</div>
                            `;
                            
                            predictionsGrid.appendChild(card);
                        }
                    });
                }
            } catch (error) {
                addLog(`Error updating ML predictions: ${error.message}`, 'ERROR');
            }
        }

        // Place market order
        async function placeMarketOrder() {
            const symbol = document.getElementById('symbolSelect').value;
            const side = document.getElementById('sideSelect').value;
            const size = parseFloat(document.getElementById('sizeInput').value);
            
            if (!size || size <= 0) {
                addLog('Invalid order size', 'ERROR');
                return;
            }
            
            try {
                const response = await fetch('/kraken/place-order', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        symbol: symbol,
                        side: side,
                        size: size,
                        type: 'market'
                    })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    addLog(`Order placed: ${side.toUpperCase()} ${size} ${symbol}`, 'SUCCESS');
                    setTimeout(refreshData, 2000); // Refresh after 2 seconds
                } else {
                    addLog(`Order failed: ${result.detail || 'Unknown error'}`, 'ERROR');
                }
            } catch (error) {
                addLog(`Error placing order: ${error.message}`, 'ERROR');
            }
        }

        // Add log entry
        function addLog(message, level = 'INFO') {
            const logsContainer = document.getElementById('systemLogs');
            const timestamp = new Date().toLocaleTimeString();
            
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            
            let color = '#fff';
            if (level === 'SUCCESS') color = '#4CAF50';
            else if (level === 'ERROR') color = '#F44336';
            else if (level === 'WARNING') color = '#FF9800';
            
            logEntry.innerHTML = `
                <span class="log-timestamp">[${timestamp}]</span>
                <span style="color: ${color}">[${level}]</span>
                ${message}
            `;
            
            logsContainer.insertBefore(logEntry, logsContainer.firstChild);
            
            // Keep only last 50 log entries
            while (logsContainer.children.length > 50) {
                logsContainer.removeChild(logsContainer.lastChild);
            }
        }

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>
"""

def add_kraken_dashboard_route(app):
    """Add Kraken dashboard route to FastAPI app"""
    
    @app.get("/kraken-dashboard", response_class=HTMLResponse)
    async def kraken_dashboard():
        """Serve Kraken-enhanced dashboard"""
        return KRAKEN_DASHBOARD_HTML