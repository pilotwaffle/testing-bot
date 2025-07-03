"""
File: fix_dashboard_mismatch.py
Location: E:\Trade Chat Bot\G Trading Bot\fix_dashboard_mismatch.py

Dashboard HTML/JS Mismatch Fix
Fixes the mismatch between dashboard.html and dashboard.js expectations
"""

import shutil
from datetime import datetime
from pathlib import Path

def backup_files():
    """Backup dashboard files"""
    files_to_backup = ["templates/dashboard.html", "static/js/dashboard.js"]
    backups = []
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            backup_name = f"{file_path}.backup_mismatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_name)
            backups.append(backup_name)
            print(f"📁 Backup created: {backup_name}")
    
    return backups

def create_fixed_dashboard_html():
    """Create dashboard.html with all required elements for dashboard.js"""
    print("🔧 Creating Fixed dashboard.html")
    print("=" * 50)
    
    fixed_dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Industrial Crypto Trading Bot v3.0</title>
    <link rel="stylesheet" href="/static/css/style.css">
    <link rel="icon" href="data:,">  <!-- Fixes favicon 404 -->
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Industrial Crypto Trading Bot v3.0</h1>
            <p>Enhanced with ML Features and Gemini AI | <span id="lastUpdate">Never</span></p>
        </div>

        <!-- Status Bar -->
        <div class="status-bar">
            <div class="status {{ 'running' if status == 'RUNNING' else 'stopped' }}">
                Status: {{ status }}
            </div>
            <div>
                ML Models: <span id="mlModelsCount">{{ ml_status|length }}</span> | 
                Strategies: <span id="activeStrategiesCount">{{ active_strategies|length }}</span> | 
                AI: <span id="aiStatus">{{ "🤖 Enabled" if ai_enabled else "📝 Basic" }}</span> |
                Connection: <span id="connectionStatus" class="status-value">Checking...</span>
            </div>
        </div>

        <!-- Performance Metrics -->
        <div class="card">
            <h3>Portfolio Performance</h3>
            <div class="metrics">
                <div class="metric">
                    <h4>Total Value</h4>
                    <p class="positive" data-metric="total_value" id="totalValue">${{ "%.2f"|format(metrics.total_value) }}</p>
                </div>
                <div class="metric">
                    <h4>Cash Balance</h4>
                    <p id="cashBalance">${{ "%.2f"|format(metrics.cash_balance) }}</p>
                </div>
                <div class="metric">
                    <h4>Unrealized P&L</h4>
                    <p class="{{ 'positive' if metrics.unrealized_pnl >= 0 else 'negative' }}" data-metric="pnl_24h" id="unrealizedPnl">${{ "%.2f"|format(metrics.unrealized_pnl) }}</p>
                </div>
                <div class="metric">
                    <h4>Total Profit</h4>
                    <p class="{{ 'positive' if metrics.total_profit >= 0 else 'negative' }}" id="totalProfit">${{ "%.2f"|format(metrics.total_profit) }}</p>
                </div>
                <div class="metric">
                    <h4>Open Positions</h4>
                    <p id="openPositions">{{ metrics.num_positions }}</p>
                </div>
            </div>
        </div>

        <!-- Trading Controls -->
        <div class="card">
            <h3>Trading Controls</h3>
            <button id="startBot" class="button success" onclick="startTrading()">Start Trading</button>
            <button id="stopBot" class="button danger" onclick="stopTrading()">Stop Trading</button>
            <button class="button" onclick="getStatus()">Get Status</button>
            <button class="button" onclick="getPositions()">View Positions</button>
            <button id="refreshData" class="button" onclick="getMarketData()">Market Data</button>
            <div class="response-display" id="trading-response">
                Click any button above to see real-time responses!
            </div>
        </div>

        <!-- Market Data Display -->
        <div class="card">
            <h3>Market Data</h3>
            <div class="market-data" id="marketData">
                <div class="price-card">
                    <div class="symbol">BTC/USDT</div>
                    <div class="price">$43,250.50</div>
                    <div class="change positive">+2.5%</div>
                </div>
                <div class="price-card">
                    <div class="symbol">ETH/USDT</div>
                    <div class="price">$2,650.75</div>
                    <div class="change positive">+1.8%</div>
                </div>
                <div class="price-card">
                    <div class="symbol">ADA/USDT</div>
                    <div class="price">$0.485</div>
                    <div class="change negative">-0.5%</div>
                </div>
            </div>
        </div>

        <!-- Enhanced ML Training Section -->
        <div class="ml-section">
            <h3>Enhanced ML Training (4 Advanced Models)</h3>
            <p>Train sophisticated AI models for market prediction and analysis:</p>

            <!-- ML System Test -->
            <div style="margin-bottom: 15px;">
                <button class="button warning" onclick="testMLSystem()">Test ML System</button>
                <div class="response-display" id="ml-test-response">Click to test ML system functionality...</div>
            </div>

            <!-- ML Models Training Section -->
            <div class="ml-models">
                {% if ml_status %}
                    {% for model_key, info in ml_status.items() %}
                    <div class="ml-model">
                        <h4>{{ info.model_type if info.model_type else model_key.replace('_', ' ').title() }}</h4>
                        <p><small>{{ info.description }}</small></p>
                        <p><strong>Status:</strong> {{ info.status|default('Ready') }}</p>
                        <p><strong>Last Trained:</strong> {{ info.last_trained|default('Not trained') }}</p>
                        <p><strong>Performance:</strong> {{ info.metric_name|default('N/A') }}: {{ info.metric_value_fmt|default('N/A') }}</p>
                        <p><strong>Training Samples:</strong> {{ info.training_samples|default('N/A') }}</p>
                        
                        <select id="{{ model_key }}-symbol">
                            {% for symbol in symbols %}
                            <option value="{{ symbol }}">{{ symbol }}</option>
                            {% endfor %}
                        </select>
                        <button class="button" onclick="trainModel('{{ model_key }}', '{{ model_key }}-symbol', '{{ model_key }}-response')">Train {{ info.model_type }}</button>
                        <div class="response-display" id="{{ model_key }}-response">Ready to train...</div>
                    </div>
                    {% endfor %}
                {% else %}
                    <!-- Default models if ml_status is empty -->
                    <div class="ml-model">
                        <h4>Lorentzian Classifier</h4>
                        <p><small>k-NN with Lorentzian distance, using RSI, Williams %R, CCI, ADX features</small></p>
                        <select id="lorentzian-symbol">
                            {% for symbol in symbols %}
                            <option value="{{ symbol }}">{{ symbol }}</option>
                            {% endfor %}
                        </select>
                        <button class="button" onclick="trainModel('lorentzian_classifier', 'lorentzian-symbol', 'lorentzian-response')">Train Lorentzian</button>
                        <div class="response-display" id="lorentzian-response">Ready to train...</div>
                    </div>

                    <div class="ml-model">
                        <h4>Neural Network</h4>
                        <p><small>Deep MLP for price prediction with technical indicators and volume analysis</small></p>
                        <select id="neural-symbol">
                            {% for symbol in symbols %}
                            <option value="{{ symbol }}">{{ symbol }}</option>
                            {% endfor %}
                        </select>
                        <button class="button" onclick="trainModel('neural_network', 'neural-symbol', 'neural-response')">Train Neural Net</button>
                        <div class="response-display" id="neural-response">Ready to train...</div>
                    </div>

                    <div class="ml-model">
                        <h4>Social Sentiment</h4>
                        <p><small>NLP analysis of Reddit, Twitter, Telegram sentiment</small></p>
                        <select id="sentiment-symbol">
                            {% for symbol in symbols %}
                            <option value="{{ symbol }}">{{ symbol }}</option>
                            {% endfor %}
                        </select>
                        <button class="button" onclick="trainModel('social_sentiment', 'sentiment-symbol', 'sentiment-response')">Train Sentiment</button>
                        <div class="response-display" id="sentiment-response">Ready to train...</div>
                    </div>

                    <div class="ml-model">
                        <h4>Risk Assessment</h4>
                        <p><small>Portfolio risk calculation using VaR, CVaR, volatility correlation</small></p>
                        <select id="risk-symbol">
                            {% for symbol in symbols %}
                            <option value="{{ symbol }}">{{ symbol }}</option>
                            {% endfor %}
                        </select>
                        <button class="button" onclick="trainModel('risk_assessment', 'risk-symbol', 'risk-response')">Train Risk Model</button>
                        <div class="response-display" id="risk-response">Ready to train...</div>
                    </div>
                {% endif %}
            </div>
        </div>

        <!-- Chat Interface -->
        <div class="card">
            <h3>Enhanced Chat with Gemini AI</h3>
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages" data-status="bot">
                    <div><strong>Bot:</strong> Hello! I'm your enhanced AI trading assistant with Gemini intelligence. Type 'help' for commands or ask me anything about trading!</div>
                </div>
                <div class="chat-input-container">
                    <input type="text" id="chat-input" class="chat-input" placeholder="Ask me about trading, portfolio, or market analysis..." onkeypress="handleChatEnter(event)">
                    <button id="sendChat" class="button" onclick="sendChatMessage()">Send</button>
                </div>
            </div>
            <p><small>Try: 'What's my portfolio status?', 'Should I buy Bitcoin?', 'Analyze ETH trends', 'help'</small></p>
        </div>

        <!-- Quick Links -->
        <div class="card">
            <h3>Quick Access</h3>
            <a href="/chat" class="button">Full Chat Interface</a>
            <a href="/api" class="button">API Information</a>
            <a href="/health" class="button">System Health</a>
            <button class="button warning" onclick="window.location.reload()">Refresh Dashboard</button>
        </div>
    </div>

    <script src="/static/js/dashboard.js"></script>
</body>
</html>'''
    
    try:
        with open("templates/dashboard.html", 'w', encoding='utf-8') as f:
            f.write(fixed_dashboard_html)
        print("✅ Fixed dashboard.html created with all required elements")
        return True
    except Exception as e:
        print(f"❌ Error creating fixed dashboard.html: {e}")
        return False

def create_compatible_dashboard_js():
    """Create dashboard.js that's compatible with the HTML"""
    print("\n🔧 Creating Compatible dashboard.js")
    print("=" * 50)
    
    compatible_js = '''// Enhanced Dashboard JavaScript - FIXED VERSION
// Compatible with dashboard.html elements, no more null reference errors

class TradingDashboard {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 3000;
        this.init();
    }

    init() {
        console.log('🚀 Trading Dashboard initializing...');
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadInitialData();
    }

    // FIXED: Safe element updates with null checks
    safeUpdateElement(elementId, content, property = 'textContent') {
        const element = document.getElementById(elementId);
        if (element) {
            if (property === 'textContent') {
                element.textContent = content;
            } else if (property === 'innerHTML') {
                element.innerHTML = content;
            } else if (property === 'className') {
                element.className = content;
            }
            return true;
        } else {
            console.warn(`⚠️ Element not found: ${elementId}`);
            return false;
        }
    }

    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            console.log(`Connecting to WebSocket: ${wsUrl}`);
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = (event) => {
                console.log('✅ WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('Connected', 'success');
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('🔴 WebSocket disconnected');
                this.updateConnectionStatus('Disconnected', 'error');
                this.scheduleReconnect();
            };
            
            this.ws.onerror = (event) => {
                console.error('WebSocket error:', event);
                this.updateConnectionStatus('Error', 'error');
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.updateConnectionStatus('Failed', 'error');
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect in ${this.reconnectInterval/1000}s... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connectWebSocket();
            }, this.reconnectInterval);
        } else {
            console.log('Max reconnection attempts reached. Switching to polling mode.');
            this.startPollingMode();
        }
    }

    startPollingMode() {
        console.log('🔄 Starting polling mode for updates');
        this.updateConnectionStatus('Polling Mode', 'warning');
        
        setInterval(() => {
            this.loadInitialData();
        }, 10000);
    }

    handleWebSocketMessage(data) {
        console.log('📨 WebSocket message received:', data.type);
        
        switch (data.type) {
            case 'real_time_update':
            case 'initial_data':
                this.updateMarketData(data.market_data);
                this.updatePortfolio(data.portfolio);
                this.updateTradingStatus(data.trading_status);
                break;
            
            case 'trading_status_update':
                this.showAlert(data.message, data.status === 'started' ? 'success' : 'info');
                break;
            
            case 'ml_training_update':
                this.showAlert(`ML Training: ${data.model_type} for ${data.symbol} completed`, 'success');
                break;
            
            case 'pong':
                console.log('🏓 WebSocket ping/pong successful');
                break;
            
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    updateMarketData(marketData) {
        if (!marketData) return;
        
        this.safeUpdateElement('marketData', 
            Object.entries(marketData).map(([symbol, data]) => `
                <div class="price-card">
                    <div class="symbol">${symbol}</div>
                    <div class="price">$${data.price.toFixed(2)}</div>
                    <div class="change ${data.change_24h >= 0 ? 'positive' : 'negative'}">
                        ${data.change_24h >= 0 ? '+' : ''}${data.change_24h.toFixed(2)}%
                    </div>
                </div>
            `).join(''), 'innerHTML');
    }

    updatePortfolio(portfolio) {
        if (!portfolio) return;
        
        this.safeUpdateElement('totalValue', `$${portfolio.total_value.toFixed(2)}`);
        this.safeUpdateElement('unrealizedPnl', `$${portfolio.pnl_24h.toFixed(2)}`);
        
        // Update PnL color
        const pnlElement = document.getElementById('unrealizedPnl');
        if (pnlElement) {
            pnlElement.className = portfolio.pnl_24h >= 0 ? 'positive' : 'negative';
        }
    }

    updateTradingStatus(status) {
        if (!status) return;
        
        const statusText = status.is_running ? 'Running' : 'Stopped';
        const statusClass = status.is_running ? 'running' : 'stopped';
        
        // Update multiple possible status elements
        this.safeUpdateElement('botStatus', statusText);
        this.safeUpdateElement('tradingStatus', statusText);
        
        const statusElement = document.querySelector('[data-status="bot"]');
        if (statusElement) {
            statusElement.className = statusClass;
        }
    }

    updateConnectionStatus(status, type) {
        this.safeUpdateElement('connectionStatus', status);
        
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.className = `status-value text-${type}`;
        }
    }

    setupEventListeners() {
        // Trading control buttons - with null checks
        const startBtn = document.getElementById('startBot');
        const stopBtn = document.getElementById('stopBot');
        const refreshBtn = document.getElementById('refreshData');
        const sendChatBtn = document.getElementById('sendChat');
        const chatInput = document.getElementById('chat-input');
        
        if (startBtn) startBtn.addEventListener('click', () => this.startBot());
        if (stopBtn) stopBtn.addEventListener('click', () => this.stopBot());
        if (refreshBtn) refreshBtn.addEventListener('click', () => this.refreshData());
        if (sendChatBtn) sendChatBtn.addEventListener('click', () => this.sendChatMessage());
        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.sendChatMessage();
            });
        }
    }

    async loadInitialData() {
        try {
            await this.updateStatus();
            await this.updateMarketDataFromAPI();
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }

    async updateStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            this.safeUpdateElement('lastUpdate', new Date().toLocaleTimeString());
            
            // Update component counts
            if (data.components) {
                const mlCount = Object.values(data.components).filter(Boolean).length;
                this.safeUpdateElement('mlModelsCount', mlCount);
            }
            
        } catch (error) {
            console.error('Error updating status:', error);
        }
    }

    async updateMarketDataFromAPI() {
        try {
            const response = await fetch('/api/market-data');
            const data = await response.json();
            
            if (data.status === 'success' && data.data) {
                this.updateMarketData(data.data);
            }
        } catch (error) {
            console.error('Error updating market data:', error);
        }
    }

    // Trading Controls
    async startBot() {
        this.showAlert('Starting trading bot...', 'info');
        try {
            const response = await fetch('/api/trading/start', { method: 'POST' });
            const data = await response.json();
            this.showAlert(data.message, data.status === 'success' ? 'success' : 'error');
            this.safeUpdateElement('trading-response', data.message);
        } catch (error) {
            this.showAlert('Failed to start trading bot: ' + error.message, 'error');
        }
    }

    async stopBot() {
        this.showAlert('Stopping trading bot...', 'info');
        try {
            const response = await fetch('/api/trading/stop', { method: 'POST' });
            const data = await response.json();
            this.showAlert(data.message, data.status === 'success' ? 'success' : 'error');
            this.safeUpdateElement('trading-response', data.message);
        } catch (error) {
            this.showAlert('Failed to stop trading bot: ' + error.message, 'error');
        }
    }

    async refreshData() {
        this.showAlert('Refreshing data...', 'info');
        await this.loadInitialData();
        this.showAlert('Data refreshed successfully!', 'success');
    }

    // Chat functionality
    async sendChatMessage() {
        const input = document.getElementById('chat-input');
        const messages = document.getElementById('chat-messages');
        
        if (!input || !messages || !input.value.trim()) return;

        const message = input.value.trim();
        input.value = '';

        this.addChatMessage('You', message);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.addChatMessage('Bot', data.response);
            } else {
                this.addChatMessage('Bot', 'Error: ' + data.message);
            }
        } catch (error) {
            this.addChatMessage('Bot', 'Error: ' + error.message);
        }
    }

    addChatMessage(sender, message) {
        const messages = document.getElementById('chat-messages');
        if (!messages) return;

        const messageDiv = document.createElement('div');
        messageDiv.style.marginBottom = '10px';
        messageDiv.style.padding = '8px 12px';
        messageDiv.style.borderRadius = '8px';
        
        if (sender === 'You') {
            messageDiv.style.background = 'rgba(0, 212, 255, 0.2)';
            messageDiv.style.border = '1px solid rgba(0, 212, 255, 0.3)';
        } else {
            messageDiv.style.background = 'rgba(76, 175, 80, 0.2)';
            messageDiv.style.border = '1px solid rgba(76, 175, 80, 0.3)';
        }

        messageDiv.innerHTML = `
            <strong>${sender}:</strong> ${message}
            <div style="font-size: 0.8em; opacity: 0.7; margin-top: 5px;">
                ${new Date().toLocaleTimeString()}
            </div>
        `;

        messages.appendChild(messageDiv);
        messages.scrollTop = messages.scrollHeight;
    }

    showAlert(message, type = 'info') {
        // Remove existing alerts
        document.querySelectorAll('.alert').forEach(alert => alert.remove());

        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;
        alert.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;

        const colors = {
            success: '#4CAF50',
            error: '#f44336',
            warning: '#ff9800',
            info: '#2196F3'
        };
        alert.style.backgroundColor = colors[type] || colors.info;

        document.body.appendChild(alert);
        setTimeout(() => alert.remove(), 5000);
    }

    startHeartbeat() {
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }
}

// Global functions for backward compatibility
window.startTrading = async function() {
    if (window.dashboard) await window.dashboard.startBot();
};

window.stopTrading = async function() {
    if (window.dashboard) await window.dashboard.stopBot();
};

window.getStatus = async function() {
    if (window.dashboard) await window.dashboard.updateStatus();
};

window.getPositions = async function() {
    if (window.dashboard) await window.dashboard.refreshData();
};

window.getMarketData = async function() {
    if (window.dashboard) await window.dashboard.updateMarketDataFromAPI();
};

window.testMLSystem = async function() {
    try {
        const response = await fetch('/api/ml/test', { method: 'POST' });
        const data = await response.json();
        const responseElement = document.getElementById('ml-test-response');
        if (responseElement) {
            responseElement.textContent = data.message || data.status;
        }
    } catch (error) {
        const responseElement = document.getElementById('ml-test-response');
        if (responseElement) {
            responseElement.textContent = '❌ Error: ' + error.message;
        }
    }
};

window.trainModel = async function(modelType, symbolSelectId, responseId) {
    if (window.dashboard) {
        const symbolSelect = document.getElementById(symbolSelectId);
        const symbol = symbolSelect ? symbolSelect.value : 'BTC/USDT';
        
        const responseElement = document.getElementById(responseId);
        if (responseElement) {
            responseElement.textContent = `🔄 Training ${modelType} for ${symbol}...`;
        }
        
        try {
            const response = await fetch(`/api/ml/train/${modelType}?symbol=${symbol}`, { 
                method: 'POST' 
            });
            const data = await response.json();
            
            if (responseElement) {
                if (data.status === 'success') {
                    responseElement.textContent = `✅ ${data.message}\\nAccuracy: ${data.accuracy || 'N/A'}`;
                } else {
                    responseElement.textContent = '❌ ' + data.message;
                }
            }
        } catch (error) {
            if (responseElement) {
                responseElement.textContent = '❌ Error: ' + error.message;
            }
        }
    }
};

window.sendChatMessage = async function() {
    if (window.dashboard) await window.dashboard.sendChatMessage();
};

window.handleChatEnter = function(event) {
    if (event.key === 'Enter' && window.dashboard) {
        window.dashboard.sendChatMessage();
    }
};

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing Trading Dashboard...');
    window.dashboard = new TradingDashboard();
    window.dashboard.startHeartbeat();
});
'''
    
    try:
        with open("static/js/dashboard.js", 'w', encoding='utf-8') as f:
            f.write(compatible_js)
        print("✅ Compatible dashboard.js created")
        return True
    except Exception as e:
        print(f"❌ Error creating compatible dashboard.js: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Dashboard HTML/JS Mismatch Fix")
    print("=" * 60)
    
    print("🎯 Fixing JavaScript errors:")
    print("   ❌ TypeError: Cannot set properties of null (setting 'textContent')")
    print("   ❌ dashboard.js looking for elements that don't exist")
    print("   ❌ Missing favicon causing 404")
    print()
    
    # Step 1: Backup files
    backup_files()
    
    # Step 2: Create fixed dashboard.html
    if create_fixed_dashboard_html():
        print("✅ dashboard.html fixed with all required elements")
    else:
        print("❌ Failed to fix dashboard.html")
        return
    
    # Step 3: Create compatible dashboard.js
    if create_compatible_dashboard_js():
        print("✅ dashboard.js fixed with null checks")
    else:
        print("❌ Failed to fix dashboard.js")
        return
    
    print("\\n🎉 DASHBOARD MISMATCH FIX COMPLETE!")
    print("=" * 60)
    
    print("🔄 Your server will auto-reload the template and JavaScript")
    print()
    print("📊 Expected results:")
    print("   ✅ No more JavaScript errors in F12 console")
    print("   ✅ Dashboard elements update correctly")
    print("   ✅ ML Training section displays properly")
    print("   ✅ All buttons and interactions work")
    print("   ✅ Chat interface functions properly")
    print("   ✅ No more favicon 404 error")
    print()
    print("🧪 Test the fix:")
    print("1. Refresh the dashboard page")
    print("2. Open F12 console")
    print("3. Should see no red errors")
    print("4. All elements should update properly")
    print()
    print("✅ Dashboard JavaScript and HTML are now synchronized!")

if __name__ == "__main__":
    main()