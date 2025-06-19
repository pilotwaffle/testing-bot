// static/js/dashboard.js - FIXED VERSION
// WebSocket connection for real-time updates
let ws = null;
const chatMessagesContainer = document.getElementById('chat-messages');

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = function() {
        console.log('WebSocket connected');
        addChatMessage('System', 'WebSocket connected for real-time updates');
    };

    ws.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket received:', data); // DEBUG
            
            if (data.type === 'chat_response') {
                addChatMessage('Bot', data.message || data.response);
            } else if (data.type === 'market_update') {
                console.log('Live Market Update (via WS):', Object.keys(data.data).length, 'symbols');
            } else if (data.type === 'bot_status') {
                console.log('Bot Status Update (via WS):', data.status);
                const statusBar = document.querySelector('.status-bar .status');
                if (statusBar) {
                    statusBar.textContent = `Status: ${data.status}`;
                    statusBar.className = `status ${data.status.toLowerCase()}`;
                }
            } else {
                console.log('Unhandled WS message type:', data.type, data);
            }
        } catch (e) {
            // Handle plain text messages (fallback)
            console.log('WebSocket plain text message:', event.data);
            addChatMessage('Bot', event.data);
        }
    };

    ws.onclose = function() {
        console.warn('WebSocket disconnected. Attempting to reconnect in 3s...');
        addChatMessage('System', 'WebSocket disconnected. Reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        addChatMessage('System', `WebSocket error. Using HTTP fallback.`);
    };
}

// Initialize WebSocket on page load
document.addEventListener('DOMContentLoaded', connectWebSocket);

// Trading Controls Functions
async function startTrading() {
    updateResponse('trading-response', 'Starting trading...', 'loading');
    try {
        const response = await fetch('/api/start', { method: 'POST' });
        const data = await response.json();
        updateResponse('trading-response', data.message || 'Trading started!', response.ok ? 'success' : 'error');
    } catch (error) {
        updateResponse('trading-response', `Client-side error: ${error.message}`, 'error');
    }
}

async function stopTrading() {
    updateResponse('trading-response', 'Stopping trading...', 'loading');
    try {
        const response = await fetch('/api/stop', { method: 'POST' });
        const data = await response.json();
        updateResponse('trading-response', data.message || 'Trading stopped!', response.ok ? 'success' : 'error');
    } catch (error) {
        updateResponse('trading-response', `Client-side error: ${error.message}`, 'error');
    }
}

async function getStatus() {
    updateResponse('trading-response', 'Getting status...', 'loading');
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (response.ok && data) {
            const statusText = `
Status: ${data.running ? 'Running' : 'Stopped'}
Total Value: ${data.total_account_value?.toFixed(2) || 'N/A'}
Active Strategies: ${data.active_strategies_count || 0}
ML Models: ${data.ml_models_loaded || 0}
Market Feeds: ${Object.keys(data.current_market_data || {}).length}`;
            updateResponse('trading-response', statusText, 'success');
        } else {
            updateResponse('trading-response', data.detail || data.error || 'Failed to get status.', 'error');
        }
    } catch (error) {
        updateResponse('trading-response', `Client-side error: ${error.message}`, 'error');
    }
}

async function getPositions() {
    updateResponse('trading-response', 'Getting positions...', 'loading');
    try {
        const response = await fetch('/api/positions');
        const data = await response.json();
        
        if (response.ok) {
            const positions = Object.keys(data).length === 0 ?
                'No open positions' :
                JSON.stringify(data, null, 2);
            updateResponse('trading-response', positions, 'success');
        } else {
            updateResponse('trading-response', data.detail || data.error || 'Failed to get positions.', 'error');
        }
    } catch (error) {
        updateResponse('trading-response', `Client-side error: ${error.message}`, 'error');
    }
}

async function getMarketData() {
    updateResponse('trading-response', 'Getting market data...', 'loading');
    try {
        const response = await fetch('/api/market-data');
        const data = await response.json();
        
        if (response.ok) {
            let marketInfo = 'Live Market Data:\n';
            if (Object.keys(data).length === 0) {
                marketInfo += 'No market data available yet.';
            } else {
                Object.entries(data).forEach(([symbol, info]) => {
                    marketInfo += `${symbol}: ${info.price?.toFixed(2) || 'N/A'} (${info.change_24h?.toFixed(1) || '0'}%)\n`;
                });
            }
            updateResponse('trading-response', marketInfo, 'success');
        } else {
            updateResponse('trading-response', data.detail || data.error || 'Failed to get market data.', 'error');
        }
    } catch (error) {
        updateResponse('trading-response', `Client-side error: ${error.message}`, 'error');
    }
}

// ML Training Functions
async function testMLSystem() {
    updateResponse('ml-test-response', 'Testing ML system...', 'loading');
    try {
        const response = await fetch('/api/ml/test');
        const data = await response.json();

        if (response.ok && data.success) {
            const result = `
ML System Test: PASSED
Test Accuracy: ${data.test_accuracy}
Dependencies: scikit-learn v${data.scikit_learn_version}, numpy v${data.numpy_version}, pandas v${data.pandas_version}
Message: ${data.message}`;
            updateResponse('ml-test-response', result, 'success');
        } else {
            updateResponse('ml-test-response', `Test failed: ${data.detail || data.error}`, 'error');
        }
    } catch (error) {
        updateResponse('ml-test-response', `Client-side error: ${error.message}`, 'error');
    }
}

async function trainModel(modelType, symbolSelectId, responseId) {
    const symbolSelect = document.getElementById(symbolSelectId);
    const symbol = symbolSelect?.value || 'BTC/USDT';

    updateResponse(responseId, `Training ${modelType} for ${symbol}...`, 'loading');

    try {
        const formData = new FormData();
        formData.append('model_type', modelType);
        formData.append('symbol', symbol);

        const response = await fetch('/api/ml/train', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            const result = `
${data.model_type} Training Complete!
Performance: ${data.accuracy || data.performance}
Features Used (${data.features_used?.length || 0}): ${data.features_used?.join(', ') || 'N/A'}
Training Data Points: ${data.training_data_points}
Description: ${data.description || 'N/A'}`;
            updateResponse(responseId, result, 'success');
        } else {
            updateResponse(responseId, `Training failed: ${data.detail || data.error}`, 'error');
        }
    } catch (error) {
        updateResponse(responseId, `Client-side error: ${error.message}`, 'error');
    }
}

// FIXED Chat Functions
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    if (!input) {
        console.error('Chat input element not found!');
        return;
    }
    
    const message = input.value.trim();
    if (!message) return;

    console.log('Sending chat message:', message); // DEBUG
    addChatMessage('You', message);
    input.value = '';

    // Try HTTP POST first (more reliable)
    try {
        console.log('Attempting HTTP POST to /api/chat'); // DEBUG
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        console.log('Chat API response status:', response.status); // DEBUG
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Chat API error:', response.status, errorText);
            addChatMessage('Bot', `Error ${response.status}: ${errorText}`);
            return;
        }

        const data = await response.json();
        console.log('Chat API response data:', data); // DEBUG
        
        // Handle the response structure from ChatManager
        if (data.response) {
            addChatMessage('Bot', data.response);
        } else if (data.error) {
            addChatMessage('Bot', `Error: ${data.error}`);
        } else if (data.detail) {
            addChatMessage('Bot', `Error: ${data.detail}`);
        } else {
            addChatMessage('Bot', 'Unknown response format');
            console.log('Full response:', data);
        }
        
    } catch (error) {
        console.error('Chat HTTP error:', error);
        addChatMessage('Bot', `Connection error: ${error.message}`);
        
        // Fallback to WebSocket if HTTP fails
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log('Fallback to WebSocket');
            ws.send(JSON.stringify({ type: 'chat', message: message }));
        }
    }
}

function handleChatEnter(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendChatMessage();
    }
}

function addChatMessage(sender, message) {
    if (!chatMessagesContainer) {
        console.error('Chat messages container not found!');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender === 'You' ? 'user-message' : 'bot-message'}`;
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatMessagesContainer.appendChild(messageDiv);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
}

// Utility Function to update response display
function updateResponse(elementId, message, type) {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`Element with ID '${elementId}' not found.`);
        return;
    }
    element.textContent = message;
    element.className = `response-display ${type}`;
}

// Auto-refresh market data every 30 seconds
setInterval(async () => {
    try {
        const response = await fetch('/api/market-data');
        if (response.ok) {
            const data = await response.json();
            console.log('Dashboard market data auto-refreshed via HTTP.');
        }
    } catch (error) {
        console.error('Market data auto-refresh failed:', error);
    }
}, 30000);

// DEBUG: Log when script loads
console.log('Enhanced Dashboard Logic Loaded with Chat Debugging.');