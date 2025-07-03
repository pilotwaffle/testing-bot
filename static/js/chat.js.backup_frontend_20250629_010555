// static/js/chat.js - Enhanced Version with Advanced AI Features
let ws = null;
const chatMessagesContainer = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const commandSuggestionsContainer = document.getElementById('command-suggestions');
const insightsContainer = document.getElementById('proactive-insights');
const voiceButton = document.getElementById('voice-button');

// Enhanced message handling
class EnhancedMessageHandler {
    constructor() {
        this.messageTypes = {
            'text': this.renderTextMessage,
            'trade_signal': this.renderTradeSignal,
            'analysis': this.renderAnalysis,
            'command_result': this.renderCommandResult,
            'notification': this.renderNotification,
            'error': this.renderError
        };
        
        this.commandHistory = [];
        this.currentSuggestions = [];
    }
    
    renderTextMessage(sender, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender === 'You' ? 'user-message' : 'bot-message'}`;
        
        // Enhanced formatting with markdown-like support
        const formattedContent = this.formatContent(content);
        messageDiv.innerHTML = `
            <div class="message-header">
                <strong>${sender}:</strong>
                <span class="timestamp">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">${formattedContent}</div>
        `;
        
        return messageDiv;
    }
    
    renderTradeSignal(sender, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message trade-signal-message';
        
        const signal = metadata.signal || {};
        messageDiv.innerHTML = `
            <div class="trade-signal-card">
                <div class="signal-header">
                    <h4>🎯 Trade Signal</h4>
                    <span class="confidence ${this.getConfidenceClass(signal.confidence)}">
                        ${(signal.confidence * 100 || 0).toFixed(1)}% confidence
                    </span>
                </div>
                <div class="signal-details">
                    <div class="signal-row">
                        <span class="label">Symbol:</span>
                        <span class="value">${signal.symbol || 'N/A'}</span>
                    </div>
                    <div class="signal-row">
                        <span class="label">Action:</span>
                        <span class="value ${signal.action}">${(signal.action || 'N/A').toUpperCase()}</span>
                    </div>
                    <div class="signal-row">
                        <span class="label">Price:</span>
                        <span class="value">$${signal.price || 'Market'}</span>
                    </div>
                    <div class="signal-row">
                        <span class="label">Quantity:</span>
                        <span class="value">${signal.quantity || 'Auto'}</span>
                    </div>
                </div>
                <div class="signal-reason">
                    <strong>Reason:</strong> ${content}
                </div>
                ${signal.signal_id ? `
                    <div class="signal-actions">
                        <button class="execute-button" onclick="executeTradeSignal('${signal.signal_id}')">
                            Execute Trade
                        </button>
                        <button class="analyze-button" onclick="analyzeSignal('${signal.signal_id}')">
                            Analyze Further
                        </button>
                    </div>
                ` : ''}
            </div>
        `;
        
        return messageDiv;
    }
    
    renderAnalysis(sender, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message analysis-message';
        
        messageDiv.innerHTML = `
            <div class="analysis-card">
                <div class="analysis-header">
                    <h4>📊 AI Analysis</h4>
                    <span class="analysis-type">${metadata.analysis_type || 'Market Analysis'}</span>
                </div>
                <div class="analysis-content">
                    ${this.formatAnalysisContent(content, metadata)}
                </div>
                <div class="analysis-actions">
                    <button onclick="requestDetailedAnalysis('${metadata.symbol || ''}')">
                        Get Detailed Analysis
                    </button>
                    <button onclick="shareAnalysis('${metadata.analysis_id || ''}')">
                        Share Analysis
                    </button>
                </div>
            </div>
        `;
        
        return messageDiv;
    }
    
    renderCommandResult(sender, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message command-result-message';
        
        messageDiv.innerHTML = `
            <div class="command-result-card">
                <div class="command-header">
                    <h4>⚡ Command Result</h4>
                    <span class="command-name">${metadata.command || 'Command'}</span>
                </div>
                <pre class="command-output">${content}</pre>
                ${metadata.execution_time ? `
                    <div class="command-footer">
                        <span class="execution-time">Executed in ${metadata.execution_time.toFixed(2)}s</span>
                    </div>
                ` : ''}
            </div>
        `;
        
        return messageDiv;
    }
    
    renderNotification(sender, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message notification-message ${metadata.priority || 'info'}`;
        
        const icons = {
            'info': '💡',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅',
            'high': '🚨'
        };
        
        messageDiv.innerHTML = `
            <div class="notification-card">
                <div class="notification-icon">${icons[metadata.priority] || icons.info}</div>
                <div class="notification-content">
                    <div class="notification-title">${metadata.title || 'Notification'}</div>
                    <div class="notification-message">${content}</div>
                </div>
            </div>
        `;
        
        return messageDiv;
    }
    
    formatContent(content) {
        // Enhanced content formatting with emoji and structure preservation
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>')
            .replace(/(\$\d+(?:\.\d{2})?)/g, '<span class="price">$1</span>')
            .replace(/([+-]\d+(?:\.\d+)?%)/g, '<span class="percentage">$1</span>');
    }
    
    formatAnalysisContent(content, metadata) {
        // Special formatting for analysis content
        if (metadata.structured) {
            // Handle structured analysis data
            let formatted = '<div class="analysis-sections">';
            
            if (metadata.technical_indicators) {
                formatted += '<div class="indicator-section">';
                formatted += '<h5>Technical Indicators</h5>';
                for (const [key, value] of Object.entries(metadata.technical_indicators)) {
                    formatted += `<div class="indicator"><span class="indicator-name">${key}:</span> <span class="indicator-value">${value}</span></div>`;
                }
                formatted += '</div>';
            }
            
            formatted += `<div class="analysis-summary">${content}</div>`;
            formatted += '</div>';
            
            return formatted;
        }
        
        return this.formatContent(content);
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'high-confidence';
        if (confidence >= 0.6) return 'medium-confidence';
        return 'low-confidence';
    }
}

// Enhanced command suggestions
class CommandSuggestionManager {
    constructor() {
        this.commands = {
            '/status': { description: 'Get comprehensive bot status', category: 'status' },
            '/portfolio': { description: 'View portfolio details', category: 'portfolio' },
            '/analyze': { description: 'AI market analysis', category: 'analysis' },
            '/positions': { description: 'Current positions', category: 'portfolio' },
            '/strategies': { description: 'Manage strategies', category: 'strategy' },
            '/risk': { description: 'Risk assessment', category: 'risk' },
            '/market': { description: 'Market overview', category: 'market' },
            '/help': { description: 'Show all commands', category: 'help' },
            '/settings': { description: 'Configure preferences', category: 'settings' },
            '/history': { description: 'Conversation history', category: 'history' }
        };
        
        this.isVisible = false;
    }
    
    showSuggestions(input) {
        if (!commandSuggestionsContainer) return;
        
        if (input.startsWith('/')) {
            const matchingCommands = Object.entries(this.commands)
                .filter(([cmd]) => cmd.startsWith(input.toLowerCase()))
                .slice(0, 5);
            
            if (matchingCommands.length > 0) {
                this.renderSuggestions(matchingCommands);
                this.show();
            } else {
                this.hide();
            }
        } else {
            this.hide();
        }
    }
    
    renderSuggestions(commands) {
        commandSuggestionsContainer.innerHTML = commands.map(([cmd, info]) => `
            <div class="suggestion-item" onclick="selectCommand('${cmd}')">
                <div class="suggestion-command">${cmd}</div>
                <div class="suggestion-description">${info.description}</div>
                <div class="suggestion-category">${info.category}</div>
            </div>
        `).join('');
    }
    
    show() {
        if (commandSuggestionsContainer) {
            commandSuggestionsContainer.style.display = 'block';
            this.isVisible = true;
        }
    }
    
    hide() {
        if (commandSuggestionsContainer) {
            commandSuggestionsContainer.style.display = 'none';
            this.isVisible = false;
        }
    }
    
    handleKeyNavigation(event) {
        if (!this.isVisible) return false;
        
        const suggestions = commandSuggestionsContainer.querySelectorAll('.suggestion-item');
        let selectedIndex = Array.from(suggestions).findIndex(item => item.classList.contains('selected'));
        
        if (event.key === 'ArrowDown') {
            event.preventDefault();
            selectedIndex = (selectedIndex + 1) % suggestions.length;
            this.updateSelection(suggestions, selectedIndex);
            return true;
        } else if (event.key === 'ArrowUp') {
            event.preventDefault();
            selectedIndex = selectedIndex <= 0 ? suggestions.length - 1 : selectedIndex - 1;
            this.updateSelection(suggestions, selectedIndex);
            return true;
        } else if (event.key === 'Tab' || event.key === 'Enter') {
            event.preventDefault();
            if (selectedIndex >= 0 && suggestions[selectedIndex]) {
                const command = suggestions[selectedIndex].querySelector('.suggestion-command').textContent;
                selectCommand(command);
            }
            return true;
        }
        
        return false;
    }
    
    updateSelection(suggestions, selectedIndex) {
        suggestions.forEach((item, index) => {
            item.classList.toggle('selected', index === selectedIndex);
        });
    }
}

// Proactive insights manager
class ProactiveInsightsManager {
    constructor() {
        this.insights = [];
        this.maxInsights = 3;
    }
    
    showInsights(insights) {
        if (!insightsContainer || !insights || insights.length === 0) {
            this.hideInsights();
            return;
        }
        
        this.insights = insights.slice(0, this.maxInsights);
        this.renderInsights();
        this.showContainer();
    }
    
    renderInsights() {
        insightsContainer.innerHTML = `
            <div class="insights-header">
                <h4>💡 AI Insights</h4>
                <button class="close-insights" onclick="hideProactiveInsights()">×</button>
            </div>
            <div class="insights-list">
                ${this.insights.map((insight, index) => `
                    <div class="insight-item" data-index="${index}">
                        <div class="insight-content">${insight}</div>
                        <div class="insight-actions">
                            <button onclick="actOnInsight(${index})">Act on This</button>
                            <button onclick="dismissInsight(${index})">Dismiss</button>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    showContainer() {
        if (insightsContainer) {
            insightsContainer.style.display = 'block';
            insightsContainer.classList.add('show');
        }
    }
    
    hideInsights() {
        if (insightsContainer) {
            insightsContainer.style.display = 'none';
            insightsContainer.classList.remove('show');
        }
    }
    
    dismissInsight(index) {
        this.insights.splice(index, 1);
        if (this.insights.length === 0) {
            this.hideInsights();
        } else {
            this.renderInsights();
        }
    }
}

// Voice interface
class VoiceInterface {
    constructor() {
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.setupRecognition();
        } else if ('SpeechRecognition' in window) {
            this.recognition = new SpeechRecognition();
            this.setupRecognition();
        }
    }
    
    setupRecognition() {
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';
        
        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateVoiceButton();
        };
        
        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            chatInput.value = transcript;
            this.isListening = false;
            this.updateVoiceButton();
            sendMessage();
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.isListening = false;
            this.updateVoiceButton();
        };
        
        this.recognition.onend = () => {
            this.isListening = false;
            this.updateVoiceButton();
        };
    }
    
    startListening() {
        if (this.recognition && !this.isListening) {
            this.recognition.start();
        }
    }
    
    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
    }
    
    speakResponse(text) {
        if (this.synthesis) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.8;
            utterance.pitch = 1;
            this.synthesis.speak(utterance);
        }
    }
    
    updateVoiceButton() {
        if (voiceButton) {
            voiceButton.innerHTML = this.isListening ? '🔴 Stop' : '🎤 Voice';
            voiceButton.className = `voice-button ${this.isListening ? 'listening' : ''}`;
        }
    }
}

// Initialize enhanced components
const messageHandler = new EnhancedMessageHandler();
const suggestionManager = new CommandSuggestionManager();
const insightsManager = new ProactiveInsightsManager();
const voiceInterface = new VoiceInterface();

// Enhanced WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        addMessage('System', 'Connected to enhanced trading bot AI.', 'notification', { priority: 'success' });
        // Request initial status
        ws.send(JSON.stringify({ type: 'status' }));
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (e) {
            // Fallback for plain text messages
            addMessage('Bot', event.data);
            console.error('WebSocket message parsing error:', e, event.data);
        }
    };

    ws.onclose = () => {
        console.warn('WebSocket disconnected. Attempting to reconnect in 3s...');
        addMessage('System', 'Disconnected from bot. Reconnecting...', 'notification', { priority: 'warning' });
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addMessage('System', 'WebSocket connection error. Using HTTP fallback.', 'notification', { priority: 'error' });
    };
}

function handleWebSocketMessage(data) {
    switch(data.type) {
        case 'chat_response':
            handleEnhancedChatResponse(data);
            break;
        case 'trade_signal':
            addMessage('AI Signal', data.content, 'trade_signal', { signal: data.signal });
            break;
        case 'ai_analysis':
            addMessage('AI Analysis', data.analysis, 'analysis', data.metadata);
            break;
        case 'bot_status':
            handleStatusUpdate(data);
            break;
        case 'notification':
            addMessage('System', data.message, 'notification', { priority: data.priority, title: data.title });
            break;
        case 'proactive_insight':
            insightsManager.showInsights([data.message]);
            break;
        case 'error':
            addMessage('Error', data.message, 'error');
            break;
        default:
            console.log('Unhandled WS message type:', data.type, data);
            addMessage('System', `Received message: ${data.type}`, 'notification');
    }
}

function handleEnhancedChatResponse(data) {
    // Handle the enhanced response from the new chat manager
    const messageType = data.message_type || 'text';
    const intent = data.intent;
    const responseTime = data.response_time;
    
    // Add main response
    addMessage('Bot', data.message || data.response, messageType, {
        intent: intent,
        response_time: responseTime
    });
    
    // Show proactive insights
    if (data.proactive_insights && data.proactive_insights.length > 0) {
        insightsManager.showInsights(data.proactive_insights);
    }
    
    // Show command suggestions
    if (data.suggestions && data.suggestions.length > 0) {
        showCommandSuggestions(data.suggestions);
    }
    
    // Speak response if enabled
    if (window.speechEnabled && data.message) {
        voiceInterface.speakResponse(data.message);
    }
}

function handleStatusUpdate(data) {
    const statusMessage = `Status: ${data.status}. Portfolio: $${data.metrics?.total_value?.toFixed(2) || 'N/A'}`;
    addMessage('Bot Status', statusMessage, 'notification', { priority: 'info' });
}

// Enhanced message sending
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Hide suggestions
    suggestionManager.hide();
    
    // Add to command history
    messageHandler.commandHistory.push(message);
    
    addMessage('You', message);
    chatInput.value = '';

    // Track analytics
    const startTime = performance.now();

    if (ws && ws.readyState === WebSocket.OPEN) {
        // Send as enhanced JSON message
        ws.send(JSON.stringify({ 
            type: 'chat', 
            message: message,
            timestamp: new Date().toISOString(),
            session_id: getSessionId()
        }));
    } else {
        // Enhanced HTTP fallback
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message: message,
                    timestamp: new Date().toISOString(),
                    session_id: getSessionId()
                })
            });
            
            const data = await response.json();
            const endTime = performance.now();
            
            if (response.ok) {
                // Handle enhanced response format
                data.response_time = (endTime - startTime) / 1000;
                handleEnhancedChatResponse(data);
            } else {
                addMessage('Error', data.detail || data.error || 'Unknown error occurred.', 'error');
            }
        } catch (error) {
            addMessage('Error', `Connection error: ${error.message}`, 'error');
        }
    }
}

// Enhanced message adding with new handler
function addMessage(sender, content, messageType = 'text', metadata = {}) {
    if (!chatMessagesContainer) return;
    
    const renderMethod = messageHandler.messageTypes[messageType] || messageHandler.renderTextMessage;
    const messageElement = renderMethod.call(messageHandler, sender, content, metadata);
    
    chatMessagesContainer.appendChild(messageElement);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    
    // Add animation
    messageElement.style.opacity = '0';
    messageElement.style.transform = 'translateY(20px)';
    setTimeout(() => {
        messageElement.style.transition = 'all 0.3s ease';
        messageElement.style.opacity = '1';
        messageElement.style.transform = 'translateY(0)';
    }, 10);
}

// Enhanced input handling
function handleEnter(event) {
    // Handle command suggestions navigation
    if (suggestionManager.handleKeyNavigation(event)) {
        return;
    }
    
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Input event for command suggestions
function handleInput(event) {
    const value = event.target.value;
    suggestionManager.showSuggestions(value);
}

// Command suggestion functions
function selectCommand(command) {
    chatInput.value = command + ' ';
    chatInput.focus();
    suggestionManager.hide();
}

function showCommandSuggestions(suggestions) {
    if (!commandSuggestionsContainer) return;
    
    commandSuggestionsContainer.innerHTML = suggestions.map(suggestion => `
        <div class="command-suggestion" onclick="chatInput.value='${suggestion}'; chatInput.focus();">
            ${suggestion}
        </div>
    `).join('');
    
    commandSuggestionsContainer.style.display = 'block';
    setTimeout(() => suggestionManager.hide(), 5000); // Auto-hide after 5 seconds
}

// Proactive insights functions
function hideProactiveInsights() {
    insightsManager.hideInsights();
}

function actOnInsight(index) {
    const insight = insightsManager.insights[index];
    if (insight) {
        // Extract actionable command from insight
        let command = '';
        if (insight.includes('rebalancing')) command = '/rebalance';
        else if (insight.includes('risk')) command = '/risk';
        else if (insight.includes('strategy')) command = '/strategies';
        else command = '/analyze';
        
        chatInput.value = command;
        sendMessage();
        insightsManager.dismissInsight(index);
    }
}

function dismissInsight(index) {
    insightsManager.dismissInsight(index);
}

// Trading action functions
function executeTradeSignal(signalId) {
    if (confirm('Execute this trade signal?')) {
        chatInput.value = `/execute ${signalId}`;
        sendMessage();
    }
}

function analyzeSignal(signalId) {
    chatInput.value = `/analyze signal ${signalId}`;
    sendMessage();
}

function requestDetailedAnalysis(symbol) {
    chatInput.value = `/analyze ${symbol} detailed`;
    sendMessage();
}

// Voice functions
function toggleVoice() {
    if (voiceInterface.isListening) {
        voiceInterface.stopListening();
    } else {
        voiceInterface.startListening();
    }
}

// Utility functions
function getSessionId() {
    let sessionId = localStorage.getItem('chat_session_id');
    if (!sessionId) {
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('chat_session_id', sessionId);
    }
    return sessionId;
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    
    // Enhanced input event listeners
    if (chatInput) {
        chatInput.addEventListener('keydown', handleEnter);
        chatInput.addEventListener('input', handleInput);
        chatInput.addEventListener('focus', () => suggestionManager.hide());
    }
    
    // Voice button
    if (voiceButton) {
        voiceButton.addEventListener('click', toggleVoice);
    }
    
    // Click outside to hide suggestions
    document.addEventListener('click', (event) => {
        if (!event.target.closest('.chat-input-container')) {
            suggestionManager.hide();
        }
    });
});

console.log('Enhanced Trading Bot Chat Interface Loaded with Advanced AI Features.');