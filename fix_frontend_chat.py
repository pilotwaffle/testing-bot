"""
File: fix_frontend_chat.py
Location: E:\Trade Chat Bot\G Trading Bot\fix_frontend_chat.py

Frontend JavaScript Fix Script
Fixes chat.js to properly handle the backend responses
"""

import shutil
from datetime import datetime
from pathlib import Path

def backup_js_files():
    """Backup JavaScript files"""
    js_files = ["static/js/chat.js", "static/js/dashboard.js"]
    backups = []
    
    for js_file in js_files:
        if Path(js_file).exists():
            backup_name = f"{js_file}.backup_frontend_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(js_file, backup_name)
            backups.append(backup_name)
            print(f"📁 Backup created: {backup_name}")
    
    return backups

def create_fixed_chat_js():
    """Create fixed chat.js that properly handles responses"""
    print("🔧 Creating Fixed chat.js")
    print("=" * 50)
    
    fixed_chat_js = '''// static/js/chat.js - FIXED VERSION
// Handles backend responses properly, no more "Error: undefined"

let ws = null;
const chatMessagesContainer = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');

// Enhanced message handling with proper error handling
class MessageHandler {
    constructor() {
        this.messageTypes = {
            'text': this.renderTextMessage,
            'trade_signal': this.renderTradeSignal,
            'analysis': this.renderAnalysis,
            'command_result': this.renderCommandResult,
            'notification': this.renderNotification,
            'error': this.renderError
        };
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
        return this.renderTextMessage(sender, content, metadata);
    }
    
    renderAnalysis(sender, content, metadata = {}) {
        return this.renderTextMessage(sender, content, metadata);
    }
    
    renderCommandResult(sender, content, metadata = {}) {
        return this.renderTextMessage(sender, content, metadata);
    }
    
    renderNotification(sender, content, metadata = {}) {
        return this.renderTextMessage(sender, content, metadata);
    }
    
    renderError(sender, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message error-message';
        messageDiv.innerHTML = `
            <div class="message-header">
                <strong>Error:</strong>
                <span class="timestamp">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content" style="color: #ff6b6b;">${content}</div>
        `;
        return messageDiv;
    }
    
    formatContent(content) {
        if (!content || content === 'undefined' || content === undefined) {
            return 'Sorry, I encountered an issue processing your request. Please try again.';
        }
        
        // Convert string content to safe HTML
        return String(content)
            .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
            .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\\n/g, '<br>')
            .replace(/(\\$\\d+(?:\\.\\d{2})?)/g, '<span class="price">$1</span>')
            .replace(/([+-]\\d+(?:\\.\\d+)?%)/g, '<span class="percentage">$1</span>');
    }
}

const messageHandler = new MessageHandler();

// Enhanced WebSocket connection with better error handling
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log('Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        addMessage('System', 'Connected to enhanced trading bot AI.', 'notification');
    };

    ws.onmessage = (event) => {
        console.log('📨 WebSocket message received:', event.data);
        
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (e) {
            console.error('❌ WebSocket message parsing error:', e);
            // Fallback for plain text messages
            addMessage('Bot', event.data || 'Received message but could not parse it.');
        }
    };

    ws.onclose = (event) => {
        console.warn('🔌 WebSocket disconnected. Code:', event.code, 'Reason:', event.reason);
        addMessage('System', 'Disconnected from bot. Reconnecting...', 'notification');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        addMessage('System', 'WebSocket connection error. Using HTTP fallback.', 'error');
    };
}

function handleWebSocketMessage(data) {
    console.log('📊 Processing message:', data);
    
    try {
        switch(data.type) {
            case 'chat_response':
                handleChatResponse(data);
                break;
            case 'trade_signal':
                addMessage('AI Signal', data.content || data.message || 'Trade signal received', 'trade_signal');
                break;
            case 'ai_analysis':
                addMessage('AI Analysis', data.analysis || data.message || 'Analysis completed', 'analysis');
                break;
            case 'bot_status':
                handleStatusUpdate(data);
                break;
            case 'notification':
                addMessage('System', data.message || 'Notification received', 'notification');
                break;
            case 'error':
                addMessage('Error', data.message || 'An error occurred', 'error');
                break;
            case 'pong':
                console.log('🏓 WebSocket ping/pong successful');
                break;
            default:
                console.log('🤔 Unknown message type:', data.type);
                addMessage('System', `Received message: ${data.type}`, 'notification');
        }
    } catch (error) {
        console.error('❌ Error handling WebSocket message:', error);
        addMessage('Error', 'Failed to process server message', 'error');
    }
}

function handleChatResponse(data) {
    console.log('💬 Processing chat response:', data);
    
    try {
        // Extract response content with fallbacks
        let responseContent = data.response || data.message || data.content;
        
        // Handle undefined or null responses
        if (!responseContent || responseContent === 'undefined') {
            console.warn('⚠️ Undefined response received:', data);
            responseContent = 'Sorry, I encountered an issue. Please try asking again.';
        }
        
        // Get message type
        const messageType = data.message_type || 'text';
        
        // Add message to chat
        addMessage('Bot', responseContent, messageType);
        
        // Handle additional features
        if (data.proactive_insights && data.proactive_insights.length > 0) {
            console.log('💡 Proactive insights:', data.proactive_insights);
            // Could add insight display here
        }
        
        if (data.suggestions && data.suggestions.length > 0) {
            console.log('💭 Suggestions:', data.suggestions);
            // Could add suggestion display here
        }
        
    } catch (error) {
        console.error('❌ Error handling chat response:', error);
        addMessage('Error', 'Failed to display chat response', 'error');
    }
}

function handleStatusUpdate(data) {
    const statusMessage = `Status: ${data.status || 'Unknown'}`;
    addMessage('Bot Status', statusMessage, 'notification');
}

// Enhanced message sending with better error handling
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    console.log('📤 Sending message:', message);
    
    addMessage('You', message);
    chatInput.value = '';

    const startTime = performance.now();

    if (ws && ws.readyState === WebSocket.OPEN) {
        // Send via WebSocket
        try {
            const messageData = { 
                type: 'chat', 
                message: message,
                timestamp: new Date().toISOString(),
                session_id: getSessionId()
            };
            
            ws.send(JSON.stringify(messageData));
            console.log('✅ Message sent via WebSocket');
            
        } catch (error) {
            console.error('❌ WebSocket send error:', error);
            // Fallback to HTTP
            sendMessageHTTP(message, startTime);
        }
    } else {
        console.log('🌐 WebSocket not available, using HTTP');
        sendMessageHTTP(message, startTime);
    }
}

async function sendMessageHTTP(message, startTime) {
    try {
        console.log('📡 Sending via HTTP API');
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: message,
                timestamp: new Date().toISOString(),
                session_id: getSessionId()
            })
        });
        
        console.log('📡 HTTP response status:', response.status);
        
        if (response.ok) {
            try {
                const data = await response.json();
                console.log('📡 HTTP response data:', data);
                
                // Calculate response time
                const endTime = performance.now();
                data.response_time = (endTime - startTime) / 1000;
                
                // Handle as chat response
                handleChatResponse(data);
                
            } catch (jsonError) {
                console.error('❌ JSON parse error:', jsonError);
                const textResponse = await response.text();
                console.log('📄 Raw response:', textResponse);
                addMessage('Error', 'Received invalid response from server', 'error');
            }
        } else {
            console.error('❌ HTTP error:', response.status);
            const errorText = await response.text();
            console.log('📄 Error response:', errorText);
            addMessage('Error', `Server error: ${response.status}`, 'error');
        }
    } catch (error) {
        console.error('❌ HTTP request error:', error);
        addMessage('Error', `Connection error: ${error.message}`, 'error');
    }
}

// Enhanced message adding with proper error handling
function addMessage(sender, content, messageType = 'text', metadata = {}) {
    if (!chatMessagesContainer) {
        console.error('❌ Chat messages container not found');
        return;
    }
    
    try {
        // Ensure content is not undefined
        if (!content || content === 'undefined') {
            content = 'Message content unavailable';
        }
        
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
        
        console.log('✅ Message added:', sender, content.substring(0, 50));
        
    } catch (error) {
        console.error('❌ Error adding message:', error);
        
        // Fallback simple message
        const simpleDiv = document.createElement('div');
        simpleDiv.className = 'message bot-message';
        simpleDiv.innerHTML = `<strong>${sender}:</strong> ${content}`;
        chatMessagesContainer.appendChild(simpleDiv);
    }
}

// Enhanced input handling
function handleEnter(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
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
    console.log('🚀 Chat interface loading...');
    
    connectWebSocket();
    
    // Enhanced input event listeners
    if (chatInput) {
        chatInput.addEventListener('keydown', handleEnter);
        chatInput.addEventListener('focus', () => {
            console.log('💭 Chat input focused');
        });
    } else {
        console.error('❌ Chat input element not found');
    }
    
    console.log('✅ Enhanced Trading Bot Chat Interface Loaded');
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('🚨 Global JavaScript error:', event.error);
});

// Export functions for debugging
window.debugChat = {
    sendMessage,
    addMessage,
    connectWebSocket,
    getSessionId
};
'''
    
    try:
        with open("static/js/chat.js", 'w', encoding='utf-8') as f:
            f.write(fixed_chat_js)
        print("✅ Fixed chat.js created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating fixed chat.js: {e}")
        return False

def create_simple_test_page():
    """Create a simple test page to verify chat is working"""
    print("\n🧪 Creating Simple Test Page")
    print("=" * 50)
    
    test_page = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Test Page</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chat-container { max-width: 600px; margin: 0 auto; }
        .chat-messages { 
            height: 400px; 
            border: 1px solid #ccc; 
            overflow-y: auto; 
            padding: 10px; 
            margin-bottom: 10px;
            background: #f9f9f9;
        }
        .message { 
            margin: 10px 0; 
            padding: 8px 12px;
            border-radius: 8px;
        }
        .user-message { 
            background: #007bff; 
            color: white; 
            text-align: right;
        }
        .bot-message { 
            background: #28a745; 
            color: white; 
        }
        .error-message {
            background: #dc3545;
            color: white;
        }
        .input-container { 
            display: flex; 
            gap: 10px;
        }
        .chat-input { 
            flex: 1; 
            padding: 10px; 
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        .send-button { 
            padding: 10px 20px; 
            background: #007bff; 
            color: white; 
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .send-button:hover { background: #0056b3; }
        .status { margin: 10px 0; padding: 10px; background: #e9ecef; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>🤖 Trading Bot Chat Test</h1>
        <div class="status">
            <strong>Status:</strong> <span id="status">Initializing...</span>
        </div>
        
        <div class="chat-messages" id="chat-messages">
            <div class="message bot-message">
                <strong>Bot:</strong> Hello! I'm your trading assistant. Send me a message to test the chat system.
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="chat-input" class="chat-input"
                   placeholder="Type your message..." onkeypress="handleEnter(event)" autofocus>
            <button class="send-button" onclick="sendMessage()">Send</button>
        </div>
        
        <div class="status">
            <button onclick="testAPI()">Test API</button>
            <button onclick="testWebSocket()">Test WebSocket</button>
            <button onclick="clearChat()">Clear Chat</button>
        </div>
    </div>

    <script>
        // Simple test functions
        function updateStatus(message) {
            document.getElementById('status').textContent = message;
        }
        
        function clearChat() {
            const container = document.getElementById('chat-messages');
            container.innerHTML = '<div class="message bot-message"><strong>Bot:</strong> Chat cleared. Ready for testing!</div>';
        }
        
        async function testAPI() {
            updateStatus('Testing API...');
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: 'API test message' })
                });
                
                const data = await response.json();
                console.log('API Response:', data);
                
                if (data.response) {
                    addSimpleMessage('API Test', data.response);
                    updateStatus('API test successful');
                } else {
                    addSimpleMessage('Error', 'API returned no response');
                    updateStatus('API test failed');
                }
            } catch (error) {
                addSimpleMessage('Error', 'API test failed: ' + error.message);
                updateStatus('API test failed');
            }
        }
        
        function testWebSocket() {
            updateStatus('Testing WebSocket...');
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                updateStatus('WebSocket connected');
                ws.send(JSON.stringify({
                    type: 'chat',
                    message: 'WebSocket test message'
                }));
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.response) {
                    addSimpleMessage('WebSocket Test', data.response);
                    updateStatus('WebSocket test successful');
                }
                ws.close();
            };
            
            ws.onerror = () => {
                addSimpleMessage('Error', 'WebSocket test failed');
                updateStatus('WebSocket test failed');
            };
        }
        
        function addSimpleMessage(sender, content) {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = sender === 'You' ? 'message user-message' : 'message bot-message';
            div.innerHTML = `<strong>${sender}:</strong> ${content}`;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        updateStatus('Ready for testing');
    </script>
    <script src="/static/js/chat.js"></script>
</body>
</html>'''
    
    try:
        with open("templates/chat_test.html", 'w', encoding='utf-8') as f:
            f.write(test_page)
        print("✅ Test page created: templates/chat_test.html")
        return True
    except Exception as e:
        print(f"❌ Error creating test page: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Frontend JavaScript Fix")
    print("=" * 60)
    
    print("🎯 Fixing 'Error: undefined' in chat interface")
    print("📊 Backend tests showed all components working perfectly")
    print("🔧 Issue is in frontend JavaScript handling responses")
    print()
    
    # Step 1: Backup JavaScript files
    backup_js_files()
    
    # Step 2: Create fixed chat.js
    if create_fixed_chat_js():
        print("✅ chat.js fixed with proper error handling")
    else:
        print("❌ Failed to fix chat.js")
        return
    
    # Step 3: Create test page
    if create_simple_test_page():
        print("✅ Test page created")
    
    print("\n🎉 FRONTEND FIX COMPLETE!")
    print("=" * 60)
    
    print("🔄 Your server will auto-reload the JavaScript files")
    print()
    print("🧪 Test the fix:")
    print("1. Visit: http://localhost:8000/chat")
    print("2. Type: 'Hello! Test message.'")
    print("3. Expected: Smart Gemini AI response (no more 'Error: undefined')")
    print()
    print("🧪 Alternative test page:")
    print("Visit: http://localhost:8000/chat_test.html")
    print("(Simple test interface to debug if needed)")
    print()
    print("📊 What should happen:")
    print("✅ You type a message")
    print("✅ Get intelligent trading advice from Gemini AI")
    print("✅ No more 'Error: undefined'")
    print("✅ Proper chat conversation flow")
    print()
    print("🔍 If still issues:")
    print("• Open browser console (F12)")
    print("• Look for any red error messages")
    print("• Check Network tab for failed requests")

if __name__ == "__main__":
    main()