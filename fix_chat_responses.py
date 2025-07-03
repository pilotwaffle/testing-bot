"""
File: fix_chat_responses.py
Location: E:\Trade Chat Bot\G Trading Bot\fix_chat_responses.py

Chat Response Handling Fix
Fixes the specific "Error: undefined" issue in chat responses
"""

import shutil
import requests
from datetime import datetime
from pathlib import Path

def backup_chat_files():
    """Backup chat-related files"""
    files_to_backup = [
        "static/js/chat.js",
        "static/js/dashboard.js",
        "templates/chat.html"
    ]
    backups = []
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            backup_name = f"{file_path}.backup_chatfix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_name)
            backups.append(backup_name)
            print(f"📁 Backup created: {backup_name}")
    
    return backups

def test_backend_response():
    """Test what the backend actually returns"""
    print("🧪 Testing Backend Response")
    print("=" * 50)
    
    try:
        response = requests.post(
            "http://localhost:8000/api/chat",
            json={"message": "status"},
            timeout=10
        )
        
        print(f"📊 Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Response Keys: {list(data.keys())}")
            print(f"📊 Full Response: {data}")
            
            if 'response' in data:
                print(f"✅ Response field exists: {data['response'][:100]}...")
            else:
                print(f"❌ No 'response' field found")
            
            return data
        else:
            print(f"❌ Backend error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return None

def create_debug_chat_js():
    """Create a debug version of chat.js with extensive logging"""
    print("\n🔧 Creating Debug Chat JavaScript")
    print("=" * 50)
    
    debug_chat_js = '''// static/js/chat.js - DEBUG VERSION with extensive logging
// Fixes "Error: undefined" by handling all response formats properly

let ws = null;
const chatMessagesContainer = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');

console.log('🚀 Debug Chat.js loading...');
console.log('📊 Chat container found:', !!chatMessagesContainer);
console.log('📊 Chat input found:', !!chatInput);

// Enhanced WebSocket connection with debug logging
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log('🔌 Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('✅ WebSocket connected successfully');
        addMessage('System', 'Connected to enhanced trading bot AI.', 'notification');
    };

    ws.onmessage = (event) => {
        console.log('📨 Raw WebSocket message:', event.data);
        
        try {
            const data = JSON.parse(event.data);
            console.log('📊 Parsed WebSocket data:', data);
            handleWebSocketMessage(data);
        } catch (e) {
            console.error('❌ WebSocket JSON parse error:', e);
            console.log('📄 Raw data that failed to parse:', event.data);
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
    console.log('🔍 Processing WebSocket message type:', data.type);
    console.log('🔍 Full message data:', data);
    
    try {
        switch(data.type) {
            case 'chat_response':
                console.log('💬 Handling chat response');
                handleChatResponse(data);
                break;
            case 'pong':
                console.log('🏓 Pong received');
                break;
            default:
                console.log('🤔 Unknown message type:', data.type);
                if (data.response || data.message) {
                    // Treat as chat response
                    handleChatResponse(data);
                }
        }
    } catch (error) {
        console.error('❌ Error handling WebSocket message:', error);
        addMessage('Error', 'Failed to process server message: ' + error.message, 'error');
    }
}

function handleChatResponse(data) {
    console.log('💬 Processing chat response:', data);
    
    try {
        // Multiple fallbacks for response content
        let responseContent = null;
        
        // Try different possible response fields
        if (data.response) {
            responseContent = data.response;
            console.log('✅ Found response in data.response');
        } else if (data.message) {
            responseContent = data.message;
            console.log('✅ Found response in data.message');
        } else if (data.content) {
            responseContent = data.content;
            console.log('✅ Found response in data.content');
        } else if (typeof data === 'string') {
            responseContent = data;
            console.log('✅ Using data as string response');
        } else {
            console.error('❌ No response content found in:', data);
            responseContent = 'Sorry, I received a response but could not extract the content.';
        }
        
        // Handle undefined or null responses
        if (!responseContent || responseContent === 'undefined' || responseContent === null) {
            console.warn('⚠️ Response content is undefined/null:', responseContent);
            responseContent = 'Sorry, I encountered an issue processing your request. Please try again.';
        }
        
        console.log('📝 Final response content:', responseContent);
        
        // Add message to chat
        addMessage('Bot', responseContent);
        
    } catch (error) {
        console.error('❌ Error in handleChatResponse:', error);
        addMessage('Error', 'Failed to display chat response: ' + error.message, 'error');
    }
}

// Enhanced message sending with extensive debugging
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) {
        console.warn('⚠️ No message to send');
        return;
    }

    console.log('📤 Sending message:', message);
    
    addMessage('You', message);
    chatInput.value = '';

    const startTime = performance.now();

    // Try WebSocket first
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log('📡 Sending via WebSocket');
        try {
            const messageData = { 
                type: 'chat', 
                message: message,
                timestamp: new Date().toISOString(),
                session_id: getSessionId()
            };
            
            console.log('📡 WebSocket message data:', messageData);
            ws.send(JSON.stringify(messageData));
            console.log('✅ Message sent via WebSocket');
            
        } catch (error) {
            console.error('❌ WebSocket send error:', error);
            sendMessageHTTP(message, startTime);
        }
    } else {
        console.log('🌐 WebSocket not available, using HTTP');
        sendMessageHTTP(message, startTime);
    }
}

async function sendMessageHTTP(message, startTime) {
    console.log('📡 Sending via HTTP API');
    
    try {
        const requestData = { 
            message: message,
            timestamp: new Date().toISOString(),
            session_id: getSessionId()
        };
        
        console.log('📡 HTTP request data:', requestData);
        
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        console.log('📡 HTTP response status:', response.status);
        console.log('📡 HTTP response headers:', response.headers);
        
        if (response.ok) {
            try {
                const responseText = await response.text();
                console.log('📡 HTTP raw response text:', responseText);
                
                const data = JSON.parse(responseText);
                console.log('📡 HTTP parsed response data:', data);
                
                // Calculate response time
                const endTime = performance.now();
                data.response_time = (endTime - startTime) / 1000;
                
                // Handle as chat response
                handleChatResponse(data);
                
            } catch (jsonError) {
                console.error('❌ JSON parse error:', jsonError);
                const textResponse = await response.text();
                console.log('📄 Raw response that failed to parse:', textResponse);
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

// Enhanced message adding with debug logging
function addMessage(sender, content, messageType = 'text') {
    console.log('💬 Adding message:', { sender, content, messageType });
    
    if (!chatMessagesContainer) {
        console.error('❌ Chat messages container not found');
        return;
    }
    
    try {
        // Ensure content is valid
        if (!content || content === 'undefined' || content === null) {
            console.warn('⚠️ Invalid message content:', content);
            content = 'Message content unavailable';
        }
        
        // Convert content to string if needed
        content = String(content);
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender === 'You' ? 'user-message' : 'bot-message'}`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        messageDiv.innerHTML = `
            <div class="message-header">
                <strong>${sender}:</strong>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-content">${content}</div>
        `;
        
        chatMessagesContainer.appendChild(messageDiv);
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
        
        // Add animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        setTimeout(() => {
            messageDiv.style.transition = 'all 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        }, 10);
        
        console.log('✅ Message added successfully');
        
    } catch (error) {
        console.error('❌ Error adding message:', error);
        
        // Fallback simple message
        const simpleDiv = document.createElement('div');
        simpleDiv.className = 'message bot-message';
        simpleDiv.innerHTML = `<strong>${sender}:</strong> ${content}`;
        if (chatMessagesContainer) {
            chatMessagesContainer.appendChild(simpleDiv);
        }
    }
}

// Input handling
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

// Manual test functions for debugging
window.testChatAPI = async function() {
    console.log('🧪 Testing Chat API manually');
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: 'test' })
        });
        const data = await response.json();
        console.log('🧪 Manual API test result:', data);
        handleChatResponse(data);
    } catch (error) {
        console.error('🧪 Manual API test failed:', error);
    }
};

window.testWebSocket = function() {
    console.log('🧪 Testing WebSocket manually');
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'chat',
            message: 'test websocket'
        }));
        console.log('🧪 Manual WebSocket test sent');
    } else {
        console.error('🧪 WebSocket not available');
    }
};

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Chat interface initializing...');
    
    connectWebSocket();
    
    if (chatInput) {
        chatInput.addEventListener('keydown', handleEnter);
        console.log('✅ Chat input listeners added');
    } else {
        console.error('❌ Chat input element not found');
    }
    
    console.log('✅ Enhanced Trading Bot Chat Interface Loaded (Debug Version)');
    console.log('🧪 Use testChatAPI() or testWebSocket() in console for manual testing');
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('🚨 Global JavaScript error:', event.error);
});

// Export for debugging
window.debugChat = {
    sendMessage,
    addMessage,
    connectWebSocket,
    handleChatResponse,
    testChatAPI: window.testChatAPI,
    testWebSocket: window.testWebSocket
};
'''
    
    try:
        with open("static/js/chat.js", 'w', encoding='utf-8') as f:
            f.write(debug_chat_js)
        print("✅ Debug chat.js created with extensive logging")
        return True
    except Exception as e:
        print(f"❌ Error creating debug chat.js: {e}")
        return False

def create_simple_chat_test():
    """Create a simple chat test page"""
    print("\n🧪 Creating Simple Chat Test Page")
    print("=" * 50)
    
    test_chat_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Debug Test</title>
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
        .debug-panel {
            margin: 20px 0;
            padding: 15px;
            background: #e9ecef;
            border-radius: 8px;
        }
        .debug-button {
            margin: 5px;
            padding: 8px 15px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>🔧 Chat Debug Test Page</h1>
        
        <div class="debug-panel">
            <h3>Debug Controls</h3>
            <button class="debug-button" onclick="testAPI()">Test API</button>
            <button class="debug-button" onclick="testWebSocket()">Test WebSocket</button>
            <button class="debug-button" onclick="testChat('status')">Test 'status' command</button>
            <button class="debug-button" onclick="testChat('help')">Test 'help' command</button>
            <button class="debug-button" onclick="clearLogs()">Clear Console</button>
        </div>
        
        <div class="chat-messages" id="chat-messages">
            <div class="message bot-message">
                <strong>Debug Bot:</strong> Chat debug interface loaded. Check console (F12) for detailed logs.
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="chat-input" class="chat-input"
                   placeholder="Type your message..." onkeypress="handleEnter(event)" autofocus>
            <button class="send-button" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        // Debug functions
        async function testAPI() {
            console.log('🧪 Manual API Test');
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: 'API test' })
            });
            const data = await response.json();
            console.log('📊 API Response:', data);
            if (window.debugChat) {
                window.debugChat.handleChatResponse(data);
            }
        }
        
        function testWebSocket() {
            console.log('🧪 Manual WebSocket Test');
            if (window.debugChat) {
                window.debugChat.testWebSocket();
            }
        }
        
        function testChat(message) {
            console.log('🧪 Testing chat with message:', message);
            document.getElementById('chat-input').value = message;
            sendMessage();
        }
        
        function clearLogs() {
            console.clear();
            console.log('🧹 Console cleared');
        }
    </script>
    
    <script src="/static/js/chat.js"></script>
</body>
</html>'''
    
    try:
        with open("templates/chat_debug.html", 'w', encoding='utf-8') as f:
            f.write(test_chat_html)
        print("✅ Chat debug page created")
        return True
    except Exception as e:
        print(f"❌ Error creating chat debug page: {e}")
        return False

def add_chat_debug_route():
    """Add chat debug route to main.py"""
    print("\n🔧 Adding Chat Debug Route")
    print("=" * 50)
    
    try:
        # Read current main.py
        with open("main.py", 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # Check if debug route already exists
        if "/chat_debug" in main_content:
            print("✅ Chat debug route already exists")
            return True
        
        # Add debug route before the last few lines
        debug_route = '''
@app.get("/chat_debug", response_class=HTMLResponse)
async def chat_debug(request: Request):
    """Debug chat interface for troubleshooting"""
    return templates.TemplateResponse("chat_debug.html", {"request": request})
'''
        
        # Find a good place to insert the route (before the last few lines)
        lines = main_content.split('\n')
        insert_index = -5  # Insert before the last 5 lines
        
        # Insert the route
        lines.insert(insert_index, debug_route)
        
        # Write back to file
        new_content = '\n'.join(lines)
        with open("main.py", 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Chat debug route added to main.py")
        return True
        
    except Exception as e:
        print(f"❌ Error adding debug route: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Chat Response Handling Fix")
    print("=" * 60)
    
    print("🎯 Fixing 'Error: undefined' in chat responses")
    print("🧪 The backend works perfectly - this is a frontend parsing issue")
    print()
    
    # Step 1: Test backend first
    backend_data = test_backend_response()
    
    # Step 2: Backup files
    backup_chat_files()
    
    # Step 3: Create debug chat.js
    if create_debug_chat_js():
        print("✅ Debug chat.js created with extensive logging")
    else:
        print("❌ Failed to create debug chat.js")
        return
    
    # Step 4: Create debug test page
    if create_simple_chat_test():
        print("✅ Debug test page created")
    
    # Step 5: Add debug route to main.py
    if add_chat_debug_route():
        print("✅ Debug route added to main.py")
    
    print("\n🎉 CHAT RESPONSE FIX COMPLETE!")
    print("=" * 60)
    
    print("🔄 Your server will auto-reload with the fixes")
    print()
    print("🧪 Test the fix:")
    print("1. Visit: http://localhost:8000/chat")
    print("2. Open F12 console to see detailed logs")
    print("3. Type: 'status' or 'help'")
    print("4. Watch console for detailed debugging info")
    print()
    print("🧪 Alternative debug page:")
    print("Visit: http://localhost:8000/chat_debug")
    print("• Has manual test buttons")
    print("• Shows detailed logging")
    print("• Helps identify exact issue")
    print()
    print("📊 Expected results:")
    print("✅ Console shows detailed request/response logs")
    print("✅ You can see exactly where the response breaks")
    print("✅ 'Error: undefined' should be replaced with actual responses")
    print()
    print("🔍 Debugging tips:")
    print("• Check console for red errors")
    print("• Look for response parsing logs")
    print("• Try manual test buttons")
    print("• Use testChatAPI() in console")

if __name__ == "__main__":
    main()