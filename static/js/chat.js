// static/js/chat.js
let ws = null;
const chatMessagesContainer = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        addMessage('System', 'Connected to bot via WebSocket.');
        // Optionally send a 'status' message on connect to get initial state
        // ws.send(JSON.stringify({ type: 'status' }));
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'chat_response') {
                addMessage('Bot', data.message);
            } else if (data.type === 'ai_analysis') {
                addMessage('AI Analysis', data.analysis);
            } else if (data.type === 'bot_status') {
                // Handle live status updates in chat, e.g., "Bot: Trading engine is now RUNNING."
                addMessage('Bot Status', `Current status: ${data.status}. Total Value: $${data.metrics.total_value.toFixed(2)}`);
            } else if (data.type === 'error') {
                addMessage('Error', data.message);
            } else {
                console.log('Unhandled WS message type:', data.type, data);
                addMessage('System', `Received unhandled message (type: ${data.type}).`);
            }
        } catch (e) {
            // Fallback for plain text messages not in JSON format
            addMessage('Bot', event.data);
            console.error('WebSocket message parsing error:', e, event.data);
        }
    };

    ws.onclose = () => {
        console.warn('WebSocket disconnected. Attempting to reconnect in 3s...');
        addMessage('System', 'Disconnected from bot. Reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addMessage('System', `WebSocket error: ${error.message || 'Check console'}`);
    };
}

// Initial WebSocket connection
document.addEventListener('DOMContentLoaded', connectWebSocket);

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addMessage('You', message);
    chatInput.value = '';

    if (ws && ws.readyState === WebSocket.OPEN) {
        // Send as a structured JSON message for better handling server-side
        ws.send(JSON.stringify({ type: 'chat', message: message }));
    } else {
        // Fallback to HTTP POST if WebSocket is not open
        console.warn("WebSocket not open, using HTTP POST fallback for chat.");
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }, // Ensure JSON header
                body: JSON.stringify({ message: message }) // Send JSON body
            });
            const data = await response.json();
            if (data.success) {
                addMessage('Bot', data.response);
            } else {
                addMessage('Error', data.detail || data.error || 'Unknown error.');
            }
        } catch (error) {
            addMessage('Error', `Client-side fetch error: ${error.message}`);
        }
    }
}

function addMessage(sender, message) {
    const messageDiv = document.createElement('div');
    // Apply specific classes for user/bot messages for styling
    messageDiv.className = `message ${sender === 'You' ? 'user-message' : 'bot-message'}`;
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatMessagesContainer.appendChild(messageDiv);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight; // Auto-scroll to bottom
}

function handleEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

console.log('Trading Bot Chat Page Logic Loaded.');