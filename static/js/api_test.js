// static/js/api_test.js
async function testEndpoint(url, method, data, responseId) {
    const responseDiv = document.getElementById(responseId);
    responseDiv.textContent = 'Testing...';
    responseDiv.className = 'response loading'; // Set loading class

    try {
        const options = { method: method };
        if (data) {
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);
        // Check for non-2xx response status for clearer error handling
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({})); // Try to parse error body
            responseDiv.textContent = `Error ${response.status}: ${JSON.stringify(errorData, null, 2)}`;
            responseDiv.className = 'response error';
            return;
        }

        const result = await response.json();
        responseDiv.textContent = JSON.stringify(result, null, 2);
        responseDiv.className = 'response success'; // Set success class
    } catch (error) {
        responseDiv.textContent = `Client-side Error: ${error.message || error}`;
        responseDiv.className = 'response error'; // Set error class
    }
}

async function testChat() {
    const messageInput = document.getElementById('chat-message');
    const message = messageInput.value;
    if (!message) return;
    await testEndpoint('/api/chat', 'POST', { message: message }, 'chat-response');
}

async function testCustomNotification() {
    const title = document.getElementById('notif-title').value || 'Test Title';
    const message = document.getElementById('notif-message').value || 'This is a test notification from the API test page.';
    const payload = { title: title, message: message, priority: 'medium' };
    await testEndpoint('/api/notifications/send', 'POST', payload, 'custom-notif-response');
}

async function testAddStrategy() {
    const strategyName = document.getElementById('strategy-name').value || `test_strategy_${Date.now() % 1000}`; // Generate unique name
    const payload = { name: strategyName, config: { enabled: true, version: "1.0" } };
    await testEndpoint('/api/strategies/add', 'POST', payload, 'add-strategy-response');
}

async function testRemoveStrategy() {
    const strategyName = document.getElementById('remove-strategy-name').value;
    if (!strategyName) {
        document.getElementById('remove-strategy-response').textContent = "Please enter a strategy name to remove.";
        document.getElementById('remove-strategy-response').className = 'response error';
        return;
    }
    const payload = { name: strategyName };
    await testEndpoint('/api/strategies/remove', 'POST', payload, 'remove-strategy-response');
}

async function testAIQuestion() {
    const questionInput = document.getElementById('ai-question');
    const question = questionInput.value;
    if (!question) return;

    const formData = new FormData();
    formData.append('question', question);

    const responseDiv = document.getElementById('ai-question-response');
    responseDiv.textContent = 'AI is thinking...';
    responseDiv.className = 'response loading';

    try {
        const response = await fetch('/api/ai/ask', {
            method: 'POST',
            body: formData // No 'Content-Type' header needed for FormData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            responseDiv.textContent = `Error ${response.status}: ${JSON.stringify(errorData, null, 2)}`;
            responseDiv.className = 'response error';
            return;
        }

        const result = await response.json();
        responseDiv.textContent = JSON.stringify(result, null, 2);
        responseDiv.className = 'response success';
    } catch (error) {
        responseDiv.textContent = `Client-side Error: ${error.message}`;
        responseDiv.className = 'response error';
    }
}

async function testMLTrain() {
    const modelType = document.getElementById('model-type').value;

    const formData = new FormData();
    formData.append('model_type', modelType);
    formData.append('symbol', 'BTC/USDT'); // Default symbol for testing

    const responseDiv = document.getElementById('ml-response');
    responseDiv.textContent = `Training ${modelType}... This may take a moment.`;
    responseDiv.className = 'response loading';

    try {
        const response = await fetch('/api/ml/train', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            responseDiv.textContent = `Error ${response.status}: ${JSON.stringify(errorData, null, 2)}`;
            responseDiv.className = 'response error';
            return;
        }

        const result = await response.json();
        responseDiv.textContent = JSON.stringify(result, null, 2);
        responseDiv.className = 'response success';
    } catch (error) {
        responseDiv.textContent = `Client-side Error: ${error.message}`;
        responseDiv.className = 'response error';
    }
}