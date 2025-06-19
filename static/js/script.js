// static/js/script.js

console.log("script.js loaded and executing!"); // DEBUG: Confirm script loads

// API Base URL (adjust if your FastAPI server is elsewhere)
const API_BASE_URL = window.location.origin;

// --- DOM Elements ---
const botStatusElem = document.getElementById('bot-status');
const startBotBtn = document.getElementById('start-bot-btn');
const stopBotBtn = document.getElementById('stop-bot-btn');

const totalValueElem = document.getElementById('total-value');
const usdtBalanceElem = document.getElementById('usdt-balance');
const openPositionsCountElem = document.getElementById('open-positions-count');

const marketDataContainer = document.getElementById('market-data-container');
const activeStrategiesContainer = document.getElementById('active-strategies-container');
const performanceMetricsContainer = document.getElementById('performance-metrics-container');
const mlModelsContainer = document.getElementById('ml-models-container');
const trainModelsBtn = document.getElementById('train-models-btn');

const chatMessagesElem = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendChatBtn = document.getElementById('send-chat-btn');

// --- New Strategy Add Form Elements ---
const strategyTypeSelect = document.getElementById('strategy-type');
const strategyIdInput = document.getElementById('strategy-id'); 
const strategyConfigInputsContainer = document.getElementById('strategy-config-inputs');
const addStrategyBtn = document.getElementById('add-strategy-btn'); 

// --- Global Data Store ---
let availableStrategyTypes = []; // To store fetched strategy data with config templates
let mlModelTypes = ["neural_network", "lorentzian", "risk_assessment"]; // Hardcoded for now, could fetch from backend
let availableTimeframes = ["1h", "1d", "4h", "15m", "30m", "1m"]; 

// --- Helper Functions ---

async function fetchData(endpoint) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`);
        if (!response.ok) {
            const errorText = await response.text(); // Get text for more detail
            throw new Error(`HTTP error! status: ${response.status}, detail: ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching from ${endpoint}:`, error);
        return null; 
    }
}

async function postData(endpoint, data = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        // Always attempt to get JSON response for error messages, even if !response.ok
        let jsonResponse = {};
        try {
            jsonResponse = await response.json();
            // console.log("DEBUG PostData JSON Response:", jsonResponse); // Further debug
        } catch (e) {
            console.warn(`PostData: Could not parse JSON response for ${endpoint}. Status: ${response.status}. Error: ${e}`);
            // Fallback to text if JSON parsing fails to avoid losing error info
            jsonResponse = { detail: await response.text() }; 
        }
        
        if (!response.ok) {
            const detailError = jsonResponse.detail || `Unknown error (Status: ${response.status})`;
            throw new Error(`HTTP error ${response.status}: ${detailError}`);
        }
        return jsonResponse;

    } catch (error) {
        console.error(`Error posting to ${endpoint}:`, error);
        alert(`Request to ${endpoint} failed: ${error.message}`); // Show a user-friendly alert
        return null;
    }
}

async function deleteData(endpoint) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'DELETE'
        });
        const jsonResponse = await response.json();
        if (!response.ok) {
            const detailError = jsonResponse.detail || 'Unknown error';
            throw new Error(`HTTP error ${response.status}: ${detailError}`);
        }
        return jsonResponse;
    } catch (error) {
        console.error(`Error deleting from ${endpoint}:`, error);
        alert(`Delete request failed: ${error.message}`);
        return null;
    }
}

function updateBotStatusUI(isRunning) { 
    if (isRunning) {
        botStatusElem.textContent = 'Running';
        botStatusElem.className = 'status running';
        startBotBtn.disabled = true;
        stopBotBtn.disabled = false;
    } else {
        botStatusElem.textContent = 'Stopped';
        botStatusElem.className = 'status stopped';
        startBotBtn.disabled = false;
        stopBotBtn.disabled = true;
    }
}

// --- Dynamic Strategy Form Functions ---

async function populateStrategyTypeSelect() {
    availableStrategyTypes = await fetchData('/api/strategies/available');
    strategyTypeSelect.innerHTML = '<option value="">-- Select a strategy type --</option>'; // Clear and add default
    if (availableStrategyTypes) {
        availableStrategyTypes.forEach(strategy => {
            const option = document.createElement('option');
            option.value = strategy.name;
            option.textContent = `${strategy.name} (${strategy.description})`;
            strategyTypeSelect.appendChild(option);
        });
    }
}

function generateConfigInputs(strategyName) {
    strategyConfigInputsContainer.innerHTML = ''; // Clear previous inputs
    const selectedStrategy = availableStrategyTypes.find(s => s.name === strategyName);

    if (selectedStrategy && selectedStrategy.config_template) {
        const configTemplate = selectedStrategy.config_template;
        
        if (strategyName === "MLStrategy") {
            const defaultSymbol = configTemplate.symbol || "BTC/USD"; // Fallback to BTC/USD
            strategyIdInput.value = `MLStrategy_${defaultSymbol.replace('/','_')}_${Date.now().toString().slice(-5)}`;
        } else {
            strategyIdInput.value = ''; // Clear for other strategies, user must manually enter
        }

        if (Object.keys(configTemplate).length === 0) {
            strategyConfigInputsContainer.innerHTML = '<p>No specific configuration options for this strategy.</p>';
            return;
        }

        for (const key in configTemplate) {
            const defaultValue = configTemplate[key];
            const div = document.createElement('div');
            div.className = 'config-input-group'; 
            
            if (key === "timeframes_config" && typeof defaultValue === 'object' && !Array.isArray(defaultValue)) {
                const timeframesDiv = document.createElement('div');
                timeframesDiv.innerHTML = '<h4>Timeframe Configuration:</h4>';
                timeframesDiv.className = 'timeframes-config-group';

                for (const tfKey in defaultValue) {
                    const tfConfig = defaultValue[tfKey];
                    const tfCard = document.createElement('div');
                    tfCard.className = 'card ml-model'; 
                    tfCard.innerHTML = `<h5>${tfKey} Timeframe:</h5>`;

                    for (const tfPropKey in tfConfig) {
                        const tfPropDefault = tfConfig[tfPropKey];
                        const tfPropLabel = document.createElement('label');
                        tfPropLabel.htmlFor = `config-${key}-${tfKey}-${tfPropKey}`;
                        tfPropLabel.textContent = `${tfPropKey}:`;
                        
                        let tfPropInput = document.createElement('input');
                        tfPropInput.id = `config-${key}-${tfKey}-${tfPropKey}`;
                        tfPropInput.name = `config-${key}-${tfKey}-${tfPropKey}`;
                        tfPropInput.className = 'api-input';

                        tfPropInput.type = typeof tfPropDefault === 'number' ? 'number' : 'text';
                        if(typeof tfPropDefault === 'number') tfPropInput.step = 'any';
                        tfPropInput.value = tfPropDefault;

                        tfCard.appendChild(tfPropLabel);
                        tfCard.appendChild(tfPropInput);
                    }
                    timeframesDiv.appendChild(tfCard);
                }
                div.appendChild(timeframesDiv);
                strategyConfigInputsContainer.appendChild(div);
                continue;
            }

            const label = document.createElement('label');
            label.htmlFor = `config-${key}`;
            label.textContent = `${key}:`;

            let input;

            if (strategyName === "MLStrategy" && key === "model_type") {
                input = document.createElement('select');
                mlModelTypes.forEach(modelType => {
                    const option = document.createElement('option');
                    option.value = modelType;
                    option.textContent = modelType;
                    if (modelType === defaultValue) { 
                        option.selected = true;
                    }
                    input.appendChild(option);
                });
            } else if (key === "timeframe" && Array.isArray(availableTimeframes) && availableTimeframes.length > 0) {
                input = document.createElement('select');
                availableTimeframes.forEach(tf => {
                    const option = document.createElement('option');
                    option.value = tf;
                    option.textContent = tf;
                    if (tf === defaultValue) {
                        option.selected = true;
                    }
                    input.appendChild(option);
                });
            }
            else if (typeof defaultValue === 'boolean') {
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = defaultValue;
            } else if (typeof defaultValue === 'number') {
                input = document.createElement('input');
                input.type = 'number';
                input.value = defaultValue;
                input.step = 'any'; 
            } else if (Array.isArray(defaultValue)) {
                input = document.createElement('input');
                input.type = 'text';
                input.value = defaultValue.join(', '); 
                input.placeholder = 'Comma-separated values';
            }
            else { 
                input = document.createElement('input');
                input.type = 'text';
                input.value = defaultValue;
            }
            input.id = `config-${key}`;
            input.name = `config-${key}`;
            input.className = 'api-input';

            div.appendChild(label);
            div.appendChild(input);
            strategyConfigInputsContainer.appendChild(div);
        }
    } else {
        strategyConfigInputsContainer.innerHTML = '<p>Select a strategy type above to see its configuration options.</p>';
    }
}

function collectConfigValues() {
    const config = {};
    const selectedStrategy = availableStrategyTypes.find(s => s.name === strategyTypeSelect.value);
    const configTemplate = selectedStrategy ? selectedStrategy.config_template : {};

    for (const key in configTemplate) {
        if (key === "timeframes_config" && typeof configTemplate[key] === 'object' && !Array.isArray(configTemplate[key])) {
            const timeframesConfig = {};
            for (const tfKey in configTemplate[key]) {
                const tfConfig = {};
                for (const tfPropKey in configTemplate[key][tfKey]) { 
                    const inputElement = document.getElementById(`config-${key}-${tfKey}-${tfPropKey}`); 
                    if (inputElement) {
                        tfConfig[tfPropKey] = parseInputValue(inputElement); 
                    }
                }
                timeframesConfig[tfKey] = tfConfig;
            }
            config[key] = timeframesConfig;
        } else {
            const inputElement = document.getElementById(`config-${key}`);
            if (inputElement) {
                config[key] = parseInputValue(inputElement);
            }
        }
    }
    return config;
}

function parseInputValue(inputElement) {
    if (inputElement.type === 'checkbox') {
        return inputElement.checked;
    } else if (inputElement.type === 'number') {
        const val = parseFloat(inputElement.value);
        return isNaN(val) ? null : val;
    } else if (inputElement.tagName === 'SELECT') {
        return inputElement.value;
    } else if (inputElement.placeholder === 'Comma-separated values') {
        return inputElement.value.split(',').map(item => item.trim()).filter(item => item !== '');
    } else {
        return inputElement.value;
    }
}


// --- Update Functions for Dashboard Sections --- 

async function updateAccountSummary() { 
    const statusData = await fetchData('/api/status');
    if (statusData) {
        updateBotStatusUI(statusData.running);
        totalValueElem.textContent = `$${statusData.balances.USD ? statusData.balances.USD.toFixed(2) : '0.00'}`;
        if (statusData.balances.USDT !== undefined) {
             usdtBalanceElem.textContent = `$${statusData.balances.USDT.toFixed(2)}`;
        } else {
             usdtBalanceElem.textContent = 'N/A';
        }
        openPositionsCountElem.textContent = Object.keys(statusData.positions).length;
    }
}

async function updateMarketData() { 
    const data = await fetchData('/api/market-data');
    if (data) {
        marketDataContainer.innerHTML = ''; 
        let count = 0;
        for (const symbol in data) {
            if (count >= 3) break; 
            const price = data[symbol].price.toFixed(2);
            const metricDiv = document.createElement('div');
            metricDiv.className = 'metric';
            metricDiv.innerHTML = `<h4>${symbol}</h4><p>$${price}</p>`;
            marketDataContainer.appendChild(metricDiv);
            count++;
        }
        if (Object.keys(data).length === 0) {
            marketDataContainer.innerHTML = '<p>No market data available yet. Please ensure the bot is running and configured with symbols.</p>';
        }
    } else {
        marketDataContainer.innerHTML = '<p>Failed to load market data.</p>';
    }
}

async function updateActiveStrategies() { 
    const strategies = await fetchData('/api/strategies/active');
    activeStrategiesContainer.innerHTML = ''; 
    if (strategies && strategies.length > 0) {
        strategies.forEach(strategy => {
            const strategyDiv = document.createElement('div');
            strategyDiv.innerHTML = `<p><strong>${strategy.id}</strong> (${strategy.type}) - ${strategy.symbol}</p>`;
            const removeBtn = document.createElement('button');
            removeBtn.className = 'button danger';
            removeBtn.textContent = 'Remove';
            removeBtn.onclick = async () => {
                if (confirm(`Are you sure you want to remove strategy ${strategy.id}?`)) {
                    const result = await deleteData(`/api/strategies/remove/${strategy.id}`);
                    if (result && result.status === 'success') {
                        console.log(`Strategy ${strategy.id} removed.`);
                        updateActiveStrategies(); 
                    } else {
                        console.error(`Failed to remove strategy ${strategy.id}.`, result);
                    }
                }
            };
            strategyDiv.appendChild(removeBtn);
            activeStrategiesContainer.appendChild(strategyDiv);
        });
    } else {
        activeStrategiesContainer.innerHTML = '<p>No active strategies.</p>';
    }
}

async function updatePerformanceMetrics() { 
    const performance = await fetchData('/api/performance');
    if (performance) {
        performanceMetricsContainer.innerHTML = `
            <p><strong>Total Account Value:</strong> $${performance.total_account_value.toFixed(2)}</p>
            <p>Last updated: ${new Date(performance.timestamp).toLocaleTimeString()}</p>
        `;
    } else {
        performanceMetricsContainer.innerHTML = '<p>Failed to load performance metrics.</p>';
    }
}

async function updateMlModelsStatus() { 
    const statusData = await fetchData('/api/status'); 
    mlModelsContainer.innerHTML = ''; 

    if (statusData && statusData.ml_model_status && Object.keys(statusData.ml_model_status).length > 0) {
        for (const modelId in statusData.ml_model_status) {
            const model = statusData.ml_model_status[modelId];
            const modelDiv = document.createElement('div');
            modelDiv.className = 'ml-model';
            modelDiv.innerHTML = `
                <h4>${model.model_type} for ${model.symbol} (${model.timeframe})</h4>
                <small>Trained: ${model.trained_date || 'N/A'}</small>
                <p>${model.metric_label || 'Accuracy'}: <strong>${(model.metric_value * 100).toFixed(1)}%</strong></p>
                <p>Status: ${model.loaded ? 'Loaded' : 'Not Loaded'}</p>
                <p>Scaler: ${model.scaler_loaded ? 'Available' : 'Missing'}</p>
            `;
            mlModelsContainer.appendChild(modelDiv);
        }
    } else {
        mlModelsContainer.innerHTML = '<p>No ML models loaded. Train models first using `python train_models.py --full-train`.</p>';
    }
}

// --- Event Listeners ---

startBotBtn.addEventListener('click', async () => {
    const result = await postData('/api/start');
    if (result && result.status === 'success') {
        console.log('Bot started.');
        updateBotStatusUI(true);
    } else {
        console.error('Failed to start bot.', result);
    }
});

stopBotBtn.addEventListener('click', async () => {
    const result = await postData('/api/stop');
    if (result && result.status === 'success') {
        alert('Bot stopped.');
        updateBotStatusUI(false);
    } else {
        console.error('Failed to stop bot.', result);
    }
});

// Listener for strategy type selection change
strategyTypeSelect.addEventListener('change', (event) => {
    generateConfigInputs(event.target.value);
});

// Listener for Add Strategy button
addStrategyBtn.addEventListener('click', async () => {
    const strategyId = strategyIdInput.value.trim();
    const strategyType = strategyTypeSelect.value;
    const config = collectConfigValues();

    if (!strategyId) {
        alert("Please enter a Strategy ID.");
        return;
    }
    if (!strategyType) {
        alert("Please select a Strategy Type.");
        return;
    }

    const payload = { id: strategyId, type: strategyType, config: config };
    const result = await postData('/api/strategies/add', payload);

    if (result && result.status === 'success') {
        alert(`Strategy ${strategyId} added successfully!`);
        strategyIdInput.value = ''; 
        strategyTypeSelect.value = ''; 
        generateConfigInputs(''); 
        updateActiveStrategies();
    } else {
        console.error('Failed to add strategy:', result);
    }
});


trainModelsBtn.addEventListener('click', () => {
    alert("ML model training initiated. Please monitor the console where 'python train_models.py --full-train' is running.");
});

sendChatBtn.addEventListener('click', async () => {
    const message = chatInput.value.trim();
    if (message) {
        appendChatMessage('user', message);
        chatInput.value = '';
        const response = await postData('/api/chat', { message: message });
        if (response && response.response) {
            appendChatMessage('bot', response.response);
        } else {
            appendChatMessage('bot', 'Error: Could not get a response.');
        }
    }
});

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault(); // Prevent default New Line behavior in input field
        sendChatBtn.click();
    }
});

function appendChatMessage(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = message;
    chatMessagesElem.appendChild(messageDiv);
    chatMessagesElem.scrollTop = chatMessagesElem.scrollHeight;
}

// --- Initial Data Load & Periodic Updates ---
async function initializeDashboard() {
    await populateStrategyTypeSelect(); // Populate dropdown first
    await updateAccountSummary();
    await updateMarketData();
    await updateActiveStrategies();
    await updatePerformanceMetrics();
    await updateMlModelsStatus(); 
    generateConfigInputs(''); 
}

initializeDashboard(); // Call once on load

// Set up periodic updates for various sections (e.g., every 10 seconds)
setInterval(updateAccountSummary, 10000);
setInterval(updateMarketData, 10000);
setInterval(updateActiveStrategies, 30000);
setInterval(updatePerformanceMetrics, 60000);
setInterval(updateMlModelsStatus, 60000);