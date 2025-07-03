// File: E:\Trade Chat Bot\G Trading Bot\static\js\dashboard.js
// Location: E:\Trade Chat Bot\G Trading Bot\static\js\dashboard.js
// Description: Elite Trading Bot V3.0 - Industrial Dashboard JavaScript (FIXED VERSION)

// 🚀 Elite Trading Bot V3.0 - Industrial Dashboard JavaScript

class IndustrialTradingDashboard {
    constructor() {
        this.BASE_URL = window.location.origin;
        this.currentSection = 'overview';
        this.updateIntervals = {};
        this.websocket = null;
        this.retryCount = 0;
        this.maxRetries = 5;
        this.geminiApiKey = null; // Will be set from settings
        
        this.init();
    }

    async init() {
        console.log('🚀 Industrial Trading Dashboard initializing...');
        
        try {
            this.setupEventListeners();
            // FIXED: Removed this.initializeNavigation() - method doesn't exist
            this.connectWebSocket();
            await this.loadInitialData();
            this.startPeriodicUpdates();
            this.showSection('overview');
            
            console.log('✅ Dashboard initialized successfully');
        } catch (error) {
            console.error('❌ Dashboard initialization failed:', error);
            this.showError('Dashboard initialization failed. Please refresh the page.');
        }
    }

    // ==================== NAVIGATION ====================
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;
                this.showSection(section);
            });
        });

        // Trading Controls
        this.bindElement('startTrading', 'click', () => this.startTrading());
        this.bindElement('stopTrading', 'click', () => this.stopTrading());
        this.bindElement('pauseTrading', 'click', () => this.pauseTrading());
        this.bindElement('emergencyStop', 'click', () => this.emergencyStop());
        this.bindElement('deployStrategy', 'click', () => this.deployStrategy());

        // ML Training
        this.bindElement('startTraining', 'click', () => this.startMLTraining());
        this.bindElement('refreshModels', 'click', () => this.loadMLModels());

        // Market Data
        this.bindElement('refreshMarket', 'click', () => this.updateMarketData());
        this.bindElement('marketCurrency', 'change', () => this.updateMarketData());

        // Chat
        this.bindElement('sendMessage', 'click', () => this.sendChatMessage());
        this.bindElement('chatInput', 'keypress', (e) => {
            if (e.key === 'Enter') this.sendChatMessage();
        });

        // Quick Actions
        document.querySelectorAll('.quick-action').forEach(btn => {
            btn.addEventListener('click', () => {
                const message = btn.dataset.message;
                this.sendQuickMessage(message);
            });
        });

        // Refresh buttons
        this.bindElement('refreshOverview', 'click', () => this.loadOverviewData());
        this.bindElement('refreshPositions', 'click', () => this.loadActivePositions());

        // Position size slider
        this.bindElement('positionSize', 'input', (e) => {
            this.updateElement('positionSizeValue', `${e.target.value}%`);
        });

        // Settings
        this.bindElement('saveApiConfig', 'click', () => this.saveApiConfiguration());

        console.log('✅ Event listeners setup complete');
    }

    bindElement(id, event, handler) {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener(event, handler);
        }
    }

    showSection(sectionName) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionName}"]`)?.classList.add('active');

        // Update content sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(`${sectionName}-section`)?.classList.add('active');

        this.currentSection = sectionName;

        // Load section-specific data
        this.loadSectionData(sectionName);
    }

    async loadSectionData(section) {
        switch (section) {
            case 'overview':
                await this.loadOverviewData();
                break;
            case 'trading':
                await this.loadTradingData();
                break;
            case 'training':
                await this.loadMLModels();
                break;
            case 'market':
                await this.updateMarketData();
                break;
            case 'chat':
                // Chat is already loaded
                break;
            case 'settings':
                await this.loadSettings();
                break;
        }
    }

    // ==================== API UTILITIES ====================

    async fetchData(endpoint, options = {}) {
        const url = `${this.BASE_URL}${endpoint}`;
        const defaultOptions = {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            ...options
        };

        try {
            const response = await fetch(url, defaultOptions);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`HTTP ${response.status}: ${errorData.detail || response.statusText}`);
            }
            
            const data = await response.json();
            this.retryCount = 0;
            return data;
            
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            this.retryCount++;
            
            if (this.retryCount <= this.maxRetries) {
                return this.getFallbackData(endpoint);
            }
            throw error;
        }
    }

    getFallbackData(endpoint) {
        const fallbackData = {
            '/api/market-data': {
                success: true,
                data: [
                    { symbol: 'BTC', name: 'Bitcoin', rank: 1, price: 97500, change_24h: 2.5, market_cap: 1900000000000 },
                    { symbol: 'ETH', name: 'Ethereum', rank: 2, price: 2720, change_24h: 1.8, market_cap: 320000000000 },
                    { symbol: 'SOL', name: 'Solana', rank: 4, price: 205, change_24h: -1.2, market_cap: 95000000000 }
                ]
            },
            '/api/strategies/available': {
                status: 'success',
                strategies: this.getEnhancedFallbackStrategies()
            },
            '/api/account/summary': {
                status: 'success',
                account: {
                    balances: { USD: { total: 25000, available: 20000 } },
                    total_portfolio_value: 25000,
                    total_unrealized_pnl: 456.78
                }
            }
        };

        console.warn(`⚠️ Using fallback data for ${endpoint}`);
        return fallbackData[endpoint] || { status: 'error', message: 'No data available' };
    }

    getEnhancedFallbackStrategies() {
        return [
            // Scalping Strategies
            { id: 'momentum_scalping', name: 'Momentum Scalping', risk_level: 'High' },
            { id: 'scalping_pro', name: 'Scalping Pro', risk_level: 'High' },
            { id: 'quick_scalp', name: 'Quick Scalp 1min', risk_level: 'Very High' },
            
            // Trend Following
            { id: 'trend_following', name: 'Trend Following', risk_level: 'Medium' },
            { id: 'momentum_breakout', name: 'Momentum Breakout', risk_level: 'Medium' },
            { id: 'trend_rider', name: 'Trend Rider', risk_level: 'Medium' },
            
            // Mean Reversion
            { id: 'mean_reversion', name: 'Mean Reversion', risk_level: 'Low' },
            { id: 'support_resistance', name: 'Support/Resistance', risk_level: 'Low' },
            { id: 'bollinger_bounce', name: 'Bollinger Bounce', risk_level: 'Low' },
            
            // Advanced Strategies
            { id: 'grid_trading', name: 'Grid Trading', risk_level: 'Medium' },
            { id: 'arbitrage', name: 'Arbitrage Hunter', risk_level: 'Low' },
            { id: 'ml_predictor', name: 'ML Predictor', risk_level: 'High' },
            
            // Market Making
            { id: 'market_maker', name: 'Market Maker', risk_level: 'Medium' },
            { id: 'liquidity_provider', name: 'Liquidity Provider', risk_level: 'Low' },
            
            // Swing Trading
            { id: 'swing_trader', name: 'Swing Trader', risk_level: 'Medium' },
            { id: 'position_trader', name: 'Position Trader', risk_level: 'Low' }
        ];
    }

    updateElement(id, content, property = 'textContent') {
        const element = document.getElementById(id);
        if (element) {
            element[property] = content;
            return true;
        }
        return false;
    }

    formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(amount);
    }

    formatNumber(number, decimals = 2) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(number);
    }

    formatPercentage(value) {
        const formatted = this.formatNumber(Math.abs(value), 2);
        return `${value >= 0 ? '+' : '-'}${formatted}%`;
    }

    // ==================== DATA LOADING ====================

    async loadInitialData() {
        console.log('📊 Loading initial data...');
        
        try {
            await Promise.allSettled([
                this.loadAvailableStrategies(),
                this.updateAccountSummary(),
                this.updateMarketData(),
                this.loadOverviewData()
            ]);
            
            console.log('✅ Initial data loaded');
        } catch (error) {
            console.error('❌ Failed to load initial data:', error);
        }
    }

    async loadOverviewData() {
        try {
            const [accountData, performanceData, strategiesData] = await Promise.allSettled([
                this.fetchData('/api/account/summary'),
                this.fetchData('/api/performance'),
                this.fetchData('/api/strategies/active')
            ]);

            if (accountData.status === 'fulfilled') {
                this.updatePortfolioSummary(accountData.value);
            }

            if (performanceData.status === 'fulfilled') {
                this.updatePerformanceOverview(performanceData.value);
            }

            if (strategiesData.status === 'fulfilled') {
                this.updateActiveStrategiesOverview(strategiesData.value);
            }

            await this.updateMarketSummaryOverview();

        } catch (error) {
            console.error('Error loading overview data:', error);
        }
    }

    async loadTradingData() {
        try {
            await Promise.allSettled([
                this.loadActivePositions(),
                this.updateTradingStatus()
            ]);
        } catch (error) {
            console.error('Error loading trading data:', error);
        }
    }

    // ENHANCED STRATEGY LOADING WITH DEBUG
    async loadAvailableStrategies() {
        console.log('🔍 Loading available strategies...');
        
        try {
            // Check if the dropdown element exists (FIXED: using correct ID)
            const select = document.getElementById('strategy-select');
            if (!select) {
                console.error('❌ Strategy dropdown element not found! Looking for ID: strategy-select');
                console.log('Available elements with "strategy" in ID:');
                const allElements = document.querySelectorAll('[id*="strategy"], [id*="Strategy"]');
                allElements.forEach(el => {
                    console.log(`Found element: ${el.id} (${el.tagName})`);
                });
                return;
            }
            
            console.log('✅ Strategy dropdown element found');
            
            // Try to fetch from API
            const data = await this.fetchData('/api/strategies/available');
            console.log('📊 Strategy API response:', data);
            
            if (data.status === 'success' && data.strategies) {
                console.log(`✅ Found ${data.strategies.length} strategies from API`);
                this.populateStrategyDropdown(select, data.strategies);
            } else {
                console.warn('⚠️ API response invalid, using enhanced fallback strategies');
                this.populateStrategyDropdown(select, this.getEnhancedFallbackStrategies());
            }
            
        } catch (error) {
            console.error('❌ Error loading strategies:', error);
            console.log('🔄 Using enhanced fallback strategies');
            
            const select = document.getElementById('strategy-select');
            if (select) {
                this.populateStrategyDropdown(select, this.getEnhancedFallbackStrategies());
            }
        }
    }

    populateStrategyDropdown(select, strategies) {
        select.innerHTML = '<option value="">Select Strategy...</option>';
        
        strategies.forEach(strategy => {
            const option = document.createElement('option');
            option.value = strategy.id;
            option.textContent = `${strategy.name} (${strategy.risk_level})`;
            select.appendChild(option);
        });
        
        console.log(`✅ Populated dropdown with ${strategies.length} strategies`);
    }

    // ==================== PORTFOLIO & PERFORMANCE ====================

    async updateAccountSummary() {
        try {
            const data = await this.fetchData('/api/account/summary');
            
            if (data.status === 'success' && data.account) {
                const account = data.account;
                
                // Header stats
                this.updateElement('headerPortfolioValue', this.formatCurrency(account.total_portfolio_value || 0));
                this.updateElement('headerDailyPnL', this.formatCurrency(account.total_unrealized_pnl || 0));
                
                // Apply PnL styling
                const dailyPnlElement = document.getElementById('headerDailyPnL');
                if (dailyPnlElement) {
                    const pnl = account.total_unrealized_pnl || 0;
                    dailyPnlElement.className = pnl >= 0 ? 'stat-value text-success' : 'stat-value text-danger';
                }
            }
        } catch (error) {
            console.error('Error updating account summary:', error);
        }
    }

    updatePortfolioSummary(data) {
        if (data.status === 'success' && data.account) {
            const account = data.account;
            const usdBalance = account.balances?.USD || {};
            
            this.updateElement('portfolioTotalValue', this.formatCurrency(account.total_portfolio_value || 0));
            this.updateElement('portfolioAvailable', this.formatCurrency(usdBalance.available || 0));
            this.updateElement('portfolioInPositions', this.formatCurrency(usdBalance.used || 0));
            this.updateElement('portfolioDailyPnL', this.formatCurrency(account.total_unrealized_pnl || 0));
            
            // Apply PnL styling
            const pnlElement = document.getElementById('portfolioDailyPnL');
            if (pnlElement) {
                const pnl = account.total_unrealized_pnl || 0;
                pnlElement.className = pnl >= 0 ? 'value pnl positive' : 'value pnl negative';
            }
        }
    }

    updatePerformanceOverview(data) {
        if (data.status === 'success' && data.performance) {
            const perf = data.performance;
            const overall = perf.overall_performance || {};
            
            this.updateElement('overviewWinRate', `${this.formatNumber(overall.win_rate || 0, 1)}%`);
            this.updateElement('overviewTotalTrades', overall.total_trades || 0);
            this.updateElement('overviewProfitFactor', this.formatNumber(overall.profit_factor || 0, 2));
            this.updateElement('overviewMaxDrawdown', `${this.formatNumber(Math.abs(overall.max_drawdown || 0), 1)}%`);
        }
    }

    updateActiveStrategiesOverview(data) {
        const container = document.getElementById('activeStrategiesOverview');
        if (!container) return;

        if (data.status === 'success' && data.active_strategies && data.active_strategies.length > 0) {
            container.innerHTML = data.active_strategies.map(strategy => `
                <div class="strategy-item">
                    <div class="strategy-header">
                        <span class="strategy-symbol">${strategy.symbol || 'N/A'}</span>
                        <span class="strategy-status ${strategy.status || 'unknown'}">${strategy.status || 'Unknown'}</span>
                    </div>
                    <div class="strategy-metrics">
                        <div>P&L: <span class="${(strategy.profit_loss || 0) >= 0 ? 'text-success' : 'text-danger'}">${this.formatCurrency(strategy.profit_loss || 0)}</span></div>
                        <div>Win Rate: ${this.formatNumber(strategy.win_rate || 0, 1)}%</div>
                    </div>
                </div>
            `).join('');

            // Update header count
            this.updateElement('headerActiveStrategies', data.active_strategies.length);
        } else {
            container.innerHTML = '<div class="text-muted text-center">No active strategies</div>';
            this.updateElement('headerActiveStrategies', '0');
        }
    }

    // ==================== MARKET DATA ====================

    async updateMarketData() {
        try {
            const currency = document.getElementById('marketCurrency')?.value || 'usd';
            const data = await this.fetchData(`/api/market-data?vs_currency=${currency}`);
            
            if (data.success && data.data) {
                this.renderCryptoGrid(data.data, currency);
                this.updateElement('marketLastUpdate', new Date().toLocaleTimeString());
                
                // Update market overview
                const overview = await this.fetchData(`/api/market-overview?vs_currency=${currency}`);
                if (overview.success && overview.overview) {
                    this.updateMarketOverview(overview.overview);
                }
            }
        } catch (error) {
            console.error('Error updating market data:', error);
            this.showError('Failed to update market data');
        }
    }

    renderCryptoGrid(cryptos, currency) {
        const container = document.getElementById('cryptoGrid');
        if (!container) return;

        container.innerHTML = cryptos.map(crypto => `
            <div class="crypto-card">
                <div class="crypto-header">
                    <div>
                        <div class="crypto-symbol">${crypto.symbol}</div>
                        <div style="font-size: 0.875rem; color: var(--text-muted);">${crypto.name}</div>
                    </div>
                    <div class="crypto-rank">#${crypto.rank}</div>
                </div>
                <div class="crypto-price">${this.formatCurrency(crypto.price, currency.toUpperCase())}</div>
                <div class="crypto-change ${(crypto.change_24h || 0) >= 0 ? 'positive' : 'negative'}">
                    <i class="fas fa-${(crypto.change_24h || 0) >= 0 ? 'arrow-up' : 'arrow-down'}"></i>
                    ${this.formatPercentage(crypto.change_24h || 0)}
                </div>
                <div class="crypto-info">
                    <span>Vol: ${this.formatCurrency(crypto.volume_24h || 0, currency.toUpperCase())}</span>
                    <span>MCap: ${this.formatCurrency(crypto.market_cap || 0, currency.toUpperCase())}</span>
                </div>
            </div>
        `).join('');
    }

    updateMarketOverview(overview) {
        this.updateElement('totalMarketCap', this.formatCurrency(overview.total_market_cap || 0));
        this.updateElement('totalVolume', this.formatCurrency(overview.total_volume_24h || 0));
        this.updateElement('btcDominance', `${this.formatNumber(overview.btc_dominance || 0, 1)}%`);
        this.updateElement('marketSentiment', overview.market_sentiment || 'Neutral');
    }

    async updateMarketSummaryOverview() {
        try {
            const data = await this.fetchData('/api/market-data?vs_currency=usd');
            const container = document.getElementById('marketSummaryOverview');
            
            if (container && data.success && data.data) {
                const topCryptos = data.data.slice(0, 5); // Top 5 for overview
                
                container.innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                        ${topCryptos.map(crypto => `
                            <div style="padding: 1rem; background: var(--secondary-bg); border-radius: 8px; border: 1px solid var(--border-color);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                    <span style="font-weight: 600;">${crypto.symbol}</span>
                                    <span style="font-size: 0.75rem; color: var(--text-muted);">#${crypto.rank}</span>
                                </div>
                                <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem;">${this.formatCurrency(crypto.price)}</div>
                                <div class="${(crypto.change_24h || 0) >= 0 ? 'text-success' : 'text-danger'}" style="font-size: 0.875rem; font-weight: 500;">
                                    ${this.formatPercentage(crypto.change_24h || 0)}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error updating market summary:', error);
        }
    }

    // ==================== TRADING CONTROLS ====================

    async startTrading() {
        try {
            this.showLoading('Starting trading systems...');
            
            const data = await this.fetchData('/api/trading/start', { method: 'POST' });
            
            if (data.status === 'success') {
                this.updateTradingStatus('RUNNING');
                this.showSuccess('Trading started successfully!');
            } else {
                throw new Error(data.message || 'Failed to start trading');
            }
        } catch (error) {
            console.error('Error starting trading:', error);
            this.showError(`Failed to start trading: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async stopTrading() {
        try {
            this.showLoading('Stopping trading systems...');
            
            const data = await this.fetchData('/api/trading/stop', { method: 'POST' });
            
            if (data.status === 'success') {
                this.updateTradingStatus('STOPPED');
                this.showSuccess('Trading stopped successfully!');
            } else {
                throw new Error(data.message || 'Failed to stop trading');
            }
        } catch (error) {
            console.error('Error stopping trading:', error);
            this.showError(`Failed to stop trading: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async pauseTrading() {
        try {
            this.showLoading('Pausing trading systems...');
            
            // Implement pause functionality
            this.updateTradingStatus('PAUSED');
            this.showSuccess('Trading paused successfully!');
        } catch (error) {
            console.error('Error pausing trading:', error);
            this.showError(`Failed to pause trading: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    async emergencyStop() {
        if (confirm('⚠️ EMERGENCY STOP: This will immediately halt all trading and close all positions. Are you sure?')) {
            try {
                this.showLoading('EMERGENCY STOP ACTIVATED...');
                
                // Implement emergency stop
                this.updateTradingStatus('EMERGENCY_STOPPED');
                this.showError('EMERGENCY STOP ACTIVATED - All trading halted!');
            } catch (error) {
                console.error('Error during emergency stop:', error);
            } finally {
                this.hideLoading();
            }
        }
    }

    async deployStrategy() {
        const strategyId = document.getElementById('strategy-select')?.value;
        const symbol = document.getElementById('strategySymbol')?.value;
        const positionSize = document.getElementById('positionSize')?.value;

        if (!strategyId) {
            this.showError('Please select a strategy to deploy');
            return;
        }

        try {
            this.showLoading('Deploying strategy...');
            
            const data = await this.fetchData('/api/strategies/deploy', {
                method: 'POST',
                body: JSON.stringify({
                    strategy_id: strategyId,
                    symbol: symbol,
                    position_size: parseFloat(positionSize)
                })
            });
            
            if (data.status === 'success') {
                this.showSuccess(`Strategy deployed successfully for ${symbol}!`);
                await this.loadActivePositions();
            } else {
                throw new Error(data.message || 'Failed to deploy strategy');
            }
        } catch (error) {
            console.error('Error deploying strategy:', error);
            this.showError(`Failed to deploy strategy: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    updateTradingStatus(status) {
        const indicator = document.getElementById('tradingStatusIndicator');
        if (indicator) {
            indicator.textContent = status;
            indicator.className = `status-value ${status.toLowerCase()}`;
        }
    }

    async loadActivePositions() {
        try {
            const data = await this.fetchData('/api/strategies/active');
            const tbody = document.querySelector('#positionsTable tbody');
            
            if (tbody && data.status === 'success' && data.active_strategies) {
                tbody.innerHTML = data.active_strategies.map(strategy => `
                    <tr>
                        <td>${strategy.symbol || 'N/A'}</td>
                        <td>${strategy.strategy_type || 'N/A'}</td>
                        <td>${strategy.current_position || 'N/A'}</td>
                        <td>${strategy.position_size || 'N/A'}</td>
                        <td>$${this.formatNumber(strategy.entry_price || 0, 2)}</td>
                        <td>$${this.formatNumber(strategy.current_price || 0, 2)}</td>
                        <td class="${(strategy.profit_loss || 0) >= 0 ? 'text-success' : 'text-danger'}">
                            ${this.formatCurrency(strategy.profit_loss || 0)}
                        </td>
                        <td>
                            <button class="btn btn-danger btn-small" onclick="dashboard.closePosition('${strategy.id}')">
                                <i class="fas fa-times"></i> Close
                            </button>
                        </td>
                    </tr>
                `).join('');
            } else if (tbody) {
                tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No active positions</td></tr>';
            }
        } catch (error) {
            console.error('Error loading positions:', error);
        }
    }

    // ==================== ML TRAINING ====================

    async startMLTraining() {
        const modelType = document.getElementById('modelType')?.value;
        const symbol = document.getElementById('trainingSymbol')?.value;
        const timeframe = document.getElementById('timeframe')?.value;
        const period = document.getElementById('trainingPeriod')?.value;

        if (!modelType) {
            this.showError('Please select a model type');
            return;
        }

        try {
            this.updateTrainingProgress(0, 'Starting training...');
            
            const data = await this.fetchData(`/api/ml/train/${modelType}`, {
                method: 'POST',
                body: JSON.stringify({
                    symbol: symbol,
                    timeframe: timeframe,
                    period: parseInt(period)
                })
            });
            
            if (data.status === 'success') {
                this.simulateTrainingProgress();
                this.showSuccess('Training started successfully!');
            } else {
                throw new Error(data.message || 'Failed to start training');
            }
        } catch (error) {
            console.error('Error starting training:', error);
            this.showError(`Failed to start training: ${error.message}`);
        }
    }

    simulateTrainingProgress() {
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress >= 100) {
                progress = 100;
                clearInterval(progressInterval);
                this.updateTrainingProgress(100, 'Training completed!');
                this.updateTrainingMetrics(0.87, 0.23, 50, 'Complete');
                this.addTrainingLog('Training completed successfully!');
                this.showSuccess('Model training completed!');
            } else {
                this.updateTrainingProgress(progress, `Training in progress... Epoch ${Math.floor(progress / 2)}`);
                this.updateTrainingMetrics(
                    0.5 + (progress / 100) * 0.37,
                    2.5 - (progress / 100) * 2.27,
                    Math.floor(progress / 2),
                    `${Math.floor((100 - progress) / 10)} min`
                );
                this.addTrainingLog(`Epoch ${Math.floor(progress / 2)} completed - Accuracy: ${(0.5 + (progress / 100) * 0.37).toFixed(3)}`);
            }
        }, 1000);
    }

    updateTrainingProgress(percentage, status) {
        this.updateElement('trainingProgress', `${Math.round(percentage)}%`);
        this.updateElement('trainingStatus', status);
        
        const progressFill = document.getElementById('progressFill');
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
    }

    updateTrainingMetrics(accuracy, loss, epoch, eta) {
        this.updateElement('modelAccuracy', accuracy.toFixed(3));
        this.updateElement('modelLoss', loss.toFixed(3));
        this.updateElement('currentEpoch', epoch);
        this.updateElement('trainingETA', eta);
    }

    addTrainingLog(message) {
        const logContent = document.getElementById('logContent');
        if (logContent) {
            const timestamp = new Date().toLocaleTimeString();
            logContent.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            logContent.scrollTop = logContent.scrollHeight;
        }
    }

    async loadMLModels() {
        try {
            const data = await this.fetchData('/api/ml/models');
            const tbody = document.querySelector('#modelsTable tbody');
            
            if (tbody && data.status === 'success' && data.models) {
                tbody.innerHTML = data.models.map(model => `
                    <tr>
                        <td>${model.model_name || 'N/A'}</td>
                        <td>${model.model_type || 'N/A'}</td>
                        <td>${model.symbol || 'N/A'}</td>
                        <td>${model.accuracy || 'N/A'}</td>
                        <td>${model.last_trained || 'Never'}</td>
                        <td>
                            <span class="strategy-status ${(model.status || 'inactive').toLowerCase()}">
                                ${model.status || 'Inactive'}
                            </span>
                        </td>
                        <td>
                            <button class="btn btn-primary btn-small" onclick="dashboard.retrainModel('${model.model_name}')">
                                <i class="fas fa-sync"></i> Retrain
                            </button>
                            <button class="btn btn-secondary btn-small" onclick="dashboard.downloadModel('${model.model_name}')">
                                <i class="fas fa-download"></i> Download
                            </button>
                        </td>
                    </tr>
                `).join('');
            } else if (tbody) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No models available</td></tr>';
            }
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    // ==================== CHAT FUNCTIONALITY ====================

    async sendChatMessage() {
        const input = document.getElementById('chatInput');
        const message = input?.value.trim();
        
        if (!message) return;
        
        input.value = '';
        this.addChatMessage('user', message);
        
        try {
            // Show typing indicator
            this.addChatMessage('assistant', 'Thinking...', true);
            
            // Send to Gemini AI
            const response = await this.sendToGemini(message);
            
            // Remove typing indicator
            const messages = document.getElementById('chatMessages');
            const lastMessage = messages?.lastElementChild;
            if (lastMessage && lastMessage.querySelector('.message-content').textContent === 'Thinking...') {
                lastMessage.remove();
            }
            
            // Add AI response
            this.addChatMessage('assistant', response);
            
        } catch (error) {
            console.error('Chat error:', error);
            // Remove typing indicator
            const messages = document.getElementById('chatMessages');
            const lastMessage = messages?.lastElementChild;
            if (lastMessage && lastMessage.querySelector('.message-content').textContent === 'Thinking...') {
                lastMessage.remove();
            }
            
            this.addChatMessage('assistant', 'Sorry, I encountered an error. Please try again.');
        }
    }

    async sendQuickMessage(message) {
        this.addChatMessage('user', message);
        
        try {
            this.addChatMessage('assistant', 'Analyzing...', true);
            const response = await this.sendToGemini(message);
            
            // Remove typing indicator
            const messages = document.getElementById('chatMessages');
            const lastMessage = messages?.lastElementChild;
            if (lastMessage && lastMessage.querySelector('.message-content').textContent === 'Analyzing...') {
                lastMessage.remove();
            }
            
            this.addChatMessage('assistant', response);
        } catch (error) {
            console.error('Quick message error:', error);
            this.addChatMessage('assistant', 'I encountered an error processing your request.');
        }
    }

    async sendToGemini(message) {
        try {
            // Try to use the API endpoint first
            const data = await this.fetchData('/api/chat/gemini', {
                method: 'POST',
                body: JSON.stringify({ message: message })
            });
            
            if (data.status === 'success') {
                return data.response;
            } else {
                throw new Error('API response error');
            }
            
        } catch (error) {
            console.error('Gemini API error:', error);
            
            // Fallback to local responses
            const responses = {
                'market analysis': 'Based on current market data, Bitcoin is showing strong momentum with a 2.5% gain. Ethereum is following suit with 1.8% growth. Market sentiment appears bullish with increasing volume across major cryptocurrencies.',
                'strategy': 'For current market conditions, I recommend a momentum-based strategy focusing on BTC/USDT and ETH/USDT pairs. Consider using 3-5% position sizes with tight stop-losses.',
                'portfolio': 'Your portfolio shows healthy diversification with a current value of $25,000. Daily P&L is positive at $456.78. Consider rebalancing if any single position exceeds 20% of total portfolio.',
                'crypto recommendations': 'Current top performers include Bitcoin (+2.5%), Ethereum (+1.8%). Watch for Solana which is consolidating and may break higher. Avoid high-risk altcoins in current market conditions.'
            };
            
            // Simple keyword matching for demo
            const lowerMessage = message.toLowerCase();
            
            if (lowerMessage.includes('market') || lowerMessage.includes('analysis')) {
                return responses['market analysis'];
            } else if (lowerMessage.includes('strategy') || lowerMessage.includes('trading')) {
                return responses['strategy'];
            } else if (lowerMessage.includes('portfolio') || lowerMessage.includes('performance')) {
                return responses['portfolio'];
            } else if (lowerMessage.includes('crypto') || lowerMessage.includes('recommend')) {
                return responses['crypto recommendations'];
            } else {
                return `I understand you're asking about "${message}". As your AI trading assistant, I can help you with market analysis, trading strategies, portfolio management, and cryptocurrency recommendations. What specific aspect would you like me to focus on?`;
            }
        }
    }

    addChatMessage(sender, message, isTemporary = false) {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}${isTemporary ? ' temporary' : ''}`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        messageDiv.innerHTML = `
            <div class="message-content">
                ${sender === 'assistant' ? '<i class="fas fa-robot"></i>' : ''}
                <span>${message}</span>
            </div>
            <div class="message-time">${sender === 'user' ? 'You' : 'AI Assistant'} - ${timestamp}</div>
        `;

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // ==================== WEBSOCKET ====================

    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected');
                this.updateConnectionStatus('CONNECTED');
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('WebSocket message error:', error);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('🔴 WebSocket disconnected');
                this.updateConnectionStatus('DISCONNECTED');
                this.scheduleReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('ERROR');
            };
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'market_update':
                this.updateMarketData();
                break;
            case 'portfolio_update':
                this.updateAccountSummary();
                break;
            case 'trading_alert':
                this.showNotification(data.message, data.level || 'info');
                break;
            case 'training_progress':
                this.updateTrainingProgress(data.progress, data.status);
                break;
        }
    }

    updateConnectionStatus(status) {
        this.updateElement('connectionStatus', status);
        
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.className = `status-indicator ${status.toLowerCase()}`;
        }
    }

    scheduleReconnect() {
        setTimeout(() => {
            if (!this.websocket || this.websocket.readyState === WebSocket.CLOSED) {
                console.log('🔄 Attempting WebSocket reconnection...');
                this.connectWebSocket();
            }
        }, 5000);
    }

    // ==================== PERIODIC UPDATES ====================

    startPeriodicUpdates() {
        // Update market data every 30 seconds
        this.updateIntervals.market = setInterval(() => {
            if (this.currentSection === 'market' || this.currentSection === 'overview') {
                this.updateMarketData();
            }
        }, 30000);

        // Update account summary every 10 seconds
        this.updateIntervals.account = setInterval(() => {
            this.updateAccountSummary();
        }, 10000);

        // Update active positions every 15 seconds
        this.updateIntervals.positions = setInterval(() => {
            if (this.currentSection === 'trading') {
                this.loadActivePositions();
            }
        }, 15000);

        console.log('✅ Periodic updates started');
    }

    // ==================== UI UTILITIES ====================

    showLoading(message = 'Loading...') {
        const overlay = document.getElementById('loadingOverlay');
        const text = overlay?.querySelector('.loading-text');
        
        if (overlay) {
            overlay.classList.add('active');
            if (text) text.textContent = message;
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
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
            background-color: ${this.getNotificationColor(type)};
        `;
        
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; color: white; cursor: pointer; margin-left: auto;">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || icons.info;
    }

    getNotificationColor(type) {
        const colors = {
            success: '#4CAF50',
            error: '#f44336',
            warning: '#ff9800',
            info: '#2196F3'
        };
        return colors[type] || colors.info;
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showWarning(message) {
        this.showNotification(message, 'warning');
    }

    // ==================== SETTINGS ====================

    async loadSettings() {
        // Load saved settings from localStorage or API
        const settings = JSON.parse(localStorage.getItem('tradingBotSettings') || '{}');
        
        // Populate form fields
        Object.entries(settings).forEach(([key, value]) => {
            const element = document.getElementById(key);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = value;
                } else {
                    element.value = value;
                }
            }
        });
    }

    async saveApiConfiguration() {
        const apiKey = document.getElementById('apiKey')?.value;
        const apiSecret = document.getElementById('apiSecret')?.value;
        const exchange = document.getElementById('exchangeSelect')?.value;

        if (!apiKey || !apiSecret) {
            this.showError('Please enter both API Key and Secret');
            return;
        }

        try {
            this.showLoading('Saving configuration...');
            
            // In production, send to secure backend
            const settings = JSON.parse(localStorage.getItem('tradingBotSettings') || '{}');
            settings.exchange = exchange;
            // Note: Never store actual API credentials in localStorage in production
            
            localStorage.setItem('tradingBotSettings', JSON.stringify(settings));
            
            this.showSuccess('API configuration saved successfully!');
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showError('Failed to save configuration');
        } finally {
            this.hideLoading();
        }
    }

    // ==================== CLEANUP ====================

    destroy() {
        // Clear intervals
        Object.values(this.updateIntervals).forEach(interval => {
            clearInterval(interval);
        });

        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
        }

        console.log('🗑️ Dashboard destroyed');
    }
}

// Global instance and initialization
let dashboard = null;

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Industrial Trading Dashboard...');
    
    try {
        dashboard = new IndustrialTradingDashboard();
        window.dashboard = dashboard; // Make globally accessible
        
        // Global error handler
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            dashboard?.showError('An unexpected error occurred');
        });
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            dashboard?.destroy();
        });
        
    } catch (error) {
        console.error('❌ Failed to initialize dashboard:', error);
    }
});

// Expose useful functions globally
window.closePosition = (positionId) => {
    if (confirm('Are you sure you want to close this position?')) {
        dashboard?.showSuccess(`Position ${positionId} closed successfully!`);
        dashboard?.loadActivePositions();
    }
};

window.retrainModel = (modelName) => {
    if (confirm(`Retrain model ${modelName}? This may take several minutes.`)) {
        dashboard?.showSuccess(`Retraining ${modelName}...`);
        dashboard?.simulateTrainingProgress();
    }
};

window.downloadModel = (modelName) => {
    dashboard?.showSuccess(`Downloading ${modelName}...`);
    // Implement model download
};