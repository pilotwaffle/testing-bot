#!/usr/bin/env python3
"""
File: E:\Trade Chat Bot\G Trading Bot\deploy_dashboard_js.py
Location: E:\Trade Chat Bot\G Trading Bot\deploy_dashboard_js.py

📜 Elite Trading Bot V3.0 - Dashboard.js Deployment
Create the industrial dashboard JavaScript file
"""

import os
from pathlib import Path

def create_dashboard_js():
    """Create the industrial dashboard JavaScript file"""
    print("📜 Creating dashboard.js for industrial interface...")
    
    # Ensure static/js directory exists
    js_dir = Path("static/js")
    js_dir.mkdir(parents=True, exist_ok=True)
    
    dashboard_js_content = '''// File: E:\\Trade Chat Bot\\G Trading Bot\\static\\js\\dashboard.js
// Location: E:\\Trade Chat Bot\\G Trading Bot\\static\\js\\dashboard.js

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
            this.initializeNavigation();
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
                strategies: [
                    { id: 'momentum_scalping', name: 'Momentum Scalping', risk_level: 'High' },
                    { id: 'trend_following', name: 'Trend Following', risk_level: 'Medium' },
                    { id: 'mean_reversion', name: 'Mean Reversion', risk_level: 'Low' }
                ]
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

    async loadAvailableStrategies() {
        try {
            const data = await this.fetchData('/api/strategies/available');
            
            if (data.status === 'success' && data.strategies) {
                const select = document.getElementById('strategySelect');
                if (select) {
                    select.innerHTML = '<option value="">Select Strategy...</option>';
                    
                    data.strategies.forEach(strategy => {
                        const option = document.createElement('option');
                        option.value = strategy.id;
                        option.textContent = `${strategy.name} (${strategy.risk_level})`;
                        select.appendChild(option);
                    });
                }
            }
        } catch (error) {
            console.error('Error loading strategies:', error);
        }
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