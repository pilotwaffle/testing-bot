// File: enhanced-dashboard.js
// Elite Trading Bot V3.0 - Enhanced Dashboard Extensions with Live Market Prices

// 🚀 Elite Trading Bot V3.0 - Enhanced Dashboard Extensions

// Enhanced features and utilities for the main dashboard
class DashboardEnhancements {
    constructor() {
        this.performanceMonitor = new PerformanceMonitor();
        this.advancedCharts = new AdvancedCharts();
        this.dataCache = new Map();
        this.init();
    }

    init() {
        console.log('🔧 Initializing dashboard enhancements...');
        
        // Wait for main dashboard to be ready
        if (window.dashboard) {
            this.enhanceExistingDashboard();
        } else {
            // Wait for main dashboard to load
            const checkForDashboard = setInterval(() => {
                if (window.dashboard) {
                    clearInterval(checkForDashboard);
                    this.enhanceExistingDashboard();
                }
            }, 100);
        }
    }

    enhanceExistingDashboard() {
        console.log('✨ Enhancing existing dashboard...');
        
        // Add enhanced features to existing dashboard
        this.addKeyboardShortcuts();
        this.addPerformanceMonitoring();
        this.addAdvancedNotifications();
        this.addDataCaching();
        
        console.log('✅ Dashboard enhancements loaded');
    }

    addKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + R: Refresh data
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                if (window.dashboard && window.dashboard.refreshAllData) {
                    window.dashboard.refreshAllData();
                }
            }
            
            // Number keys 1-5: Switch sections
            if (e.key >= '1' && e.key <= '5') {
                const sections = ['trading', 'training', 'market', 'chat', 'settings'];
                const sectionIndex = parseInt(e.key) - 1;
                if (sections[sectionIndex] && window.dashboard) {
                    window.dashboard.showSection(sections[sectionIndex]);
                }
            }
        });
        
        console.log('⌨️ Keyboard shortcuts enabled');
    }

    addPerformanceMonitoring() {
        this.performanceMonitor.start();
        console.log('📊 Performance monitoring enabled');
    }

    addAdvancedNotifications() {
        // Enhanced notification system
        if (window.dashboard) {
            const originalShowNotification = window.dashboard.showNotification?.bind(window.dashboard);
            
            window.dashboard.showNotification = (message, type = 'info', options = {}) => {
                // Call original notification
                if (originalShowNotification) originalShowNotification(message, type);
                
                // Add browser notification if permitted
                if ('Notification' in window && Notification.permission === 'granted' && options.browser) {
                    new Notification('Elite Trading Bot', {
                        body: message,
                        icon: '/static/favicon.ico'
                    });
                }
            };
        }
        
        console.log('🔔 Advanced notifications enabled');
    }

    addDataCaching() {
        // Simple data caching to improve performance
        if (window.dashboard) {
            const originalFetch = window.fetch;
            
            window.fetch = async (url, options = {}) => {
                // Only cache GET requests
                if (!options.method || options.method.toLowerCase() === 'get') {
                    const cacheKey = url;
                    const cached = this.dataCache.get(cacheKey);
                    
                    if (cached && Date.now() - cached.timestamp < 30000) { // 30 second cache
                        console.log(`📦 Using cached data for: ${url}`);
                        return Promise.resolve(new Response(JSON.stringify(cached.data), {
                            status: 200,
                            headers: { 'Content-Type': 'application/json' }
                        }));
                    }
                }
                
                const response = await originalFetch(url, options);
                
                // Cache successful GET responses
                if (response.ok && (!options.method || options.method.toLowerCase() === 'get')) {
                    try {
                        const clonedResponse = response.clone();
                        const data = await clonedResponse.json();
                        this.dataCache.set(url, {
                            data: data,
                            timestamp: Date.now()
                        });
                    } catch (e) {
                        // Not JSON, skip caching
                    }
                }
                
                return response;
            };
        }
        
        console.log('💾 Data caching enabled');
    }

    // Method to clear cache for specific endpoints
    clearCache(endpoint) {
        if (endpoint) {
            this.dataCache.delete(endpoint);
            console.log(`🗑️ Cleared cache for: ${endpoint}`);
        } else {
            this.dataCache.clear();
            console.log('🗑️ Cleared all cache');
        }
    }

    // Method to get cache status
    getCacheStatus() {
        const cacheInfo = {};
        this.dataCache.forEach((value, key) => {
            cacheInfo[key] = {
                timestamp: value.timestamp,
                age: Date.now() - value.timestamp,
                size: JSON.stringify(value.data).length
            };
        });
        return cacheInfo;
    }
}

// Performance monitoring utility
class PerformanceMonitor {
    constructor() {
        this.metrics = {};
        this.observers = [];
    }

    start() {
        // Monitor API response times
        this.monitorAPICalls();
        
        // Monitor page performance
        this.monitorPagePerformance();
        
        console.log('📈 Performance monitoring started');
    }

    monitorAPICalls() {
        // This would be enhanced with more sophisticated monitoring
        console.log('🔍 API call monitoring active');
    }

    monitorPagePerformance() {
        if ('PerformanceObserver' in window) {
            try {
                const observer = new PerformanceObserver((list) => {
                    for (const entry of list.getEntries()) {
                        if (entry.entryType === 'navigation') {
                            console.log(`📊 Page load time: ${entry.loadEventEnd - entry.loadEventStart}ms`);
                            this.metrics.pageLoadTime = entry.loadEventEnd - entry.loadEventStart;
                        }
                    }
                });
                
                observer.observe({ entryTypes: ['navigation'] });
                this.observers.push(observer);
            } catch (e) {
                console.warn('Performance monitoring not fully supported');
            }
        }
    }

    getMetrics() {
        return this.metrics;
    }

    stop() {
        this.observers.forEach(observer => {
            try {
                observer.disconnect();
            } catch (e) {
                console.warn('Error disconnecting performance observer');
            }
        });
        this.observers = [];
        console.log('📊 Performance monitoring stopped');
    }
}

// Advanced charting utility
class AdvancedCharts {
    constructor() {
        this.charts = {};
    }

    createChart(containerId, data, options = {}) {
        // Placeholder for advanced charting functionality
        console.log(`📊 Creating chart in: ${containerId}`);
        
        // This would integrate with charting libraries like Chart.js, D3, etc.
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="chart-placeholder" style="
                    padding: 2rem;
                    background: var(--secondary-bg, #f8f9fa);
                    border: 2px dashed var(--border-color, #dee2e6);
                    border-radius: 8px;
                    text-align: center;
                    color: var(--text-muted, #6c757d);
                ">
                    📊 Chart: ${data.length} data points
                    <br>
                    <small>Advanced charting functionality would be implemented here</small>
                </div>
            `;
        }
        
        this.charts[containerId] = { data, options, timestamp: Date.now() };
    }

    updateChart(chartId, newData) {
        console.log(`📈 Updating chart: ${chartId}`);
        
        if (this.charts[chartId]) {
            this.charts[chartId].data = newData;
            this.charts[chartId].timestamp = Date.now();
            
            // Update the display
            this.createChart(chartId, newData, this.charts[chartId].options);
        }
    }

    getChart(chartId) {
        return this.charts[chartId] || null;
    }

    removeChart(chartId) {
        if (this.charts[chartId]) {
            delete this.charts[chartId];
            const container = document.getElementById(chartId);
            if (container) {
                container.innerHTML = '';
            }
            console.log(`📊 Removed chart: ${chartId}`);
        }
    }
}

// ========== LIVE MARKET DATA (CoinGecko) ==========

const supportedCoins = [
    { symbol: "BTC", name: "Bitcoin", id: "bitcoin" },
    { symbol: "ETH", name: "Ethereum", id: "ethereum" },
    { symbol: "SOL", name: "Solana", id: "solana" },
    { symbol: "AVAX", name: "Avalanche", id: "avalanche-2" },
    { symbol: "ADA", name: "Cardano", id: "cardano" },
    { symbol: "XRP", name: "XRP", id: "ripple" },
    { symbol: "DOGE", name: "Dogecoin", id: "dogecoin" },
    { symbol: "DOT", name: "Polkadot", id: "polkadot" },
    { symbol: "ARB", name: "Arbitrum", id: "arbitrum" },
    { symbol: "MATIC", name: "Polygon", id: "matic-network" }
];

const supportedCurrencies = {
    usd: { label: "USD", symbol: "$" },
    eur: { label: "EUR", symbol: "€" },
    btc: { label: "BTC", symbol: "₿" }
};

let currentMarketCurrency = "usd";
let lastPrices = {};

function renderMarketCards(prices = {}, changes = {}) {
    const grid = document.getElementById("crypto-price-grid");
    if (!grid) return;
    grid.innerHTML = "";
    supportedCoins.forEach(coin => {
        const fiat = supportedCurrencies[currentMarketCurrency];
        let price = prices[coin.id]?.[currentMarketCurrency];
        let priceDisplay = price !== undefined
            ? fiat.symbol + price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 8})
            : `<span class="loading"></span>`;
        let change = changes[coin.id]?.[currentMarketCurrency] ?? null;
        let changeClass = "";
        if (change !== null) changeClass = change >= 0 ? "price-positive" : "price-negative";
        let changeDisplay = change !== null ? `<span class="${changeClass}">${change.toFixed(2)}%</span>` : "";
        let card = document.createElement("div");
        card.className = "crypto-card";
        card.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span class="fw-bold">${coin.symbol}</span>
                <span class="text-muted small">${coin.name}</span>
            </div>
            <div class="d-flex align-items-baseline mb-2">
                <span class="fw-bold fs-3 ${changeClass}">${priceDisplay}</span>
                <span class="ms-2">${changeDisplay}</span>
            </div>
            <div class="text-muted small">24h Change</div>
            <button class="btn btn-primary btn-sm w-100 mt-2" onclick="alert('Trading for ${coin.symbol} coming soon!')">
                Trade
            </button>
        `;
        grid.appendChild(card);
    });
}

async function fetchLiveMarketData() {
    const ids = supportedCoins.map(c => c.id).join(",");
    const vs_currency = currentMarketCurrency;
    try {
        const url = `https://api.coingecko.com/api/v3/simple/price?ids=${ids}&vs_currencies=${vs_currency}&include_24hr_change=true`;
        const res = await fetch(url);
        const data = await res.json();
        let prices = {};
        let changes = {};
        supportedCoins.forEach(coin => {
            prices[coin.id] = {};
            prices[coin.id][vs_currency] = data[coin.id]?.[vs_currency];
            changes[coin.id] = {};
            changes[coin.id][vs_currency] = data[coin.id]?.[vs_currency + '_24h_change'];
        });
        lastPrices = prices;
        renderMarketCards(prices, changes);
    } catch (e) {
        renderMarketCards(); // fallback to loading
    }
}

// Setup market data live updates
document.addEventListener("DOMContentLoaded", function() {
    // Initial render
    renderMarketCards();
    fetchLiveMarketData();

    // Currency dropdown
    const currencySelect = document.getElementById("market-currency");
    if (currencySelect) {
        currencySelect.addEventListener("change", function() {
            currentMarketCurrency = this.value;
            renderMarketCards();
            fetchLiveMarketData();
        });
    }

    // Refresh button
    const refreshBtn = document.getElementById("refresh-market");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", function() {
            renderMarketCards();
            fetchLiveMarketData();
        });
    }

    // Auto-refresh every 30 seconds
    setInterval(fetchLiveMarketData, 30000);
});

// ========== NAVIGATION (simple SPA) ==========
function showSection(section) {
    document.querySelectorAll('.section').forEach(sec => sec.classList.remove('active'));
    const activeSection = document.getElementById(section + "-section");
    if (activeSection) activeSection.classList.add('active');
    document.querySelectorAll('.nav-item').forEach(btn => {
        if (btn.dataset.section === section) btn.classList.add('active');
        else btn.classList.remove('active');
    });
}

// Sidebar navigation handlers
document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll(".nav-item").forEach(btn => {
        btn.addEventListener("click", function() {
            let section = this.getAttribute("data-section");
            showSection(section);
        });
    });
});

// ========== DEMO: Update Portfolio Value ==========
function updatePortfolioValue(value) {
    const totalValueElem = document.getElementById("portfolio-total-value");
    const headerPortfolioElem = document.getElementById("header-portfolio-value");
    if (totalValueElem) totalValueElem.textContent = value.toLocaleString(undefined, {minimumFractionDigits:2});
    if (headerPortfolioElem) headerPortfolioElem.textContent = `$${value.toLocaleString(undefined, {minimumFractionDigits:2})}`;
}
updatePortfolioValue(265710.00);

// ========== STRATEGY DROPDOWN POPULATION ==========

function populateStrategyDropdown(strategies) {
    const select = document.getElementById('strategy-select');
    if (!select) return;
    select.innerHTML = '<option value="">Select Strategy...</option>'; // Reset
    strategies.forEach(strategy => {
        const option = document.createElement('option');
        option.value = strategy.value;
        option.textContent = strategy.label;
        select.appendChild(option);
    });
}

// Example: Replace or extend with your real strategy list or API call
const availableStrategies = [
    { value: 'mean_reversion', label: 'Mean Reversion' },
    { value: 'momentum', label: 'Momentum' },
    { value: 'breakout', label: 'Breakout' },
    { value: 'arbitrage', label: 'Arbitrage' }
];

document.addEventListener('DOMContentLoaded', function() {
    populateStrategyDropdown(availableStrategies);
});

// ========== GLOBAL UTILITIES ==========

window.DashboardUtils = {
    formatCurrency: (amount, currency = 'USD') => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(amount);
    },

    formatPercentage: (value, decimals = 2) => {
        return `${(value * 100).toFixed(decimals)}%`;
    },

    formatNumber: (value, decimals = 2) => {
        if (value >= 1e9) {
            return `${(value / 1e9).toFixed(decimals)}B`;
        } else if (value >= 1e6) {
            return `${(value / 1e6).toFixed(decimals)}M`;
        } else if (value >= 1e3) {
            return `${(value / 1e3).toFixed(decimals)}K`;
        }
        return value.toFixed(decimals);
    },

    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    throttle: (func, limit) => {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    // Color utility for charts and indicators
    getStatusColor: (status) => {
        const colors = {
            success: '#28a745',
            warning: '#ffc107',
            danger: '#dc3545',
            info: '#17a2b8',
            primary: '#007bff',
            secondary: '#6c757d'
        };
        return colors[status] || colors.secondary;
    },

    // Generate random ID for dynamic elements
    generateId: (prefix = 'id') => {
        return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    },

    // Local storage utilities
    storage: {
        set: (key, value) => {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                console.error('Storage set error:', e);
                return false;
            }
        },
        
        get: (key, defaultValue = null) => {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.error('Storage get error:', e);
                return defaultValue;
            }
        },
        
        remove: (key) => {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (e) {
                console.error('Storage remove error:', e);
                return false;
            }
        }
    },

    // Validation utilities
    validators: {
        isEmail: (email) => {
            const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return re.test(email);
        },
        
        isNumeric: (value) => {
            return !isNaN(parseFloat(value)) && isFinite(value);
        },
        
        isPositive: (value) => {
            return window.DashboardUtils.validators.isNumeric(value) && parseFloat(value) > 0;
        },
        
        inRange: (value, min, max) => {
            const num = parseFloat(value);
            return window.DashboardUtils.validators.isNumeric(value) && num >= min && num <= max;
        }
    }
};

// Enhanced global functions for console debugging
window.DebugUtils = {
    // Test strategy dropdown
    testStrategyDropdown: () => {
        const select = document.getElementById('strategy-select');
        console.log('Strategy dropdown element:', select);
        console.log('Options count:', select ? select.options.length : 'N/A');
        if (select && select.options.length > 0) {
            console.log('Available strategies:');
            Array.from(select.options).forEach((option, index) => {
                console.log(`  ${index}: ${option.value} - ${option.text}`);
            });
        }
    },

    // Force reload strategies
    reloadStrategies: () => {
        if (window.dashboard) {
            console.log('🔄 Force reloading strategies...');
            window.dashboard.loadAvailableStrategies();
        } else {
            console.error('❌ Dashboard not available');
        }
    },

    // Clear all cache
    clearCache: () => {
        if (window.dashboardEnhancements) {
            window.dashboardEnhancements.clearCache();
        } else {
            console.warn('⚠️ Dashboard enhancements not available');
        }
    },

    // Get cache status
    getCacheStatus: () => {
        if (window.dashboardEnhancements) {
            console.table(window.dashboardEnhancements.getCacheStatus());
        } else {
            console.warn('⚠️ Dashboard enhancements not available');
        }
    },

    // Test all major elements
    testElements: () => {
        const importantIds = [
            'strategy-select', 'start-trading', 'stop-trading', 'chat-input',
            'market-currency', 'crypto-price-grid', 'chat-messages', 'loadingOverlay'
        ];
        
        console.log('🔍 Testing important elements:');
        importantIds.forEach(id => {
            const element = document.getElementById(id);
            console.log(`  ${id}: ${element ? '✅ Found' : '❌ Missing'}`);
        });
    }
};

// Initialize enhancements when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.dashboardEnhancements = new DashboardEnhancements();
    });
} else {
    window.dashboardEnhancements = new DashboardEnhancements();
}

console.log('🚀 Enhanced Dashboard Extensions loaded successfully!');