# 🚀 Elite Trading Bot V3.0 - Industrial Dashboard

## Overview
Professional-grade cryptocurrency trading dashboard with advanced features including:
- Real-time market data for top 10 cryptocurrencies
- Advanced trading controls and strategy management
- Machine learning model training and management
- Gemini AI-powered trading assistant
- Industrial-grade UI with dark theme
- Real-time WebSocket updates

## Quick Start

### 1. Installation
```bash
# Install dependencies
pip install fastapi uvicorn google-generativeai psutil websockets python-multipart jinja2 aiofiles

# Or use the deployment script
python deploy_industrial_dashboard.py
```

### 2. Configuration
1. Copy `.env.example` to `.env`
2. Add your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   KRAKEN_API_KEY=your_kraken_api_key_here
   KRAKEN_SECRET=your_kraken_secret_here
   ```

### 3. Launch
```bash
# Windows
start_dashboard.bat

# Linux/Mac
./start_dashboard.sh

# Or manually
python main.py
```

### 4. Access
Open your browser to: http://localhost:8000

## Features

### 📊 Overview Dashboard
- Portfolio summary with real-time P&L
- Active trading strategies overview
- Performance metrics and statistics
- Market summary with top cryptocurrencies

### 💹 Trading Control Center
- Start/stop/pause trading operations
- Strategy deployment and management
- Real-time position monitoring
- Risk management controls

### 🧠 ML Training Center
- Train multiple model types (Lorentzian, Neural Networks, XGBoost)
- Real-time training progress with WebSocket updates
- Model management and deployment
- Performance metrics and accuracy tracking

### 💰 Live Market Data
- Top 10 cryptocurrencies with real-time prices
- Market overview and sentiment analysis
- Customizable currency display (USD, EUR, BTC)
- Interactive price cards with 24h changes

### 🤖 AI Assistant (Gemini)
- Intelligent trading advice and market analysis
- Natural language queries about your portfolio
- Quick action buttons for common questions
- Real-time market context awareness

### ⚙️ Settings & Configuration
- API key management for exchanges
- System preferences and notifications
- Theme customization
- Performance tuning options

## API Endpoints

### Trading
- `POST /api/trading/start` - Start trading operations
- `POST /api/trading/stop` - Stop trading operations
- `POST /api/strategies/deploy` - Deploy a trading strategy
- `GET /api/strategies/available` - Get available strategies
- `GET /api/strategies/active` - Get active strategies

### Market Data
- `GET /api/market-data` - Get live cryptocurrency prices
- `GET /api/market-overview` - Get comprehensive market overview
- `GET /api/trading-pairs` - Get available trading pairs

### Machine Learning
- `POST /api/ml/train/{model_type}` - Start training a model
- `GET /api/ml/models` - Get available models
- `GET /api/ml/status` - Get ML system status

### Chat & AI
- `POST /api/chat/gemini` - Chat with Gemini AI assistant
- `POST /api/chat` - Legacy chat endpoint

### System
- `GET /health` - System health check
- `GET /api/system/diagnostics` - Detailed system diagnostics
- `WebSocket /ws/notifications` - Real-time updates

## WebSocket Events

The dashboard uses WebSocket connections for real-time updates:

- `market_update` - Live price updates
- `trading_status_update` - Trading operation status changes
- `training_progress` - ML training progress updates
- `portfolio_update` - Portfolio value changes
- `system_health` - System performance metrics

## Security

- CORS protection configured
- Rate limiting (120 requests/minute per IP)
- Input validation and sanitization
- Secure WebSocket connections
- Environment variable configuration

## Performance

- Optimized for 60fps UI updates
- Efficient WebSocket message handling
- Lazy loading of dashboard sections
- Responsive design for mobile devices
- Background task management

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Gemini AI not working**
   - Ensure `GEMINI_API_KEY` is set in `.env`
   - Check API key permissions and quota

3. **WebSocket connection failed**
   - Check firewall settings
   - Ensure port 8000 is not blocked

4. **Market data not loading**
   - Check internet connection
   - CoinGecko API may be rate-limited

### Logs
Check the console output when running `python main.py` for detailed error messages.

## License
Elite Trading Bot V3.0 - Industrial Dashboard
