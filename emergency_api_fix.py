# File: E:\Trade Chat Bot\G Trading Bot\emergency_api_fix.py
"""
EMERGENCY FIX for 404 Market Data API Error
Run this script to immediately fix the dashboard 404 error

This creates a standalone API server that provides the missing /api/market-data endpoint
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from datetime import datetime
import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Elite Trading Bot V3.0 - Emergency API Fix",
    description="Emergency fix for missing market data API",
    version="1.0.0"
)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to mount static files and templates if they exist
try:
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
        logger.info("✅ Static files mounted")
    else:
        logger.warning("⚠️  Static directory not found")
        
    if os.path.exists("templates"):
        templates = Jinja2Templates(directory="templates")
        logger.info("✅ Templates configured")
    else:
        logger.warning("⚠️  Templates directory not found")
        templates = None
except Exception as e:
    logger.error(f"❌ Error setting up static/templates: {e}")
    templates = None

# EMERGENCY FIX: Market Data API Endpoint
@app.get("/api/market-data")
async def get_market_data(currency: str = "USD"):
    """
    EMERGENCY FIX: Market data endpoint that was missing and causing 404 errors
    This provides fallback data to fix the dashboard immediately
    """
    logger.info(f"🔧 EMERGENCY API called for currency: {currency}")
    
    # Simulate real-time data with small variations
    import random
    base_time = datetime.now()
    
    # Base prices with small random variations to simulate real-time updates
    base_data = {
        "BTC": {"base_price": 97500.00, "base_change": 2.5, "volume": 28000000000, "market_cap": 1920000000000, "rank": 1, "name": "Bitcoin"},
        "ETH": {"base_price": 2720.00, "base_change": 1.8, "volume": 15000000000, "market_cap": 327000000000, "rank": 2, "name": "Ethereum"},
        "USDT": {"base_price": 1.00, "base_change": 0.1, "volume": 45000000000, "market_cap": 140000000000, "rank": 3, "name": "Tether"},
        "SOL": {"base_price": 205.00, "base_change": -0.5, "volume": 2500000000, "market_cap": 96000000000, "rank": 4, "name": "Solana"},
        "BNB": {"base_price": 575.00, "base_change": 0.8, "volume": 1800000000, "market_cap": 83000000000, "rank": 5, "name": "BNB"},
        "XRP": {"base_price": 0.52, "base_change": 3.2, "volume": 2100000000, "market_cap": 29000000000, "rank": 6, "name": "XRP"},
        "USDC": {"base_price": 1.00, "base_change": 0.0, "volume": 8500000000, "market_cap": 34000000000, "rank": 7, "name": "USD Coin"},
        "DOGE": {"base_price": 0.08, "base_change": -1.2, "volume": 850000000, "market_cap": 12000000000, "rank": 8, "name": "Dogecoin"},
        "ADA": {"base_price": 0.35, "base_change": 1.5, "volume": 400000000, "market_cap": 12500000000, "rank": 9, "name": "Cardano"},
        "AVAX": {"base_price": 25.50, "base_change": 2.1, "volume": 350000000, "market_cap": 10400000000, "rank": 10, "name": "Avalanche"}
    }
    
    # Add small random variations to simulate real-time updates
    symbols = {}
    for symbol, data in base_data.items():
        price_variation = random.uniform(-0.02, 0.02)  # ±2% variation
        change_variation = random.uniform(-0.5, 0.5)   # ±0.5% change variation
        
        symbols[symbol] = {
            "price": round(data["base_price"] * (1 + price_variation), 2),
            "change": round(data["base_change"] + change_variation, 2),
            "volume": data["volume"],
            "market_cap": data["market_cap"],
            "rank": data["rank"],
            "name": data["name"]
        }
    
    # Calculate market overview
    total_market_cap = sum(data["market_cap"] for data in symbols.values())
    total_volume = sum(data["volume"] for data in symbols.values())
    btc_market_cap = symbols["BTC"]["market_cap"]
    btc_dominance = (btc_market_cap / total_market_cap * 100) if total_market_cap > 0 else 62.5
    
    avg_change = sum(data["change"] for data in symbols.values()) / len(symbols)
    sentiment = "Bullish" if avg_change > 1 else "Bearish" if avg_change < -1 else "Neutral"
    
    response = {
        "success": True,
        "timestamp": base_time.isoformat(),
        "currency": currency.upper(),
        "source": "emergency_fix_api",
        "message": "Emergency API active - Dashboard should work now!",
        "symbols": symbols,
        "market_overview": {
            "total_market_cap": total_market_cap,
            "btc_dominance": round(btc_dominance, 1),
            "market_sentiment": sentiment,
            "total_volume_24h": total_volume
        }
    }
    
    logger.info(f"✅ Emergency API response sent successfully for {currency}")
    return response

@app.get("/api/market-data/health")
async def market_data_health():
    """Health check for the emergency API"""
    return {
        "status": "healthy",
        "service": "emergency_market_data_api",
        "timestamp": datetime.now().isoformat(),
        "message": "Emergency fix is active and working",
        "fix_version": "1.0.0"
    }

# Serve the dashboard if template exists
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve the main dashboard"""
    try:
        if templates and os.path.exists("templates/dashboard.html"):
            return templates.TemplateResponse("dashboard.html", {
                "request": request,
                "title": "Elite Trading Bot V3.0 - Emergency Fix Active",
                "ml_status": True
            })
        else:
            # Provide a simple HTML response if dashboard.html doesn't exist
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Elite Trading Bot V3.0 - Emergency Fix</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: white; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .status {{ background: #00d4aa; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                    .api-test {{ background: #2a2d3a; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                    code {{ background: #444; padding: 5px; border-radius: 3px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🔧 Elite Trading Bot V3.0 - Emergency Fix Active</h1>
                    
                    <div class="status">
                        <h2>✅ Emergency API Fix is Running</h2>
                        <p>The missing <code>/api/market-data</code> endpoint has been restored!</p>
                        <p>Your dashboard should now work without 404 errors.</p>
                    </div>
                    
                    <div class="api-test">
                        <h3>🧪 Test API Endpoints:</h3>
                        <ul>
                            <li><a href="/api/market-data" style="color: #00d4aa;">Market Data API</a></li>
                            <li><a href="/api/market-data/health" style="color: #00d4aa;">API Health Check</a></li>
                        </ul>
                    </div>
                    
                    <div class="api-test">
                        <h3>📊 Next Steps:</h3>
                        <ol>
                            <li>Your dashboard is now accessible (if dashboard.html exists)</li>
                            <li>The market data API is providing fallback data</li>
                            <li>To integrate real market data, replace this emergency fix with your full API</li>
                            <li>Check the console logs for any remaining issues</li>
                        </ol>
                    </div>
                    
                    <p><small>Emergency Fix Server - Version 1.0.0 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
                </div>
            </body>
            </html>
            """, status_code=200)
    except Exception as e:
        logger.error(f"❌ Error serving dashboard: {e}")
        return HTMLResponse(content=f"""
        <h1>Dashboard Error</h1>
        <p>Error: {e}</p>
        <p>But the API fix is still active at <a href="/api/market-data">/api/market-data</a></p>
        """, status_code=500)

# Generic health check
@app.get("/health")
async def health_check():
    """General health check"""
    return {
        "status": "healthy",
        "service": "elite_trading_bot_emergency_fix",
        "timestamp": datetime.now().isoformat(),
        "message": "Emergency fix server is running",
        "apis_available": ["/api/market-data", "/api/market-data/health"]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    logger.warning(f"404 Error: {request.url}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested URL {request.url.path} was not found",
            "available_endpoints": ["/api/market-data", "/api/market-data/health", "/health"],
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    print("🚨 EMERGENCY API FIX SERVER")
    print("=" * 50)
    print("🎯 Purpose: Fix the 404 /api/market-data error")
    print("🔧 This will restore your dashboard functionality")
    print("📊 Provides fallback market data")
    print("")
    print("🚀 Starting server...")
    print("📍 Dashboard: http://localhost:8000/")
    print("🔌 API Test: http://localhost:8000/api/market-data")
    print("💚 Health: http://localhost:8000/health")
    print("")
    print("⚠️  This is a temporary fix. Integrate with your main app when ready.")
    print("=" * 50)
    
    # Get configuration from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    try:
        uvicorn.run(
            "emergency_api_fix:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Emergency fix server stopped.")
        print("✅ Your dashboard should now work when you run your main app!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Try changing the port: python emergency_api_fix.py")
        sys.exit(1)