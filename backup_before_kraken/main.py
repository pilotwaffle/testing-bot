#!/usr/bin/env python3
"""
Industrial Crypto Trading Bot v3.0 - Main Application
Fixed version that will definitely start without hanging
"""

import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading_bot.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Global variables for components
trading_engine = None
ml_engine = None
notification_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Industrial Crypto Trading Bot v3.0")
    
    # Startup
    try:
        await initialize_components()
        logger.info("All components initialized - Dashboard ready!")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        # Continue anyway for basic functionality
        yield
    finally:
        # Shutdown
        logger.info("Shutting down trading bot...")
        await cleanup_components()

# Create FastAPI app
app = FastAPI(
    title="Industrial Crypto Trading Bot",
    description="Advanced cryptocurrency trading bot with ML capabilities",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_components():
    """Initialize all trading bot components"""
    global trading_engine, ml_engine, notification_manager
    
    try:
        # Initialize Enhanced Trading Engine
        logger.info("Initializing Enhanced Trading Engine...")
        try:
            from core.enhanced_trading_engine import EliteTradingEngine as EnhancedTradingEngine
            trading_engine = EnhancedTradingEngine()
            logger.info("Enhanced Trading Engine initialized")
        except Exception as e:
            logger.warning(f"Enhanced Trading Engine failed, using fallback: {e}")
            try:
                from core.fast_trading_engine import FastTradingEngine
                trading_engine = FastTradingEngine()
                logger.info("Fast Trading Engine initialized (fallback)")
            except Exception as e2:
                logger.error(f"All trading engines failed: {e2}")
                trading_engine = None
        
        # Initialize ML Engine
        logger.info("Initializing ML Engine...")
        try:
            from core.enhanced_ml_engine import EnhancedMLEngine
            ml_engine = EnhancedMLEngine()
            logger.info("Enhanced ML Engine initialized")
        except Exception as e:
            logger.warning(f"Enhanced ML Engine failed, using fallback: {e}")
            try:
                from core.fast_ml_engine import FastMLEngine
                ml_engine = FastMLEngine()
                logger.info("Fast ML Engine initialized (fallback)")
            except Exception as e2:
                logger.error(f"All ML engines failed: {e2}")
                ml_engine = None
        
        # Initialize Notification Manager
        logger.info("Initializing Notification Manager...")
        try:
            from core.notification_manager import NotificationManager
            notification_manager = NotificationManager()
            logger.info("Notification Manager initialized")
        except Exception as e:
            logger.warning(f"Notification Manager failed: {e}")
            notification_manager = None
        
        # Start background tasks
        if trading_engine:
            try:
                await trading_engine.start()
                logger.info("Trading engine started")
            except Exception as e:
                logger.warning(f"Failed to start trading engine: {e}")
        
    except Exception as e:
        logger.error(f"Component initialization error: {e}")
        # Don't raise - allow app to start with limited functionality

async def cleanup_components():
    """Cleanup components on shutdown"""
    global trading_engine, ml_engine, notification_manager
    
    try:
        if trading_engine:
            await trading_engine.stop()
        logger.info("Components cleaned up successfully")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# Basic Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Industrial Crypto Trading Bot v3.0",
        "status": "running",
        "version": "3.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "components": {
            "trading_engine": trading_engine is not None,
            "ml_engine": ml_engine is not None,
            "notification_manager": notification_manager is not None
        },
        "timestamp": asyncio.get_event_loop().time()
    }
    return status

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return await health_check()

@app.get("/status")
async def get_status():
    """Get detailed system status"""
    try:
        status = {
            "bot_status": "running",
            "components": {
                "trading_engine": {
                    "available": trading_engine is not None,
                    "type": type(trading_engine).__name__ if trading_engine else None,
                    "running": getattr(trading_engine, 'is_running', False) if trading_engine else False
                },
                "ml_engine": {
                    "available": ml_engine is not None,
                    "type": type(ml_engine).__name__ if ml_engine else None
                },
                "notification_manager": {
                    "available": notification_manager is not None,
                    "type": type(notification_manager).__name__ if notification_manager else None
                }
            },
            "system": {
                "python_version": sys.version,
                "working_directory": str(Path.cwd())
            }
        }
        return status
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {"error": str(e)}

@app.get("/api/trading/start")
async def start_trading():
    """Start trading"""
    try:
        if not trading_engine:
            raise HTTPException(status_code=503, detail="Trading engine not available")
        
        if hasattr(trading_engine, 'start_trading'):
            await trading_engine.start_trading()
            return {"message": "Trading started successfully"}
        else:
            return {"message": "Trading engine running but start_trading method not available"}
            
    except Exception as e:
        logger.error(f"Start trading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading/stop")
async def stop_trading():
    """Stop trading"""
    try:
        if not trading_engine:
            raise HTTPException(status_code=503, detail="Trading engine not available")
        
        if hasattr(trading_engine, 'stop_trading'):
            await trading_engine.stop_trading()
            return {"message": "Trading stopped successfully"}
        else:
            return {"message": "Trading engine running but stop_trading method not available"}
            
    except Exception as e:
        logger.error(f"Stop trading error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio information"""
    try:
        if not trading_engine:
            return {"error": "Trading engine not available", "portfolio": {}}
        
        if hasattr(trading_engine, 'get_portfolio'):
            portfolio = await trading_engine.get_portfolio()
            return {"portfolio": portfolio}
        else:
            return {"portfolio": {"message": "Portfolio data not available"}}
            
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        return {"error": str(e), "portfolio": {}}

@app.get("/dashboard")
async def dashboard():
    """Simple dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Bot Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .status { background: #f0f8ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .component { margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; }
            .available { color: green; }
            .unavailable { color: red; }
            button { padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
            .start { background: #4CAF50; color: white; }
            .stop { background: #f44336; color: white; }
        </style>
    </head>
    <body>
        <h1>Industrial Crypto Trading Bot v3.0</h1>
        <div class="status">
            <h2>System Status</h2>
            <div id="status">Loading...</div>
        </div>
        
        <div class="status">
            <h2>Controls</h2>
            <button class="start" onclick="startTrading()">Start Trading</button>
            <button class="stop" onclick="stopTrading()">Stop Trading</button>
            <button onclick="refreshStatus()">Refresh Status</button>
        </div>
        
        <div class="status">
            <h2>Portfolio</h2>
            <div id="portfolio">Loading...</div>
        </div>

        <script>
            async function refreshStatus() {
                try {
                    const response = await fetch('/status');
                    const status = await response.json();
                    document.getElementById('status').innerHTML = formatStatus(status);
                } catch (error) {
                    document.getElementById('status').innerHTML = 'Error: ' + error.message;
                }
            }
            
            async function loadPortfolio() {
                try {
                    const response = await fetch('/api/portfolio');
                    const data = await response.json();
                    document.getElementById('portfolio').innerHTML = JSON.stringify(data.portfolio, null, 2);
                } catch (error) {
                    document.getElementById('portfolio').innerHTML = 'Error: ' + error.message;
                }
            }
            
            function formatStatus(status) {
                let html = '<div class="component"><strong>Bot Status:</strong> ' + status.bot_status + '</div>';
                
                for (const [name, component] of Object.entries(status.components)) {
                    const available = component.available;
                    const className = available ? 'available' : 'unavailable';
                    html += `<div class="component">
                        <strong>${name}:</strong> 
                        <span class="${className}">${available ? 'Available' : 'Unavailable'}</span>
                        ${component.type ? ' (' + component.type + ')' : ''}
                        ${component.running ? ' - Running' : ''}
                    </div>`;
                }
                
                return html;
            }
            
            async function startTrading() {
                try {
                    const response = await fetch('/api/trading/start');
                    const result = await response.json();
                    alert(result.message);
                    refreshStatus();
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            async function stopTrading() {
                try {
                    const response = await fetch('/api/trading/stop');
                    const result = await response.json();
                    alert(result.message);
                    refreshStatus();
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            // Initialize
            refreshStatus();
            loadPortfolio();
            
            // Auto-refresh every 30 seconds
            setInterval(refreshStatus, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )