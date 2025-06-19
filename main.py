# main.py - Enhanced with FreqTrade-style features
import asyncio
import logging
import os
import sys
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Local imports
from core.config import settings, ConfigManager
from core.notification_manager import SimpleNotificationManager
from core.trading_engine import IndustrialTradingEngine
from strategies.strategy_base import StrategyBase
from ai.chat_manager import ChatManager

# Enhanced imports (gracefully handle missing components)
try:
    from core.database import DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    
try:
    from core.risk_manager import RiskManager
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False

try:
    from core.backtester import Backtester
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

try:
    from core.optimizer import StrategyOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# --- Pydantic Models for Enhanced API ---
class StrategyConfig(BaseModel):
    id: str
    type: str
    config: Dict[str, Any] = {}
    symbols: List[str] = []
    enabled: bool = True

class TradeRequest(BaseModel):
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    strategy_id: Optional[str] = None

class BacktestRequest(BaseModel):
    strategy_type: str
    config: Dict[str, Any]
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float = 10000

class OptimizationRequest(BaseModel):
    strategy_type: str
    symbol: str
    start_date: str
    end_date: str
    parameters: Dict[str, List[Union[int, float]]]
    initial_balance: float = 10000

class NotificationRequest(BaseModel):
    message: str
    title: str = "Custom Notification"
    priority: str = "INFO"

# --- Global Instances (initialized during lifespan) ---
notification_manager_instance: Optional[SimpleNotificationManager] = None
trading_engine_instance: Optional[IndustrialTradingEngine] = None
chat_manager_instance: Optional[ChatManager] = None
config_manager_instance: Optional[ConfigManager] = None
database_manager_instance: Optional[Any] = None
risk_manager_instance: Optional[Any] = None
backtester_instance: Optional[Any] = None
optimizer_instance: Optional[Any] = None

# --- Logger Setup ---
logger = logging.getLogger("crypto-bot")
logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

log_file_path = os.path.join(os.getcwd(), 'logs', 'trading_bot.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.info("Enhanced main application logger configured.")

# --- FastAPI Application Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Enhanced lifespan with configuration management and optional components.
    """
    global notification_manager_instance
    global trading_engine_instance
    global chat_manager_instance
    global config_manager_instance
    global database_manager_instance
    global risk_manager_instance
    global backtester_instance
    global optimizer_instance

    logger.info("Enhanced application starting up...")
    
    try:
        # 1. Initialize Configuration Manager
        config_manager_instance = ConfigManager()
        
        # Debug information
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Config file exists: {os.path.exists('config.json')}")
        if os.path.exists('config.json'):
            logger.info(f"Config file size: {os.path.getsize('config.json')} bytes")
            # Test if we can read the file
            try:
                with open('config.json', 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.info(f"Config file content length: {len(content)} characters")
                    logger.info(f"First 100 characters: {content[:100]}")
            except Exception as read_error:
                logger.error(f"Cannot read config.json: {read_error}")
        else:
            logger.error("config.json file does not exist!")
        
        # Try to load the config
        try:
            config = config_manager_instance.load_config("config.json")
            logger.info("Configuration loaded successfully!")
        except Exception as config_error:
            logger.error(f"Config loading failed: {config_error}")
            # Fallback to default config
            config = {
                "trading": {"dry_run": True, "max_open_trades": 3},
                "exchange": {"enabled": False},
                "strategy": {"name": "MLStrategy"}
            }
            logger.info("Using fallback default configuration")
        
        logger.info("Configuration management system initialized")
        
        # 2. Initialize Database (if available)
        if DATABASE_AVAILABLE:
            try:
                database_manager_instance = DatabaseManager(config.get('database', {}))
                await database_manager_instance.initialize()
                logger.info("Database system initialized")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
                database_manager_instance = None
        
        # 3. Initialize Notification Manager
        notification_manager_instance = SimpleNotificationManager()
        
        # 4. Initialize Risk Manager (if available)
        if RISK_MANAGEMENT_AVAILABLE:
            try:
                risk_manager_instance = RiskManager(config.get('risk_management', {}))
                logger.info("Risk management system initialized")
            except Exception as e:
                logger.warning(f"Risk manager initialization failed: {e}")
                risk_manager_instance = None
        
        # 5. Initialize Trading Engine with enhanced config
        trading_engine_instance = IndustrialTradingEngine(
            notification_manager_instance, 
            config=config,
            database_manager=database_manager_instance,
            risk_manager=risk_manager_instance
        )
        
        # 6. Initialize Enhanced Components
        if BACKTESTING_AVAILABLE:
            try:
                backtester_instance = Backtester(config.get('backtesting', {}))
                logger.info("Backtesting system initialized")
            except Exception as e:
                logger.warning(f"Backtester initialization failed: {e}")
                backtester_instance = None
                
        if OPTIMIZATION_AVAILABLE:
            try:
                optimizer_instance = StrategyOptimizer(config.get('optimization', {}))
                logger.info("Strategy optimization system initialized")
            except Exception as e:
                logger.warning(f"Optimizer initialization failed: {e}")
                optimizer_instance = None
        
        # 7. Initialize Chat Manager
        chat_manager_instance = ChatManager(trading_engine_instance)
        
        logger.info("All systems initialized successfully")
        
        # Start the trading engine
        logger.info("Starting enhanced trading engine...")
        await trading_engine_instance.start()
        
    except Exception as e:
        logger.critical(f"Critical startup failure: {e}", exc_info=True)
        if notification_manager_instance:
            await notification_manager_instance.notify(
                "CRITICAL STARTUP ERROR",
                f"Enhanced bot failed to start: {e}. Check logs.",
                "EMERGENCY"
            )
        sys.exit(1)

    yield
    
    # Shutdown
    logger.info("Enhanced application shutting down...")
    if trading_engine_instance and trading_engine_instance.running:
        await trading_engine_instance.stop()
    if database_manager_instance:
        await database_manager_instance.close()
    logger.info("Enhanced application shutdown complete.")

# Create the FastAPI app instance
app = FastAPI(
    lifespan=lifespan, 
    title=f"{settings.APP_NAME} - Enhanced", 
    version="4.0",
    description="Industrial-grade crypto trading bot with FreqTrade compatibility"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Core API Endpoints (Backward Compatible) ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Enhanced dashboard with feature availability indicators."""
    features = {
        "database": DATABASE_AVAILABLE and database_manager_instance is not None,
        "risk_management": RISK_MANAGEMENT_AVAILABLE and risk_manager_instance is not None,
        "backtesting": BACKTESTING_AVAILABLE and backtester_instance is not None,
        "optimization": OPTIMIZATION_AVAILABLE and optimizer_instance is not None
    }
    
    context = {
        "request": request,
        "app_name": f"{settings.APP_NAME} Enhanced",
        "user_id": settings.APP_USER_ID,
        "features": features,
        "version": "4.0"
    }
    return templates.TemplateResponse("index.html", context)

@app.get("/api/health")
async def health_check():
    """Enhanced health check with component status."""
    components = {
        "trading_engine": trading_engine_instance and trading_engine_instance.running,
        "database": database_manager_instance is not None,
        "risk_manager": risk_manager_instance is not None,
        "backtester": backtester_instance is not None,
        "optimizer": optimizer_instance is not None
    }
    
    if trading_engine_instance and trading_engine_instance.running:
        return {
            "status": "healthy",
            "message": "Enhanced trading bot is fully operational",
            "components": components,
            "timestamp": datetime.utcnow().isoformat()
        }
    elif trading_engine_instance:
        return {
            "status": "initialized",
            "message": "Trading bot initialized but not running",
            "components": components
        }
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.get("/api/status/enhanced")
async def get_enhanced_status():
    """Comprehensive system status with all components."""
    if not trading_engine_instance:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    status = trading_engine_instance.get_status()
    
    # Add enhanced component status
    status.update({
        "enhanced_features": {
            "database_connected": database_manager_instance is not None,
            "risk_management_active": risk_manager_instance is not None,
            "backtesting_available": backtester_instance is not None,
            "optimization_available": optimizer_instance is not None,
            "config_management": config_manager_instance is not None
        },
        "system_info": {
            "version": "4.0",
            "freqtrade_compatible": True,
            "uptime": (datetime.utcnow() - datetime.utcnow()).total_seconds() if trading_engine_instance.running else 0
        }
    })
    
    return status

# --- Original Endpoints (Maintained for Backward Compatibility) ---

@app.get("/api/market-data")
async def get_all_market_data():
    """Retrieves the latest market data for all tracked symbols."""
    if trading_engine_instance:
        if trading_engine_instance.current_market_data:
            return trading_engine_instance.current_market_data
        raise HTTPException(status_code=404, detail="No market data available yet")
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.get("/api/market-data/{symbol}")
async def get_market_data_for_symbol(symbol: str):
    """Retrieves the latest market data for a given symbol."""
    if trading_engine_instance:
        data = trading_engine_instance.get_market_data(symbol.upper())
        if data:
            return data
        raise HTTPException(status_code=404, detail=f"Market data for {symbol.upper()} not found")
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.get("/api/balances")
async def get_balances():
    """Retrieves current account balances."""
    if trading_engine_instance:
        return trading_engine_instance.balances
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.get("/api/positions")
async def get_positions():
    """Retrieves current open trading positions."""
    if trading_engine_instance:
        return trading_engine_instance.positions
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.get("/api/status")
async def get_bot_status():
    """Retrieves the comprehensive status of the bot."""
    if trading_engine_instance:
        return trading_engine_instance.get_status()
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.get("/api/performance")
async def get_performance():
    """Retrieves the bot's performance metrics."""
    if trading_engine_instance:
        return trading_engine_instance.get_performance_metrics()
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

# --- Strategy Management (Enhanced) ---

@app.get("/api/strategies")
async def get_available_strategies_list():
    """Lists all available strategy types."""
    if trading_engine_instance:
        return trading_engine_instance.list_available_strategies()
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.get("/api/strategies/available")
async def get_available_strategies_explicit():
    """Lists all available strategy types with detailed info."""
    if trading_engine_instance:
        strategies = trading_engine_instance.list_available_strategies()
        # Add FreqTrade compatibility info
        if isinstance(strategies, dict) and 'strategies' in strategies:
            for strategy in strategies.get('strategies', []):
                strategy['freqtrade_compatible'] = True
                strategy['enhanced_features'] = True
        return strategies
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.get("/api/strategies/active")
async def get_active_strategies_list():
    """Lists all currently active strategy instances."""
    if trading_engine_instance:
        return trading_engine_instance.list_active_strategies()
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.post("/api/strategies/add")
async def add_strategy_route(strategy_data: StrategyConfig):
    """Adds a new strategy instance with enhanced validation."""
    if not trading_engine_instance:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    # Enhanced validation with risk management
    if risk_manager_instance:
        validation_result = risk_manager_instance.validate_strategy_config(strategy_data.dict())
        if not validation_result.get('valid', True):
            raise HTTPException(status_code=400, detail=validation_result.get('error', 'Strategy validation failed'))
    
    if trading_engine_instance.add_strategy(strategy_data.id, strategy_data.type, strategy_data.config):
        return {
            "status": "success",
            "message": f"Enhanced strategy {strategy_data.id} ({strategy_data.type}) added",
            "strategy_id": strategy_data.id
        }
    raise HTTPException(status_code=409, detail=f"Strategy '{strategy_data.id}' already exists or validation failed")

@app.delete("/api/strategies/remove/{strategy_id}")
async def remove_strategy_route(strategy_id: str):
    """Removes an active strategy instance."""
    if not trading_engine_instance:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    if trading_engine_instance.remove_strategy(strategy_id):
        return {"status": "success", "message": f"Strategy {strategy_id} removed"}
    raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")

# --- Enhanced Trading Endpoints ---

@app.post("/api/trade")
async def place_trade_route(trade_data: TradeRequest):
    """Enhanced trade placement with risk management."""
    if not trading_engine_instance:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")

    # Risk management validation
    if risk_manager_instance:
        risk_check = risk_manager_instance.validate_trade(trade_data.dict())
        if not risk_check.get('approved', True):
            raise HTTPException(status_code=400, detail=f"Trade rejected by risk management: {risk_check.get('reason')}")
    
    result = await trading_engine_instance.place_order(
        trade_data.symbol, 
        trade_data.side, 
        trade_data.qty, 
        trade_data.order_type, 
        trade_data.limit_price,
        strategy_id=trade_data.strategy_id
    )
    
    if result.get("status") == "filled":
        return {"status": "success", "message": "Trade placed successfully", "details": result}
    raise HTTPException(status_code=400, detail=result.get("error", "Failed to place trade"))

@app.get("/api/trades/history")
async def get_trade_history(limit: int = 100, offset: int = 0):
    """Retrieve trade history from database."""
    if not database_manager_instance:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        trades = await database_manager_instance.get_trade_history(limit=limit, offset=offset)
        return {"trades": trades, "total": len(trades)}
    except Exception as e:
        logger.error(f"Error retrieving trade history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trade history")

@app.get("/api/trades/statistics")
async def get_trade_statistics():
    """Get comprehensive trading statistics."""
    if not database_manager_instance:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        stats = await database_manager_instance.get_trade_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error retrieving trade statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

# --- Control Endpoints (Enhanced) ---

@app.post("/api/start")
async def start_trading_route():
    """Start the enhanced trading engine."""
    if not trading_engine_instance:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    if trading_engine_instance.running:
        return {"status": "success", "message": "Enhanced trading engine is already running"}
    
    try:
        await trading_engine_instance.start()
        return {"status": "success", "message": "Enhanced trading engine started successfully"}
    except Exception as e:
        logger.error(f"Failed to start enhanced trading engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start trading engine: {e}")

@app.post("/api/stop")
async def stop_trading_route():
    """Stop the enhanced trading engine."""
    if not trading_engine_instance:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    if not trading_engine_instance.running:
        return {"status": "success", "message": "Enhanced trading engine is already stopped"}
    
    try:
        await trading_engine_instance.stop()
        return {"status": "success", "message": "Enhanced trading engine stopped successfully"}
    except Exception as e:
        logger.error(f"Failed to stop enhanced trading engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop trading engine: {e}")

# --- Notification Endpoints (Enhanced) ---

@app.get("/api/notifications")
async def get_notification_status_route():
    """Get enhanced notification system status."""
    if notification_manager_instance:
        status = notification_manager_instance.get_status_report()
        status.update({
            "enhanced_features": True,
            "version": "4.0"
        })
        return status
    raise HTTPException(status_code=500, detail="Notification manager not initialized")

@app.post("/api/notifications/send")
async def send_custom_notification_route(notification_data: NotificationRequest):
    """Send enhanced custom notification."""
    if notification_manager_instance:
        try:
            await notification_manager_instance.notify(
                notification_data.title, 
                notification_data.message, 
                notification_data.priority
            )
            return {"status": "success", "message": "Enhanced notification sent successfully"}
        except Exception as e:
            logger.error(f"Error sending enhanced notification: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to send notification: {e}")
    raise HTTPException(status_code=500, detail="Notification manager not initialized")

@app.post("/api/notifications/test")
async def test_all_notifications_channels_route():
    """Test all enhanced notification channels."""
    if notification_manager_instance:
        try:
            await notification_manager_instance.notify(
                "Enhanced Test Notification", 
                "This is a test from the enhanced trading bot v4.0", 
                "INFO"
            )
            return {"status": "success", "message": "Enhanced test notifications sent successfully"}
        except Exception as e:
            logger.error(f"Error sending enhanced test notification: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to send test notification: {e}")
    raise HTTPException(status_code=500, detail="Notification manager not initialized")

@app.get("/api/notifications/history")
async def get_notification_history_route():
    """Get enhanced notification history."""
    if notification_manager_instance:
        history = {"notifications": notification_manager_instance.get_notification_history()}
        history.update({
            "enhanced_features": True,
            "total_count": len(history["notifications"])
        })
        return history
    raise HTTPException(status_code=500, detail="Notification manager not initialized")

# --- Chat Endpoint (Enhanced) ---

@app.post("/api/chat")
async def chat_with_bot(message: Dict[str, Any]):
    """Enhanced chat with AI assistant."""
    user_message = message.get("message")
    if not user_message:
        raise HTTPException(status_code=400, detail="Message content is empty")
    
    if not chat_manager_instance:
        raise HTTPException(status_code=500, detail="Chat manager not initialized")
    
    try:
        response = await chat_manager_instance.process_chat_message(user_message)
        response.update({
            "enhanced_features": True,
            "version": "4.0",
            "timestamp": datetime.utcnow().isoformat()
        })
        return response
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {e}")

# --- WebSocket (Enhanced) ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket with proper chat message handling."""
    await websocket.accept()
    logger.info("Enhanced WebSocket client connected")
    
    try:
        # Send welcome message with capabilities
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to Enhanced Trading Bot v4.0",
            "features": {
                "database": database_manager_instance is not None,
                "risk_management": risk_manager_instance is not None,
                "backtesting": backtester_instance is not None,
                "optimization": optimizer_instance is not None
            }
        })
        
        while True:
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                message_type = data.get("type", "echo")
                
                if message_type == "chat":
                    # Handle chat messages properly
                    user_message = data.get("message", "")
                    if user_message and chat_manager_instance:
                        try:
                            logger.info(f"Processing WebSocket chat message: {user_message}")
                            chat_response = await chat_manager_instance.process_chat_message(user_message)
                            await websocket.send_json({
                                "type": "chat_response",
                                "message": chat_response.get("response", "No response"),
                                "action": chat_response.get("action", {})
                            })
                        except Exception as e:
                            logger.error(f"Chat processing error: {e}")
                            await websocket.send_json({
                                "type": "chat_response", 
                                "message": f"Chat error: {str(e)}",
                                "error": True
                            })
                    else:
                        await websocket.send_json({
                            "type": "chat_response",
                            "message": "Chat manager not available or empty message",
                            "error": True
                        })
                        
                elif message_type == "status":
                    # Send current status
                    if trading_engine_instance:
                        status = trading_engine_instance.get_status()
                        await websocket.send_json({"type": "status", "data": status})
                    else:
                        await websocket.send_json({
                            "type": "status", 
                            "error": "Trading engine not initialized"
                        })
                        
                elif message_type == "market_data":
                    # Send market data
                    if trading_engine_instance and trading_engine_instance.current_market_data:
                        await websocket.send_json({
                            "type": "market_data", 
                            "data": trading_engine_instance.current_market_data
                        })
                    else:
                        await websocket.send_json({
                            "type": "market_data",
                            "error": "No market data available"
                        })
                else:
                    # Echo back for testing
                    await websocket.send_json({"type": "echo", "message": message})
                    
            except json.JSONDecodeError:
                # Handle plain text messages as chat
                if message.strip() and chat_manager_instance:
                    try:
                        logger.info(f"Processing plain text WebSocket message: {message}")
                        chat_response = await chat_manager_instance.process_chat_message(message.strip())
                        await websocket.send_json({
                            "type": "chat_response",
                            "message": chat_response.get("response", "No response"),
                            "action": chat_response.get("action", {})
                        })
                    except Exception as e:
                        logger.error(f"Plain text chat processing error: {e}")
                        await websocket.send_text(f"Chat error: {str(e)}")
                else:
                    # Echo back plain text
                    await websocket.send_text(f"Enhanced Echo: {message}")
            
    except WebSocketDisconnect:
        logger.info("Enhanced WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Enhanced WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"WebSocket error: {str(e)}"
            })
        except:
            pass  # Connection might be closed

# --- System Information Endpoint ---

@app.get("/api/system/info")
async def get_system_info():
    """Get comprehensive system information."""
    return {
        "version": "4.0",
        "name": f"{settings.APP_NAME} Enhanced",
        "freqtrade_compatible": True,
        "features": {
            "database": DATABASE_AVAILABLE and database_manager_instance is not None,
            "risk_management": RISK_MANAGEMENT_AVAILABLE and risk_manager_instance is not None,
            "backtesting": BACKTESTING_AVAILABLE and backtester_instance is not None,
            "optimization": OPTIMIZATION_AVAILABLE and optimizer_instance is not None,
            "configuration_management": config_manager_instance is not None,
            "enhanced_notifications": True,
            "websocket_streaming": True,
            "ai_chat": chat_manager_instance is not None
        },
        "components": {
            "trading_engine": "IndustrialTradingEngine v4.0",
            "notification_system": "SimpleNotificationManager Enhanced",
            "ai_assistant": "ChatManager v4.0",
            "database": "DatabaseManager" if database_manager_instance else "Not Available",
            "risk_management": "RiskManager" if risk_manager_instance else "Not Available",
            "backtesting": "Backtester" if backtester_instance else "Not Available",
            "optimization": "StrategyOptimizer" if optimizer_instance else "Not Available"
        },
        "api_endpoints": {
            "total": len([route for route in app.routes]),
            "enhanced_endpoints": [
                "/api/status/enhanced",
                "/api/trades/history",
                "/api/trades/statistics", 
                "/api/system/info"
            ]
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Main entry point
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)