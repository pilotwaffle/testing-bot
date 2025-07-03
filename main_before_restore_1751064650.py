# main.py - Enhanced with Advanced AI Chat Interface Integration
import asyncio
# Startup optimization: Add timeout handling
import asyncio
from functools import wraps

def with_timeout(timeout_seconds=10):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(f"{func.__name__} timed out after {timeout_seconds}s")
                return None
            except Exception as e:
                logger.warning(f"{func.__name__} failed: {e}")
                return None
        return wrapper
    return decorator

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
from utils.chat_manager import EnhancedChatManager

# Local imports
from core.config import settings, ConfigManager
from core.notification_manager import SimpleNotificationManager
from core.trading_engine import IndustrialTradingEngine
from strategies.strategy_base import StrategyBase

# Enhanced AI Chat imports
try:
    from ai.chat_manager import EnhancedChatManager
    ENHANCED_CHAT_AVAILABLE = True
except ImportError:
    try:
        from ai.chat_manager import ChatManager as EnhancedChatManager
        ENHANCED_CHAT_AVAILABLE = False
        logging.warning("Using fallback ChatManager - enhanced features disabled")
    except ImportError:
        ENHANCED_CHAT_AVAILABLE = False
        logging.error("No ChatManager available")

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
    from core.backtesting_engine import BacktestingEngine
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

try:
    from ml.ml_engine import OctoBotMLEngine
    ML_ENGINE_AVAILABLE = True
except ImportError:
    ML_ENGINE_AVAILABLE = False

try:
    from core.data_fetcher import CryptoDataFetcher
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False

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

class EnhancedChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    user_id: Optional[str] = "default"
    context: Optional[Dict[str, Any]] = None

class NotificationRequest(BaseModel):
    message: str
    title: str = "Custom Notification"
    priority: str = "INFO"

# --- Global Instances ---
notification_manager_instance: Optional[SimpleNotificationManager] = None
trading_engine_instance: Optional[IndustrialTradingEngine] = None
enhanced_chat_manager_instance: Optional[EnhancedChatManager] = None
config_manager_instance: Optional[ConfigManager] = None
database_manager_instance: Optional[Any] = None
risk_manager_instance: Optional[Any] = None
ml_engine_instance: Optional[Any] = None
data_fetcher_instance: Optional[Any] = None

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
logger.info("Enhanced main application logger configured with AI chat support.")

# --- Dependency Injection Functions ---
def get_trading_engine_dep():
    """Dependency function to get trading engine instance"""
    if trading_engine_instance is None:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    return trading_engine_instance

def get_ml_engine_dep():
    """Dependency function to get ML engine instance"""
    if not ML_ENGINE_AVAILABLE or ml_engine_instance is None:
        raise HTTPException(status_code=503, detail="ML engine not available")
    return ml_engine_instance

def get_data_fetcher_dep():
    """Dependency function to get data fetcher instance"""
    if not DATA_FETCHER_AVAILABLE or data_fetcher_instance is None:
        raise HTTPException(status_code=503, detail="Data fetcher not available")
    return data_fetcher_instance

def get_notification_manager_dep():
    """Dependency function to get notification manager instance"""
    if notification_manager_instance is None:
        raise HTTPException(status_code=500, detail="Notification manager not initialized")
    return notification_manager_instance

def get_enhanced_chat_manager_dep():
    """Dependency function to get enhanced chat manager instance"""
    if not ENHANCED_CHAT_AVAILABLE or enhanced_chat_manager_instance is None:
        raise HTTPException(status_code=503, detail="Enhanced chat manager not available")
    return enhanced_chat_manager_instance

def get_database_manager_dep():
    """Dependency function to get database manager instance"""
    if not DATABASE_AVAILABLE or database_manager_instance is None:
        raise HTTPException(status_code=503, detail="Database manager not available")
    return database_manager_instance

def get_risk_manager_dep():
    """Dependency function to get risk manager instance"""
    if not RISK_MANAGEMENT_AVAILABLE or risk_manager_instance is None:
        raise HTTPException(status_code=503, detail="Risk manager not available")
    return risk_manager_instance

# --- FastAPI Application Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan with advanced AI chat integration."""
    global notification_manager_instance
    global trading_engine_instance
    global enhanced_chat_manager_instance
    global config_manager_instance
    global database_manager_instance
    global risk_manager_instance
    global ml_engine_instance
    global data_fetcher_instance

    logger.info("Enhanced application starting up with advanced AI chat...")
    
    try:
        # 1. Initialize Configuration Manager
        config_manager_instance = ConfigManager()
        
        # Debug information
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Config file exists: {os.path.exists('config.json')}")
        
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
                database_manager_instance = DatabaseManager(getattr(config,'database', {}))
# OPTIMIZED:                 await database_manager_instance.initialize()
# OPTIMIZED:                 logger.info("Database system initialized")
            except Exception as e:
# OPTIMIZED:                 logger.warning(f"Database initialization failed: {e}")
                database_manager_instance = None
        
        # 3. Initialize ML Engine (if available)
        if ML_ENGINE_AVAILABLE:
            try:
                ml_engine_instance = OctoBotMLEngine()
                logger.info("ML engine initialized")
            except Exception as e:
                logger.warning(f"ML engine initialization failed: {e}")
                ml_engine_instance = None
        
        # 4. Initialize Data Fetcher (if available)
        if DATA_FETCHER_AVAILABLE:
            try:
                data_fetcher_instance = CryptoDataFetcher()
                logger.info("Data fetcher initialized")
            except Exception as e:
                logger.warning(f"Data fetcher initialization failed: {e}")
                data_fetcher_instance = None
        
        # 5. Initialize Notification Manager
        notification_manager_instance = SimpleNotificationManager()
        
        # 6. Initialize Risk Manager (if available)
        if RISK_MANAGEMENT_AVAILABLE:
            try:
                risk_manager_instance = RiskManager(getattr(config,'risk_management', {}))
                logger.info("Risk management system initialized")
            except Exception as e:
                logger.warning(f"Risk manager initialization failed: {e}")
                risk_manager_instance = None
        
        # 7. Initialize Trading Engine with enhanced config
        trading_engine_instance = IndustrialTradingEngine(
            notification_manager_instance, 
            config=config
        )
        
        # 8. Initialize Enhanced Chat Manager
        if ENHANCED_CHAT_AVAILABLE:
            try:
                enhanced_chat_manager_instance = EnhancedChatManager(
                    trading_engine=trading_engine_instance,
                    ml_engine=ml_engine_instance,
                    data_fetcher=data_fetcher_instance,
                    notification_manager=notification_manager_instance
                )
                logger.info("Enhanced AI Chat Manager initialized with advanced features")
            except Exception as e:
                logger.error(f"Enhanced chat manager initialization failed: {e}")
                enhanced_chat_manager_instance = None
        else:
            logger.warning("Enhanced chat features not available - using basic fallback")
        
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

# OPTIMIZED: Fallback components for failed initialization
class FallbackComponent:
    def __init__(self, component_name):
        self.component_name = component_name
        self.logger = logging.getLogger(__name__)
        self.logger.warning(f"Using fallback for {component_name}")
    
    def __getattr__(self, name):
        def fallback_method(*args, **kwargs):
            self.logger.debug(f"Fallback method called: {self.component_name}.{name}")
            return {"status": "fallback", "component": self.component_name, "method": name}
        return fallback_method

app = FastAPI(
    lifespan=lifespan, 
    title=f"{settings.APP_NAME} - Enhanced AI", 
    version="4.0",
    description="Industrial-grade crypto trading bot with advanced AI chat interface"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Include Enhanced Chat Routes ---
try:
    from api.routers.chat_routes import router as enhanced_chat_router
    app.include_router(enhanced_chat_router)
    logger.info("Enhanced chat routes loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced chat routes not available: {e}")

# --- Core API Endpoints (Enhanced) ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Enhanced dashboard with advanced AI chat features."""
    features = {
        "database": DATABASE_AVAILABLE and database_manager_instance is not None,
        "risk_management": RISK_MANAGEMENT_AVAILABLE and risk_manager_instance is not None,
        "backtesting": BACKTESTING_AVAILABLE,
        "ml_engine": ML_ENGINE_AVAILABLE and ml_engine_instance is not None,
        "enhanced_chat": ENHANCED_CHAT_AVAILABLE and enhanced_chat_manager_instance is not None,
        "data_fetcher": DATA_FETCHER_AVAILABLE and data_fetcher_instance is not None,
        "ai_features": {
            "intent_classification": ENHANCED_CHAT_AVAILABLE,
            "proactive_insights": ENHANCED_CHAT_AVAILABLE,
            "command_suggestions": ENHANCED_CHAT_AVAILABLE,
            "voice_interface": True,
            "conversation_memory": ENHANCED_CHAT_AVAILABLE
        }
    }
    
    context = {
        "request": request,
        "app_name": f"{settings.APP_NAME} Enhanced AI",
        "user_id": settings.APP_USER_ID,
        "features": features,
        "version": "4.0",
        "ai_enabled": ENHANCED_CHAT_AVAILABLE
    }
    return templates.TemplateResponse("index.html", context)

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Dedicated enhanced AI chat page."""
    context = {
        "request": request,
        "app_name": f"{settings.APP_NAME} AI Chat",
        "enhanced_features": ENHANCED_CHAT_AVAILABLE,
        "ai_enabled": settings.GOOGLE_AI_ENABLED if hasattr(settings, 'GOOGLE_AI_ENABLED') else False
    }
    return templates.TemplateResponse("chat.html", context)

@app.get("/api/health")
async def health_check():
    """Enhanced health check with AI chat component status."""
    components = {
        "trading_engine": trading_engine_instance and trading_engine_instance.running,
        "database": database_manager_instance is not None,
        "risk_manager": risk_manager_instance is not None,
        "ml_engine": ml_engine_instance is not None,
        "enhanced_chat": enhanced_chat_manager_instance is not None,
        "data_fetcher": data_fetcher_instance is not None
    }
    
    ai_features = {
        "intent_classification": ENHANCED_CHAT_AVAILABLE,
        "conversation_memory": ENHANCED_CHAT_AVAILABLE,
        "proactive_insights": ENHANCED_CHAT_AVAILABLE,
        "voice_interface": True,
        "command_system": ENHANCED_CHAT_AVAILABLE
    }
    
    if trading_engine_instance and trading_engine_instance.running:
        return {
            "status": "healthy",
            "message": "Enhanced trading bot with AI chat is fully operational",
            "components": components,
            "ai_features": ai_features,
            "timestamp": datetime.utcnow().isoformat()
        }
    elif trading_engine_instance:
        return {
            "status": "initialized",
            "message": "Trading bot initialized but not running",
            "components": components,
            "ai_features": ai_features
        }
    raise HTTPException(status_code=500, detail="Trading engine not initialized")

@app.get("/api/status/enhanced")
async def get_enhanced_status():
    """Comprehensive system status including AI chat capabilities."""
    if not trading_engine_instance:
        raise HTTPException(status_code=500, detail="Trading engine not initialized")
    
    # Get base status from trading engine
    if hasattr(trading_engine_instance, 'get_enhanced_status'):
        status = trading_engine_instance.get_enhanced_status()
    else:
        status = trading_engine_instance.get_status()
    
    # Add AI chat specific status
    ai_chat_status = {}
    if enhanced_chat_manager_instance:
        try:
            ai_chat_status = {
                "chat_system": "Enhanced AI Chat Manager v4.0",
                "features_active": {
                    "intent_classification": True,
                    "conversation_memory": len(enhanced_chat_manager_instance.memory.short_term),
                    "proactive_insights": True,
                    "command_suggestions": True,
                    "user_preferences": True
                },
                "memory_status": {
                    "short_term_messages": len(enhanced_chat_manager_instance.memory.short_term),
                    "topic_threads": len(enhanced_chat_manager_instance.memory.topic_threads),
                    "session_duration": str(datetime.now() - enhanced_chat_manager_instance.memory.session_start)
                },
                "performance": {
                    "average_response_time": sum(enhanced_chat_manager_instance.response_times) / len(enhanced_chat_manager_instance.response_times) if enhanced_chat_manager_instance.response_times else 0,
                    "total_interactions": len(enhanced_chat_manager_instance.response_times)
                }
            }
        except Exception as e:
            ai_chat_status = {"error": f"Failed to get chat status: {e}"}
    
    # Enhanced status with AI features
    status.update({
        "ai_chat": ai_chat_status,
        "enhanced_features": {
            "database_connected": database_manager_instance is not None,
            "risk_management_active": risk_manager_instance is not None,
            "ml_engine_active": ml_engine_instance is not None,
            "enhanced_chat_active": enhanced_chat_manager_instance is not None,
            "data_fetcher_active": data_fetcher_instance is not None
        },
        "system_info": {
            "version": "4.0",
            "ai_chat_enabled": ENHANCED_CHAT_AVAILABLE,
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

# --- Strategy Management ---

@app.get("/api/strategies/available")
async def get_available_strategies_explicit():
    """Lists all available strategy types with detailed info."""
    if trading_engine_instance:
        return trading_engine_instance.list_available_strategies()
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

# --- Control Endpoints ---

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

# --- Legacy Chat Endpoint (Backward Compatibility) ---

@app.post("/api/chat")
async def chat_with_bot(message: Union[Dict[str, Any], EnhancedChatMessage]):
    """Enhanced chat endpoint with backward compatibility."""
    
    # Handle both old and new message formats
    if isinstance(message, dict):
        user_message = message.get("message", "")
        session_id = message.get("session_id", "default")
        user_id = message.get("user_id", "default")
    else:
        user_message = message.message
        session_id = message.session_id or "default"
        user_id = message.user_id or "default"
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message content is empty")
    
    if not enhanced_chat_manager_instance:
        # Fallback for basic functionality
        return {
            "success": False,
            "response": "Enhanced AI chat not available. Please check system configuration.",
            "ai_enabled": False,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Process with enhanced chat manager
        response_data = await enhanced_chat_manager_instance.process_message(
            user_message, 
            user_id=user_id
        )
        
        return {
            "success": True,
            "response": response_data.get("response"),
            "message_type": response_data.get("message_type", "text"),
            "intent": response_data.get("intent"),
            "response_time": response_data.get("response_time"),
            "proactive_insights": response_data.get("proactive_insights", []),
            "suggestions": response_data.get("suggestions", []),
            "ai_enabled": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}", exc_info=True)
        return {
            "success": False,
            "response": f"I apologize, but I encountered an error processing your message. Please try again or use '/help' for available commands.",
            "error": str(e),
            "suggestions": ["/help", "/status", "/portfolio"],
            "timestamp": datetime.utcnow().isoformat()
        }

# --- Enhanced WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket with advanced AI chat features."""
    await websocket.accept()
    logger.info("Enhanced WebSocket client connected")
    
    try:
        # Send welcome message with enhanced capabilities
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to Enhanced AI Trading Bot v4.0",
            "features": {
                "enhanced_chat": ENHANCED_CHAT_AVAILABLE,
                "ai_features": {
                    "intent_classification": ENHANCED_CHAT_AVAILABLE,
                    "proactive_insights": ENHANCED_CHAT_AVAILABLE,
                    "conversation_memory": ENHANCED_CHAT_AVAILABLE,
                    "command_suggestions": ENHANCED_CHAT_AVAILABLE,
                    "voice_interface": True
                },
                "database": database_manager_instance is not None,
                "ml_engine": ml_engine_instance is not None,
                "risk_management": risk_manager_instance is not None
            }
        })
        
        while True:
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                message_type = data.get("type", "echo")
                
                if message_type == "chat":
                    # Enhanced chat message handling
                    user_message = data.get("message", "")
                    session_id = data.get("session_id", "default")
                    
                    if user_message and enhanced_chat_manager_instance:
                        try:
                            logger.info(f"Processing enhanced WebSocket chat: {user_message}")
                            response_data = await enhanced_chat_manager_instance.process_message(
                                user_message, user_id=session_id
                            )
                            
                            # Send enhanced response
                            await websocket.send_json({
                                "type": "chat_response",
                                "message": response_data.get("response"),
                                "message_type": response_data.get("message_type", "text"),
                                "intent": response_data.get("intent"),
                                "suggestions": response_data.get("suggestions", []),
                                "proactive_insights": response_data.get("proactive_insights", []),
                                "response_time": response_data.get("response_time"),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            
                        except Exception as e:
                            logger.error(f"Enhanced chat processing error: {e}")
                            await websocket.send_json({
                                "type": "chat_response", 
                                "message": f"I encountered an error: {str(e)}. Please try again.",
                                "message_type": "error",
                                "suggestions": ["/help", "/status"],
                                "error": True
                            })
                    else:
                        await websocket.send_json({
                            "type": "chat_response",
                            "message": "Enhanced chat manager not available or empty message",
                            "message_type": "error",
                            "error": True
                        })
                        
                elif message_type == "status":
                    # Send enhanced status
                    if trading_engine_instance:
                        if hasattr(trading_engine_instance, 'get_enhanced_status'):
                            status = trading_engine_instance.get_enhanced_status()
                        else:
                            status = trading_engine_instance.get_status()
                        
                        await websocket.send_json({
                            "type": "bot_status",
                            "status": "Running" if status.get("running") else "Stopped",
                            "data": status,
                            "metrics": {
                                "total_value": sum(status.get("balances", {}).values()),
                                "active_strategies": len(status.get("active_strategies_count", 0))
                            }
                        })
                    else:
                        await websocket.send_json({
                            "type": "bot_status", 
                            "error": "Trading engine not initialized"
                        })
                        
                elif message_type == "market_data":
                    # Send market data
                    if trading_engine_instance and trading_engine_instance.current_market_data:
                        await websocket.send_json({
                            "type": "market_update", 
                            "data": trading_engine_instance.current_market_data,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    else:
                        await websocket.send_json({
                            "type": "market_update",
                            "error": "No market data available"
                        })
                        
                elif message_type == "ping":
                    # Health check ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                        "features": {
                            "enhanced_chat": ENHANCED_CHAT_AVAILABLE,
                            "ai_enabled": enhanced_chat_manager_instance is not None
                        }
                    })
                else:
                    # Echo back for testing
                    await websocket.send_json({"type": "echo", "message": message})
                    
            except json.JSONDecodeError:
                # Handle plain text messages as enhanced chat
                if message.strip() and enhanced_chat_manager_instance:
                    try:
                        logger.info(f"Processing plain text enhanced chat: {message}")
                        response_data = await enhanced_chat_manager_instance.process_message(
                            message.strip(), user_id="websocket_user"
                        )
                        
                        await websocket.send_json({
                            "type": "chat_response",
                            "message": response_data.get("response"),
                            "message_type": response_data.get("message_type", "text"),
                            "suggestions": response_data.get("suggestions", []),
                            "proactive_insights": response_data.get("proactive_insights", [])
                        })
                        
                    except Exception as e:
                        logger.error(f"Plain text chat processing error: {e}")
                        await websocket.send_text(f"Enhanced Chat Error: {str(e)}")
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
                "message": f"WebSocket error: {str(e)}",
                "enhanced_features": False
            })
        except:
            pass  # Connection might be closed

# --- System Information Endpoint ---

@app.get("/api/system/info")
async def get_system_info():
    """Get comprehensive system information including AI chat features."""
    return {
        "version": "4.0",
        "name": f"{settings.APP_NAME} Enhanced AI",
        "description": "Industrial-grade crypto trading bot with advanced AI chat interface",
        "ai_features": {
            "enhanced_chat": ENHANCED_CHAT_AVAILABLE,
            "intent_classification": ENHANCED_CHAT_AVAILABLE,
            "conversation_memory": ENHANCED_CHAT_AVAILABLE,
            "proactive_insights": ENHANCED_CHAT_AVAILABLE,
            "command_suggestions": ENHANCED_CHAT_AVAILABLE,
            "voice_interface": True,
            "ml_integration": ML_ENGINE_AVAILABLE
        },
        "components": {
            "trading_engine": "IndustrialTradingEngine v4.0",
            "ai_chat": "Enhanced AI Chat Manager v4.0" if ENHANCED_CHAT_AVAILABLE else "Basic Chat",
            "ml_engine": "OctoBotMLEngine" if ml_engine_instance else "Not Available",
            "database": "DatabaseManager" if database_manager_instance else "Not Available",
            "risk_management": "RiskManager" if risk_manager_instance else "Not Available",
            "data_fetcher": "CryptoDataFetcher" if data_fetcher_instance else "Not Available",
            "notification_system": "SimpleNotificationManager Enhanced"
        },
        "api_endpoints": {
            "total": len([route for route in app.routes]),
            "enhanced_endpoints": [
                "/api/status/enhanced",
                "/api/chat",
                "/api/system/info",
                "/chat"
            ],
            "ai_chat_endpoints": [
                "/api/chat",
                "/api/chat/analyze",
                "/api/chat/preferences",
                "/api/chat/history"
            ] if ENHANCED_CHAT_AVAILABLE else []
        },
        "performance": {
            "chat_interactions": len(enhanced_chat_manager_instance.response_times) if enhanced_chat_manager_instance else 0,
            "average_response_time": sum(enhanced_chat_manager_instance.response_times) / len(enhanced_chat_manager_instance.response_times) if enhanced_chat_manager_instance and enhanced_chat_manager_instance.response_times else 0
        } if enhanced_chat_manager_instance else {},
        "timestamp": datetime.utcnow().isoformat()
    }

# Main entry point
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)