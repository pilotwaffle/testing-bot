# File: E:\Trade Chat Bot\G Trading Bot\fastapi_integration.py
# Location: E:\Trade Chat Bot\G Trading Bot\fastapi_integration.py

"""
FastAPI Integration for Exchange Error Handling
Elite Trading Bot V3.0 - API Error Handling Integration
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging
from exchange_error_handler import (
    ExchangeErrorHandler, 
    ExchangeHealthMonitor, 
    EnhancedKrakenAPI,
    ErrorSeverity,
    ErrorType
)

# Pydantic models for API responses
class ErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    message: str
    timestamp: datetime
    retry_suggested: bool = False
    fallback_used: bool = False

class HealthStatus(BaseModel):
    exchange: str
    status: str
    recent_errors: int
    critical_errors: int
    circuit_breakers: Dict[str, str]
    last_error: Optional[str]

class SystemHealth(BaseModel):
    overall_status: str
    exchanges: Dict[str, HealthStatus]
    total_errors_1h: int
    timestamp: datetime

class TradingResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[ErrorResponse] = None
    source: str = "primary"  # primary, fallback, cached

# WebSocket connection manager with error handling
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.error_handler = ExchangeErrorHandler()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        await self.send_connection_status(websocket, "connected")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logging.error(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients with error handling"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logging.error(f"WebSocket broadcast failed: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_error_notification(self, error_type: str, message: str, severity: str):
        """Send error notification to all connected clients"""
        notification = {
            "type": "error_notification",
            "error_type": error_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(notification)
    
    async def send_connection_status(self, websocket: WebSocket, status: str):
        """Send connection status to client"""
        message = {
            "type": "connection_status",
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_personal_message(message, websocket)
    
    async def send_health_update(self, health_data: Dict):
        """Send health status update to all clients"""
        message = {
            "type": "health_update",
            "data": health_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(message)

# Initialize components
manager = ConnectionManager()
error_handler = ExchangeErrorHandler()
health_monitor = ExchangeHealthMonitor(error_handler)

# Enhanced API exception handler
async def create_error_response(
    error: Exception, 
    exchange: str = "unknown", 
    endpoint: str = "unknown",
    fallback_used: bool = False
) -> ErrorResponse:
    """Create standardized error response"""
    
    error_type = error_handler.classify_error(error)
    severity = error_handler.get_error_severity(error_type)
    
    # Determine if retry is suggested
    retry_suggested = error_handler.should_retry(error_type, 0, 3)
    
    return ErrorResponse(
        error_type=error_type.value,
        message=str(error),
        timestamp=datetime.now(),
        retry_suggested=retry_suggested,
        fallback_used=fallback_used
    )

# FastAPI exception handlers
def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers for the FastAPI app"""
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception):
        """Global exception handler with error classification"""
        
        # Extract exchange and endpoint from request path
        path_parts = str(request.url.path).split('/')
        exchange = "unknown"
        endpoint = "unknown"
        
        if len(path_parts) > 2:
            if 'kraken' in path_parts:
                exchange = "kraken"
            elif 'binance' in path_parts:
                exchange = "binance"
            elif 'coinbase' in path_parts:
                exchange = "coinbase"
            
            endpoint = path_parts[-1] if path_parts else "unknown"
        
        error_response = await create_error_response(exc, exchange, endpoint)
        
        # Send WebSocket notification for critical errors
        if error_handler.get_error_severity(error_handler.classify_error(exc)) in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await manager.send_error_notification(
                error_response.error_type,
                error_response.message,
                error_handler.get_error_severity(error_handler.classify_error(exc)).value
            )
        
        # Return appropriate HTTP status
        status_code = 500
        if hasattr(exc, 'status_code'):
            status_code = exc.status_code
        elif 'rate limit' in str(exc).lower():
            status_code = 429
        elif 'auth' in str(exc).lower() or 'unauthorized' in str(exc).lower():
            status_code = 401
        elif 'not found' in str(exc).lower():
            status_code = 404
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict()
        )

# Enhanced API endpoints with error handling
def create_enhanced_api_routes(app: FastAPI, kraken_api: EnhancedKrakenAPI):
    """Create API routes with comprehensive error handling"""
    
    @app.get("/api/market-data/{symbol}", response_model=TradingResponse)
    async def get_market_data(symbol: str, background_tasks: BackgroundTasks):
        """Get market data with fallback handling"""
        try:
            data = await kraken_api.get_ticker(symbol)
            return TradingResponse(success=True, data=data, source="primary")
        
        except Exception as e:
            # Try fallback data source
            fallback_data = await error_handler.get_fallback_data("market_data", symbol)
            if fallback_data:
                return TradingResponse(
                    success=True, 
                    data=fallback_data, 
                    source="fallback"
                )
            
            error_response = await create_error_response(e, "kraken", "market_data")
            return TradingResponse(success=False, error=error_response)
    
    @app.post("/api/trading/order", response_model=TradingResponse)
    async def place_trading_order(
        symbol: str, 
        side: str, 
        amount: float, 
        price: float,
        background_tasks: BackgroundTasks
    ):
        """Place trading order with comprehensive error handling"""
        try:
            # Validate inputs
            if amount <= 0:
                raise ValueError("Amount must be positive")
            if price <= 0:
                raise ValueError("Price must be positive")
            if side not in ["buy", "sell"]:
                raise ValueError("Side must be 'buy' or 'sell'")
            
            result = await kraken_api.place_order(symbol, side, amount, price)
            
            # Send success notification via WebSocket
            await manager.broadcast({
                "type": "order_placed",
                "data": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return TradingResponse(success=True, data=result)
        
        except Exception as e:
            error_response = await create_error_response(e, "kraken", "trading")
            
            # Send error notification for trading failures
            await manager.send_error_notification(
                error_response.error_type,
                f"Trading order failed: {error_response.message}",
                "high"
            )
            
            return TradingResponse(success=False, error=error_response)
    
    @app.get("/api/health", response_model=SystemHealth)
    async def get_system_health():
        """Get comprehensive system health status"""
        try:
            health_data = health_monitor.get_system_health()
            return SystemHealth(
                overall_status=health_data["overall_status"],
                exchanges={k: HealthStatus(**v) for k, v in health_data["exchanges"].items()},
                total_errors_1h=health_data["total_errors_1h"],
                timestamp=datetime.now()
            )
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")
    
    @app.get("/api/health/{exchange}", response_model=HealthStatus)
    async def get_exchange_health(exchange: str):
        """Get health status for specific exchange"""
        try:
            health_data = health_monitor.get_exchange_health(exchange)
            return HealthStatus(**health_data)
        except Exception as e:
            logging.error(f"Exchange health check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Health check failed for {exchange}")
    
    @app.get("/api/errors/recent")
    async def get_recent_errors(hours: int = 1):
        """Get recent errors for debugging"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [
            {
                "type": err.type.value,
                "severity": err.severity.value,
                "message": err.message,
                "exchange": err.exchange,
                "endpoint": err.endpoint,
                "timestamp": err.timestamp.isoformat(),
                "retry_count": err.retry_count
            }
            for err in error_handler.error_history
            if err.timestamp > cutoff_time
        ]
        return {"errors": recent_errors, "count": len(recent_errors)}
    
    @app.post("/api/system/reset-circuit-breaker/{endpoint}")
    async def reset_circuit_breaker(endpoint: str):
        """Manually reset circuit breaker for an endpoint"""
        try:
            if endpoint in error_handler.circuit_breakers:
                error_handler.circuit_breakers[endpoint].failure_count = 0
                error_handler.circuit_breakers[endpoint].state = error_handler.circuit_breakers[endpoint].state.CLOSED
                return {"message": f"Circuit breaker reset for {endpoint}"}
            else:
                raise HTTPException(status_code=404, detail="Circuit breaker not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint with error handling
def create_websocket_endpoint(app: FastAPI):
    """Create WebSocket endpoint with comprehensive error handling"""
    
    @app.websocket("/ws/notifications")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                # Keep connection alive and handle incoming messages
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "ping":
                        await manager.send_personal_message(
                            {"type": "pong", "timestamp": datetime.now().isoformat()}, 
                            websocket
                        )
                    elif message.get("type") == "subscribe":
                        # Handle subscription requests
                        await manager.send_personal_message(
                            {"type": "subscribed", "channel": message.get("channel")}, 
                            websocket
                        )
                
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await manager.send_personal_message(
                        {"type": "heartbeat", "timestamp": datetime.now().isoformat()}, 
                        websocket
                    )
                except json.JSONDecodeError:
                    await manager.send_personal_message(
                        {"type": "error", "message": "Invalid JSON"}, 
                        websocket
                    )
        
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
            manager.disconnect(websocket)

# Background tasks for health monitoring
async def health_monitoring_task():
    """Background task to monitor system health and send updates"""
    while True:
        try:
            health_data = health_monitor.get_system_health()
            await manager.send_health_update(health_data)
            
            # Check for critical issues
            if health_data["overall_status"] == "unhealthy":
                await manager.send_error_notification(
                    "system_health",
                    "System health is degraded - multiple exchanges experiencing issues",
                    "critical"
                )
            
            await asyncio.sleep(60)  # Check every minute
        
        except Exception as e:
            logging.error(f"Health monitoring task failed: {e}")
            await asyncio.sleep(60)

# Complete FastAPI app setup
def create_enhanced_trading_app() -> FastAPI:
    """Create FastAPI app with comprehensive error handling"""
    
    app = FastAPI(
        title="Elite Trading Bot V3.0 - Enhanced API",
        description="Trading bot with comprehensive error handling",
        version="3.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup error handlers
    setup_exception_handlers(app)
    
    # Initialize enhanced Kraken API (use your actual credentials)
    import os
    kraken_api = EnhancedKrakenAPI(
        os.getenv("KRAKEN_API_KEY", ""), 
        os.getenv("KRAKEN_SECRET", "")
    )
    
    # Create routes
    create_enhanced_api_routes(app, kraken_api)
    create_websocket_endpoint(app)
    
    # Add startup event to begin health monitoring
    @app.on_event("startup")
    async def startup_event():
        # Start background health monitoring
        asyncio.create_task(health_monitoring_task())
        logging.info("Enhanced Trading Bot V3.0 started with comprehensive error handling")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logging.info("Enhanced Trading Bot V3.0 shutting down")
    
    return app

# Example usage
if __name__ == "__main__":
    import uvicorn
    
    app = create_enhanced_trading_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)