# api/routers/common_routes.py
import logging
from datetime import datetime
from typing import Optional # Added Optional for type hinting

from fastapi import APIRouter, HTTPException, Depends

# Import dependency providers from core.dependencies
from core.dependencies import (
    get_trading_engine_dep,
    get_chat_bot_dep,
    get_connection_manager_dep,
    get_gemini_ai_dep,
    get_vector_db_client_dep
)

router = APIRouter(
    prefix="/api",
    tags=["Common Operations"]
)

logger = logging.getLogger(__name__)

@router.post("/start")
async def api_start(engine=Depends(get_trading_engine_dep)):
    """Starts the trading bot operations."""
    try:
        engine.start()
        return {"success": True, "message": "Trading started successfully!"}
    except Exception as e:
        logger.error(f"Failed to start trading: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start trading: {str(e)}")

@router.post("/stop")
async def api_stop(engine=Depends(get_trading_engine_dep)):
    """Stops the trading bot operations."""
    try:
        engine.stop()
        return {"success": True, "message": "Trading stopped successfully!"}
    except Exception as e:
        logger.error(f"Failed to stop trading: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop trading: {str(e)}")

@router.get("/status")
async def api_status(engine=Depends(get_trading_engine_dep)):
    """Gets the current status and performance metrics of the trading bot."""
    # Ensure all components used in metrics are initialized before access.
    # The dependency injection system takes care of it, but internal methods might crash.
    try:
        metrics = engine.get_performance_metrics()
        ml_status = engine.ml_engine.get_model_status() # Get ML model status from engine

        return {
            "success": True,
            "status": "RUNNING" if engine.running else "STOPPED",
            "metrics": metrics,
            "ml_models_count": len(ml_status),
            "active_strategies_count": len(engine.list_active_strategies()),
            "market_data_symbols_count": len(engine.current_market_data),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        # Raise HTTPException so FastAPI catches it and formats it correctly
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/positions")
async def api_positions(engine=Depends(get_trading_engine_dep)):
    """Retrieves all currently open trading positions."""
    try:
        return {"success": True, "positions": engine.positions}
    except Exception as e:
        logger.error(f"Failed to get positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get positions: {str(e)}")


@router.get("/market-data")
async def api_market_data(engine=Depends(get_trading_engine_dep)):
    """Retrieves the latest collected market data."""
    try:
        if not engine.current_market_data:
            raise HTTPException(status_code=404, detail="No market data available yet. Data feed might be initializing.")
        return {"success": True, "market_data": engine.current_market_data}
    except Exception as e:
        logger.error(f"Failed to get market data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get market data: {str(e)}")


@router.get("/health")
async def api_health(
    engine_dep=Depends(get_trading_engine_dep),
    chat_bot_dep=Depends(get_chat_bot_dep),
    ws_manager_dep=Depends(get_connection_manager_dep),
    gemini_ai_dep=Depends(get_gemini_ai_dep),
    vector_db_dep=Depends(get_vector_db_client_dep)
):
    """Performs a comprehensive health check of various bot components."""
    try:
        health_status = {
            "server": "healthy", # If this endpoint is reached, FastAPI is running
            "trading_engine": "healthy" if engine_dep else "error (not initialized)",
            "chat_bot": "healthy" if chat_bot_dep else "error (not initialized)",
            "ml_engine": "healthy" if engine_dep and hasattr(engine_dep, 'ml_engine') else "error (no ml_engine)",
            "data_fetcher": "healthy" if engine_dep and engine_dep.data_fetcher.running_feed else "inactive (data feed)",
            "websocket_connections": len(ws_manager_dep.active_connections),
            "gemini_ai": "enabled" if gemini_ai_dep else "disabled",
            "vector_db": "ready" if vector_db_dep and vector_db_dep.is_ready else "not ready/disabled",
            "timestamp": datetime.now().isoformat(),
            "uptime": "running" # Simple placeholder
        }

        # More granular checks for trading engine specifically
        if engine_dep:
            health_status["market_data_feed"] = "active" if bool(engine_dep.current_market_data) else "inactive/initializing"
            # Health check for ML model loading (not actual model correctness)
            if engine_dep.ml_engine.models:
                health_status["ml_models_loaded"] = True
                health_status["ml_loaded_count"] = len(engine_dep.ml_engine.models)
            else:
                health_status["ml_models_loaded"] = False
        else:
            health_status["market_data_feed"] = "n/a (engine not initialized)"


        return {
            "success": True,
            "status": "healthy",
            "components": health_status,
            "version": "3.0.0"
        }
    except Exception as e:
        logger.error(f"Health check internal error: {e}", exc_info=True)
        # Raise HTTPException here so FastAPI catches it and formats it correctly
        raise HTTPException(status_code=500, detail=f"Health check failed due to internal error: {str(e)}")