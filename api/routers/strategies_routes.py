# api/routers/strategies_routes.py
import logging
import json
from datetime import datetime # Import datetime for strategy "added_at" timestamp
from typing import Any, Dict, Optional # Import Optional for type hints

from fastapi import APIRouter, Depends, HTTPException, Request # Ensure Depends is imported

from core.dependencies import get_trading_engine_dep # Import the dependency provider

router = APIRouter(
    prefix="/api/strategies",
    tags=["Trading Strategies"]
)

logger = logging.getLogger(__name__)

@router.get("/")
async def api_get_strategies(engine: get_trading_engine_dep = Depends(get_trading_engine_dep)):
    """Retrieves available and currently active trading strategies."""
    try:
        available = engine.list_available_strategies()
        active = engine.list_active_strategies()
        return {
            "success": True,
            "available": available,
            "active": active
        }
    except Exception as e:
        logger.error(f"Failed to retrieve strategies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve strategies: {str(e)}")

@router.post("/add")
async def api_add_strategy(
    request: Request, # Keep Request to manually inspect content-type
    engine: get_trading_engine_dep = Depends(get_trading_engine_dep)
):
    """Adds a new trading strategy to the bot's active strategies list."""
    strategy_name: Optional[str] = None # Initialize as Optional[str]
    config_dict: Dict[str, Any] = {} # Initialize as empty dictionary

    content_type = request.headers.get("content-type", "").lower()

    try:
        if "application/json" in content_type:
            try:
                data = await request.json()
                strategy_name = data.get("strategy_name") or data.get("name")
                config_dict = data.get("config", {})
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format in request body. Please ensure it's valid JSON.")
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form = await request.form()
            strategy_name = form.get("strategy_name") or form.get("name")
            config_str = form.get("config", "") # Get config as string, default to empty
            try:
                config_dict = json.loads(config_str) if config_str else {} # Parse only if string is not empty
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format in 'config' form field. Please ensure it's a valid JSON string.")
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported Media Type: '{content_type}'. Expected 'application/json', 'application/x-www-form-urlencoded', or 'multipart/form-data'.")

        if not strategy_name: # Check if strategy_name was successfully extracted
            raise HTTPException(status_code=400, detail="Strategy name is required.")

        # Call the engine's method and handle its boolean return
        if not engine.add_strategy(strategy_name, config_dict):
            raise HTTPException(status_code=409, detail=f"Strategy '{strategy_name}' already exists or could not be added. (Conflict)")

        return {
            "success": True,
            "message": f"Strategy '{strategy_name}' added successfully."
        }

    except HTTPException: # Catch FastAPI's HTTPExceptions raised above and re-raise them
        raise
    except Exception as e: # Catch any other unexpected exceptions
        logger.error(f"Add strategy internal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while adding strategy: {str(e)}")


@router.post("/remove")
async def api_remove_strategy(
    request: Request,
    engine: get_trading_engine_dep = Depends(get_trading_engine_dep)
):
    """Removes a trading strategy from the bot's active strategies list."""
    strategy_name: Optional[str] = None # Initialize as Optional[str]

    content_type = request.headers.get("content-type", "").lower()

    try:
        if "application/json" in content_type:
            try:
                data = await request.json()
                strategy_name = data.get("strategy_name") or data.get("name")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format in request body.")
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form = await request.form()
            strategy_name = form.get("strategy_name") or form.get("name")
        else:
            raise HTTPException(status_code=415, detail=f"Unsupported Media Type: '{content_type}'. Expected 'application/json' or form data.")

        if not strategy_name:
            raise HTTPException(status_code=400, detail="Strategy name is required for removal.")

        # --- Logic for Idempotent Delete ---
        # Call the engine's remove method. It returns True if removed, False if not found.
        engine_removal_success = engine.remove_strategy(strategy_name)

        if not engine_removal_success:
            # If the engine couldn't remove it (meaning it wasn't found),
            # we still report success (200 OK) because the desired state (strategy not present) is achieved.
            message_suffix = " (was already removed or not found)."
        else:
            # It was found and successfully removed.
            message_suffix = " successfully."

        return {
            "success": True, # Always True here for idempotent delete
            "message": f"Strategy '{strategy_name}' removed{message_suffix}"
            # FastAPI defaults to 200 OK for successful returns.
        }
        # --- End Idempotent Delete Logic ---

    except HTTPException: # Catch FastAPI's HTTPExceptions raised above and re-raise them
        raise
    except Exception as e: # Catch any other unexpected exceptions
        logger.error(f"Remove strategy error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while removing strategy: {str(e)}")