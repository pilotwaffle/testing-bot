# core/dependencies.py
import logging
from typing import Optional, List, Dict, Any

# These are placeholders for the actual instances, which will be set by main.py
# during its lifespan events. They are intentionally initialized to None here.
_trading_engine = None
_chat_bot = None
_gemini_ai = None
_notification_manager = None
_connection_manager = None
_vector_db_client = None

logger = logging.getLogger(__name__)


# --- Dependency Injection Provider Functions ---
# These functions will be passed to FastAPI's Depends()
# They return the globally managed instances.

def get_trading_engine_dep():
    if _trading_engine is None:
        logger.error("TradingEngine not initialized. App startup likely failed or called too early.")
        raise RuntimeError("TradingEngine not initialized. App startup likely failed or called too early.")
    return _trading_engine

def get_chat_bot_dep():
    if _chat_bot is None:
        logger.error("ChatBot not initialized. App startup likely failed.")
        raise RuntimeError("ChatBot not initialized. App startup likely failed.")
    return _chat_bot

def get_gemini_ai_dep() -> Optional[Any]: # Use Any here since it can be None
    # No error if None, as AI is optional
    return _gemini_ai

def get_notification_manager_dep():
    if _notification_manager is None:
        logger.error("NotificationManager not initialized. App startup likely failed.")
        raise RuntimeError("NotificationManager not initialized. App startup likely failed.")
    return _notification_manager

def get_connection_manager_dep():
    if _connection_manager is None:
        logger.error("ConnectionManager not initialized. App startup likely failed.")
        raise RuntimeError("ConnectionManager not initialized. App startup likely failed.")
    return _connection_manager

def get_vector_db_client_dep() -> Optional[Any]: # Use Any here since it can be None
    return _vector_db_client

# This helper allows main.py to set the initialized instances into these global variables
def set_dependencies(
    trading_engine_instance, chat_bot_instance, gemini_ai_instance,
    notification_manager_instance, connection_manager_instance, vector_db_client_instance
):
    global _trading_engine, _chat_bot, _gemini_ai, _notification_manager, _connection_manager, _vector_db_client
    _trading_engine = trading_engine_instance
    _chat_bot = chat_bot_instance
    _gemini_ai = gemini_ai_instance
    _notification_manager = notification_manager_instance
    _connection_manager = connection_manager_instance
    _vector_db_client = vector_db_client_instance