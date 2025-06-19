# api/routers/chat_routes.py - Enhanced Version
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel

from core.config import settings
from main import get_trading_engine_dep, get_ml_engine_dep, get_data_fetcher_dep, get_notification_manager_dep
from ml.models import ChatMessage
from ai.chat_manager import EnhancedChatManager, MessageType, Intent

router = APIRouter(
    prefix="/api/chat",
    tags=["Enhanced Chat Interface"]
)

logger = logging.getLogger(__name__)

# Enhanced request models
class EnhancedChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    user_id: Optional[str] = "default"
    context: Optional[Dict[str, Any]] = None

class ChatAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    analysis_type: str = "comprehensive"

class ChatPreferencesUpdate(BaseModel):
    communication_style: Optional[str] = None
    risk_tolerance: Optional[str] = None
    favorite_symbols: Optional[list] = None
    notification_preferences: Optional[Dict[str, bool]] = None

# Global chat manager instance (will be initialized in main.py)
_chat_manager: Optional[EnhancedChatManager] = None

def get_chat_manager(
    trading_engine=Depends(get_trading_engine_dep),
    ml_engine=Depends(get_ml_engine_dep),
    data_fetcher=Depends(get_data_fetcher_dep),
    notification_manager=Depends(get_notification_manager_dep)
) -> EnhancedChatManager:
    """Dependency to get or create the enhanced chat manager"""
    global _chat_manager
    
    if _chat_manager is None:
        _chat_manager = EnhancedChatManager(
            trading_engine=trading_engine,
            ml_engine=ml_engine,
            data_fetcher=data_fetcher,
            notification_manager=notification_manager
        )
        logger.info("Enhanced ChatManager initialized")
    
    return _chat_manager

@router.post("/")
async def api_enhanced_chat_message(
    message: EnhancedChatMessage,
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Enhanced chat endpoint with advanced AI features"""
    try:
        # Process message with enhanced chat manager
        response_data = await chat_manager.process_message(
            message.message, 
            user_id=message.user_id or "default"
        )
        
        # Return enhanced response format
        return {
            "success": True,
            "response": response_data.get("response"),
            "message_type": response_data.get("message_type", "text"),
            "intent": response_data.get("intent"),
            "response_time": response_data.get("response_time"),
            "proactive_insights": response_data.get("proactive_insights", []),
            "suggestions": response_data.get("suggestions", []),
            "ai_enabled": settings.GOOGLE_AI_ENABLED,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing enhanced chat message: {e}", exc_info=True)
        
        # Enhanced error response with fallback
        return {
            "success": False,
            "response": f"I apologize, but I encountered an error processing your message. Please try again or use a simpler command like '/help'.",
            "message_type": "error",
            "error": str(e),
            "suggestions": ["/help", "/status", "/portfolio"],
            "timestamp": datetime.now().isoformat()
        }

@router.post("/analyze")
async def api_chat_analysis(
    request: ChatAnalysisRequest,
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Enhanced analysis endpoint triggered from chat"""
    try:
        # Create analysis message
        analysis_message = f"/analyze {request.symbol} {request.timeframe}"
        
        response_data = await chat_manager.process_message(analysis_message)
        
        return {
            "success": True,
            "analysis": response_data.get("response"),
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "analysis_type": request.analysis_type,
            "message_type": "analysis",
            "metadata": {
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "analysis_id": f"analysis_{datetime.now().timestamp()}"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/ai/market-sentiment")
async def api_ai_market_sentiment(
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Get AI-powered market sentiment analysis"""
    try:
        # Use enhanced chat manager for market sentiment
        response_data = await chat_manager.process_message("What's the current market sentiment?")
        
        return {
            "success": True,
            "sentiment_analysis": response_data.get("response"),
            "message_type": "analysis",
            "insights": response_data.get("proactive_insights", []),
            "confidence": "high",  # This could be calculated based on data quality
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI market sentiment analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Market sentiment analysis failed: {str(e)}")

@router.post("/ai/ask")
async def api_ai_ask(
    question: str = Form(...),
    context: Optional[str] = Form(None),
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Ask the AI a trading-related question with enhanced context"""
    try:
        # Process question with enhanced chat manager
        response_data = await chat_manager.process_message(question)
        
        return {
            "success": True,
            "question": question,
            "answer": response_data.get("response"),
            "intent": response_data.get("intent"),
            "message_type": response_data.get("message_type", "text"),
            "suggestions": response_data.get("suggestions", []),
            "context_provided": True,
            "response_time": response_data.get("response_time"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI question processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI could not process your question: {str(e)}")

@router.post("/preferences")
async def update_chat_preferences(
    preferences: ChatPreferencesUpdate,
    user_id: str = "default",
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Update user chat preferences"""
    try:
        # Update user preferences in chat manager
        if preferences.communication_style:
            chat_manager.user_preferences.communication_style = preferences.communication_style
        
        if preferences.risk_tolerance:
            chat_manager.user_preferences.risk_tolerance = preferences.risk_tolerance
        
        if preferences.favorite_symbols:
            chat_manager.user_preferences.favorite_symbols = preferences.favorite_symbols
        
        if preferences.notification_preferences:
            chat_manager.user_preferences.notification_preferences.update(preferences.notification_preferences)
        
        return {
            "success": True,
            "message": "Chat preferences updated successfully",
            "preferences": {
                "communication_style": chat_manager.user_preferences.communication_style,
                "risk_tolerance": chat_manager.user_preferences.risk_tolerance,
                "favorite_symbols": chat_manager.user_preferences.favorite_symbols[:5],  # Limit for response size
                "notification_preferences": chat_manager.user_preferences.notification_preferences
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to update chat preferences: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")

@router.get("/preferences")
async def get_chat_preferences(
    user_id: str = "default",
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Get current user chat preferences"""
    try:
        return {
            "success": True,
            "preferences": {
                "communication_style": chat_manager.user_preferences.communication_style,
                "risk_tolerance": chat_manager.user_preferences.risk_tolerance,
                "favorite_symbols": chat_manager.user_preferences.favorite_symbols,
                "notification_preferences": chat_manager.user_preferences.notification_preferences,
                "response_format": chat_manager.user_preferences.response_format
            }
        }
    except Exception as e:
        logger.error(f"Failed to get chat preferences: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")

@router.get("/history")
async def get_chat_history(
    limit: int = 20,
    user_id: str = "default",
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Get conversation history"""
    try:
        # Get recent messages from memory
        recent_messages = list(chat_manager.memory.short_term)[-limit:]
        
        history = []
        for msg in recent_messages:
            history.append({
                "sender": msg.sender,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "message_type": msg.message_type.value,
                "intent": msg.intent.value if msg.intent else None
            })
        
        return {
            "success": True,
            "history": history,
            "session_summary": chat_manager.memory.create_summary(),
            "total_messages": len(chat_manager.memory.short_term),
            "topics_discussed": list(chat_manager.memory.topic_threads.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@router.post("/execute-signal")
async def execute_trade_signal(
    signal_id: str,
    confirm: bool = False,
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Execute a trade signal generated by the chat AI"""
    try:
        if not confirm:
            return {
                "success": False,
                "message": "Trade execution requires confirmation",
                "requires_confirmation": True,
                "signal_id": signal_id
            }
        
        # Process execute command through chat manager
        execute_message = f"/execute {signal_id}"
        response_data = await chat_manager.process_message(execute_message)
        
        return {
            "success": True,
            "execution_result": response_data.get("response"),
            "signal_id": signal_id,
            "message_type": "command_result",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to execute trade signal {signal_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute signal: {str(e)}")

@router.get("/commands")
async def get_available_commands():
    """Get list of available chat commands"""
    commands = {
        "/status": {
            "description": "Get comprehensive bot status and metrics",
            "usage": "/status",
            "category": "status"
        },
        "/portfolio": {
            "description": "View detailed portfolio analysis",
            "usage": "/portfolio",
            "category": "portfolio"
        },
        "/analyze": {
            "description": "AI market analysis for symbol and timeframe",
            "usage": "/analyze BTC/USDT 1h",
            "category": "analysis"
        },
        "/positions": {
            "description": "Current open positions",
            "usage": "/positions",
            "category": "portfolio"
        },
        "/strategies": {
            "description": "Manage trading strategies",
            "usage": "/strategies",
            "category": "strategy"
        },
        "/risk": {
            "description": "Risk assessment and recommendations",
            "usage": "/risk",
            "category": "risk"
        },
        "/market": {
            "description": "Current market overview",
            "usage": "/market",
            "category": "market"
        },
        "/settings": {
            "description": "Configure chat preferences",
            "usage": "/settings",
            "category": "settings"
        },
        "/history": {
            "description": "View conversation history",
            "usage": "/history",
            "category": "history"
        },
        "/help": {
            "description": "Show all available commands",
            "usage": "/help",
            "category": "help"
        }
    }
    
    return {
        "success": True,
        "commands": commands,
        "total_commands": len(commands),
        "categories": list(set(cmd["category"] for cmd in commands.values()))
    }

@router.get("/status")
async def api_chat_status(
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Get current chat system status"""
    try:
        return {
            "success": True,
            "chat_system": "Enhanced AI Chat Manager",
            "ai_enabled": settings.GOOGLE_AI_ENABLED,
            "features": {
                "intent_classification": True,
                "conversation_memory": True,
                "proactive_insights": True,
                "command_suggestions": True,
                "user_preferences": True,
                "voice_interface": True,
                "enhanced_message_types": True,
                "real_time_analysis": True
            },
            "memory_status": {
                "short_term_messages": len(chat_manager.memory.short_term),
                "topic_threads": len(chat_manager.memory.topic_threads),
                "session_duration": str(datetime.now() - chat_manager.memory.session_start)
            },
            "performance": {
                "average_response_time": sum(chat_manager.response_times) / len(chat_manager.response_times) if chat_manager.response_times else 0,
                "total_interactions": len(chat_manager.response_times)
            },
            "model_info": {
                "primary_ai": "Gemini 2.0 Flash" if settings.GOOGLE_AI_ENABLED else "Local Fallback",
                "ml_models_loaded": len(chat_manager.ml_engine.models) if chat_manager.ml_engine else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get chat status: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "chat_system": "Enhanced AI Chat Manager (Error State)"
        }

@router.post("/reset-memory")
async def reset_chat_memory(
    confirm: bool = False,
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Reset chat conversation memory"""
    try:
        if not confirm:
            return {
                "success": False,
                "message": "Memory reset requires confirmation",
                "requires_confirmation": True
            }
        
        # Reset memory
        chat_manager.memory = chat_manager.memory.__class__()
        
        return {
            "success": True,
            "message": "Chat memory has been reset",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset chat memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset memory: {str(e)}")

# Background task for proactive insights
async def generate_periodic_insights(chat_manager: EnhancedChatManager):
    """Background task to generate proactive insights"""
    try:
        insights = await chat_manager._generate_proactive_insights()
        if insights:
            # This could be sent via WebSocket to connected clients
            logger.info(f"Generated {len(insights)} proactive insights")
    except Exception as e:
        logger.error(f"Failed to generate proactive insights: {e}", exc_info=True)

@router.post("/trigger-insights")
async def trigger_proactive_insights(
    background_tasks: BackgroundTasks,
    chat_manager: EnhancedChatManager = Depends(get_chat_manager)
):
    """Manually trigger proactive insights generation"""
    try:
        background_tasks.add_task(generate_periodic_insights, chat_manager)
        
        return {
            "success": True,
            "message": "Proactive insights generation triggered",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger insights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to trigger insights: {str(e)}")