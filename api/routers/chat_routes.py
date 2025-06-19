# api/routers/chat_routes.py
import logging

from fastapi import APIRouter, Depends, HTTPException, Form # <--- Add Form here!

# ... rest of the code ...

from core.config import settings # For GOOGLE_AI_ENABLED status
from main import get_chat_bot_dep, get_trading_engine_dep, get_gemini_ai_dep # Dependency providers
from ml.models import ChatMessage # Reused Pydantic model

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat Interface"]
)

logger = logging.getLogger(__name__)

@router.post("/")
async def api_chat_message(
    message: ChatMessage, # Direct Pydantic model for JSON request body
    chat_bot=Depends(get_chat_bot_dep),
    engine=Depends(get_trading_engine_dep) # Required for AI context, though ChatBot already has it
):
    """Processes a chat message and returns a response from the trading bot or AI."""
    try:
        response_text = await chat_bot.process_message(message.message)
        return {"success": True, "response": response_text, "ai_enabled": settings.GOOGLE_AI_ENABLED}
    except Exception as e:
        logger.error(f"Error processing chat message: {e}", exc_info=True)
        # Fallback to basic processing if even the fallback in ChatBot fails
        fallback_response = chat_bot._handle_fallback_response(message.message)
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}. Fallback: {fallback_response}")

@router.post("/ai/analyze")
async def api_ai_analyze(
    gemini_ai=Depends(get_gemini_ai_dep),
    engine=Depends(get_trading_engine_dep)
):
    """Triggers an AI-powered market analysis."""
    if not settings.GOOGLE_AI_ENABLED or not gemini_ai:
        raise HTTPException(status_code=403, detail="AI not configured. Add GOOGLE_AI_API_KEY to .env file.")

    market_data = engine.current_market_data
    if not market_data:
        raise HTTPException(status_code=404, detail="No market data available for AI analysis yet.")

    try:
        analysis = await gemini_ai.analyze_market_sentiment(market_data)
        return {
            "success": True,
            "analysis": analysis,
            "data_points_analyzed": len(market_data),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"AI market analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI market analysis failed: {str(e)}")

@router.post("/ai/ask")
async def api_ai_ask(
    question: str = Form(...), # Use Form for multipart/form-data or application/x-www-form-urlencoded
    gemini_ai=Depends(get_gemini_ai_dep),
    engine=Depends(get_trading_engine_dep)
):
    """Asks the AI a trading-related question with access to bot context."""
    if not settings.GOOGLE_AI_ENABLED or not gemini_ai:
        raise HTTPException(status_code=403, detail="AI not configured. Add GOOGLE_AI_API_KEY to .env file.")

    try:
        # Get trading context including current portfolio and status
        context = chat_bot._get_trading_context() # Reusing method from ChatBot instance
        response = await gemini_ai.generate_response(question, context)

        return {
            "success": True,
            "question": question,
            "answer": response,
            "context_provided": True
        }
    except Exception as e:
        logger.error(f"AI question failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI could not process your question: {str(e)}")

@router.get("/ai/status")
async def api_ai_status():
    """Provides the current status of the AI integration."""
    return {
        "success": True,
        "ai_enabled": settings.GOOGLE_AI_ENABLED,
        "model": "gemini-2.0-flash" if settings.GOOGLE_AI_ENABLED else None,
        "features": [
            "Smart Chat Responses",
            "Market Sentiment Analysis",
            "Trading Recommendations",
            "Risk Assessment"
        
        ] if settings.GOOGLE_AI_ENABLED else ["Basic Chat Only"]
    }