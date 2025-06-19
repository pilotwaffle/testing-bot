# ai/chat_manager.py - Enhanced with debugging
import logging
from typing import Dict, Any, Optional
from core.config import settings

logger = logging.getLogger(__name__)

# Set higher log level for debugging
logger.setLevel(logging.DEBUG)

# --- GLOBAL FLAG FOR GOOGLE AI ---
GOOGLE_AI_SDK_AVAILABLE = False 

try:
    if settings.GOOGLE_AI_API_KEY and settings.GOOGLE_AI_ENABLED:
        import google.generativeai as genai
        GOOGLE_AI_SDK_AVAILABLE = True
        logger.info("Google Generative AI SDK available.")
        genai.configure(api_key=settings.GOOGLE_AI_API_KEY)
    else:
        logger.warning("Google AI API Key not found or disabled in settings. Skipping SDK load.")
except ImportError:
    logger.warning("Google Generative AI SDK not installed. AI responses will be basic.")
    GOOGLE_AI_SDK_AVAILABLE = False
except Exception as e:
    logger.error(f"Error configuring Google Generative AI: {e}. AI responses will be basic.")
    GOOGLE_AI_SDK_AVAILABLE = False
    
class ChatManager:
    def __init__(self, trading_engine_ref):
        global GOOGLE_AI_SDK_AVAILABLE
        
        self.trading_engine = trading_engine_ref
        self.model = None
        self.max_history_length = 5

        if GOOGLE_AI_SDK_AVAILABLE: 
            try:
                self.model = genai.GenerativeModel(settings.GOOGLE_AI_MODEL)
                logger.info(f"Google AI Model '{settings.GOOGLE_AI_MODEL}' loaded for chat.")
            except Exception as e:
                logger.error(f"Failed to load Google AI Model '{settings.GOOGLE_AI_MODEL}': {e}. AI responses will be basic.", exc_info=True)
                GOOGLE_AI_SDK_AVAILABLE = False

        if not GOOGLE_AI_SDK_AVAILABLE: 
            logger.warning("Google AI features are disabled. Responses will be based on hardcoded commands.")

        self.chat_history = []

    async def process_chat_message(self, message: str) -> Dict[str, Any]:
        logger.debug(f"ChatManager: Processing message: '{message}'")
        
        message_lower = message.lower().strip()
        response_text = ""
        action_performed = None

        # DEBUG: Log the processing
        logger.debug(f"ChatManager: Message lowered: '{message_lower}'")
        logger.debug(f"ChatManager: Trading engine available: {self.trading_engine is not None}")

        # --- Hardcoded Commands ---
        if message_lower == "help":
            response_text = "Available commands:\n" \
                            "- `status`: Get bot's current operational status.\n" \
                            "- `balance`: Check current account balance.\n" \
                            "- `positions`: View open trading positions.\n" \
                            "- `performance`: Get bot's performance metrics.\n" \
                            "- `active strategies`: List currently active trading strategies.\n" \
                            "- `available strategies`: List strategies available for activation.\n" \
                            "- `start trading`: Start the main trading loop.\n" \
                            "- `stop trading`: Stop the main trading loop.\n" \
                            "- `train models`: Trigger ML model training (if enabled).\n" \
                            "- For other questions, try asking Google AI (if configured)."
            action_performed = {"command": "help"}
            logger.debug("ChatManager: Processed help command")
            
        elif message_lower == "status":
            logger.debug("ChatManager: Processing status command")
            if self.trading_engine:
                try:
                    logger.debug("ChatManager: Getting status from trading engine")
                    status_data = self.trading_engine.get_status()
                    logger.debug(f"ChatManager: Status data received: {type(status_data)}")
                    
                    # Enhanced status response with safer access
                    running = status_data.get('running', False)
                    alpaca_enabled = status_data.get('alpaca_enabled', False)
                    active_strategies_count = status_data.get('active_strategies_count', 0)
                    ml_models_loaded = status_data.get('ml_engine_models_loaded', 0)
                    
                    response_text = f"Bot Status: {'Running' if running else 'Stopped'}\n" \
                                    f"Alpaca Enabled: {alpaca_enabled}\n" \
                                    f"Active Strategies: {active_strategies_count}\n" \
                                    f"ML Models Loaded: {ml_models_loaded}"
                    action_performed = {"command": "status", "data": status_data}
                    logger.debug(f"ChatManager: Status response generated: {response_text[:50]}...")
                except Exception as e:
                    logger.error(f"ChatManager: Error getting status: {e}", exc_info=True)
                    response_text = f"Error getting status: {str(e)}"
                    action_performed = {"command": "status_error", "error": str(e)}
            else:
                logger.warning("ChatManager: Trading engine not initialized")
                response_text = "Trading engine not initialized. Cannot get status."
                action_performed = {"command": "status_error"}
                
        elif message_lower == "balance":
            logger.debug("ChatManager: Processing balance command")
            if self.trading_engine:
                try:
                    status_data = self.trading_engine.get_status()
                    balance_data = status_data.get('balances', {})
                    if balance_data:
                        response_text = "Current Balances:\n" + "\n".join([f"- {k}: {v:.2f}" for k,v in balance_data.items()])
                    else:
                        response_text = "No balance data available."
                    action_performed = {"command": "balance", "data": balance_data}
                except Exception as e:
                    logger.error(f"ChatManager: Error getting balance: {e}")
                    response_text = f"Error getting balance: {str(e)}"
                    action_performed = {"command": "balance_error", "error": str(e)}
            else:
                response_text = "Trading engine not initialized. Cannot get balance."
                action_performed = {"command": "balance_error"}
                
        elif message_lower == "positions":
            logger.debug("ChatManager: Processing positions command")
            if self.trading_engine:
                try:
                    status_data = self.trading_engine.get_status()
                    positions_data = status_data.get('positions', {})
                    if positions_data:
                        response_text = "Open Positions:\n" + "\n".join([
                            f"- {s}: {p['amount']:.4f} @ {p['entry_price']:.2f} (Current: {self.trading_engine.current_market_data.get(s, {}).get('price', 'N/A'):.2f})"
                            for s, p in positions_data.items()
                        ])
                    else:
                        response_text = "No open positions."
                    action_performed = {"command": "positions", "data": positions_data}
                except Exception as e:
                    logger.error(f"ChatManager: Error getting positions: {e}")
                    response_text = f"Error getting positions: {str(e)}"
                    action_performed = {"command": "positions_error", "error": str(e)}
            else:
                response_text = "Trading engine not initialized. Cannot get positions."
                action_performed = {"command": "positions_error"}
                
        elif message_lower == "performance" or message_lower == "total value" or message_lower == "net worth":
            logger.debug("ChatManager: Processing performance command")
            if self.trading_engine:
                try:
                    performance_data = self.trading_engine.get_performance_metrics()
                    total_value = performance_data.get('total_account_value', 'N/A')
                    if isinstance(total_value, (int, float)):
                        response_text = f"Current Portfolio Value: ${total_value:.2f} USD"
                    else:
                        response_text = f"Current Portfolio Value: {total_value}"
                    action_performed = {"command": "performance", "data": performance_data}
                except Exception as e:
                    logger.error(f"ChatManager: Error getting performance: {e}")
                    response_text = f"Error getting performance: {str(e)}"
                    action_performed = {"command": "performance_error", "error": str(e)}
            else:
                response_text = "Trading engine not initialized. Cannot get performance."
                action_performed = {"command": "performance_error"}
                
        elif message_lower == "active strategies":
            logger.debug("ChatManager: Processing active strategies command")
            if self.trading_engine:
                try:
                    strategies = self.trading_engine.list_active_strategies()
                    if strategies:
                        response_text = "Active Strategies:\n" + "\n".join([
                            f"- {s.get('id', 'Unknown')} ({s.get('type', 'Unknown')}) for {s.get('symbol', 'Unknown')}" 
                            for s in strategies
                        ])
                    else:
                        response_text = "No strategies currently active."
                    action_performed = {"command": "active_strategies", "data": strategies}
                except Exception as e:
                    logger.error(f"ChatManager: Error getting active strategies: {e}")
                    response_text = f"Error getting active strategies: {str(e)}"
                    action_performed = {"command": "active_strategies_error", "error": str(e)}
            else:
                response_text = "Trading engine not initialized. Cannot list active strategies."
                action_performed = {"command": "active_strategies_error"}
                
        elif message_lower == "available strategies":
            logger.debug("ChatManager: Processing available strategies command")
            if self.trading_engine:
                try:
                    strategies = self.trading_engine.list_available_strategies()
                    if strategies and isinstance(strategies, list):
                        response_text = "Available Strategy Types:\n" + "\n".join([
                            f"- {s.get('name', 'Unknown')}: {s.get('description', 'No description')}" 
                            for s in strategies
                        ])
                    else:
                        response_text = "No strategy types found. Check strategies directory."
                    action_performed = {"command": "available_strategies", "data": strategies}
                except Exception as e:
                    logger.error(f"ChatManager: Error getting available strategies: {e}")
                    response_text = f"Error getting available strategies: {str(e)}"
                    action_performed = {"command": "available_strategies_error", "error": str(e)}
            else:
                response_text = "Trading engine not initialized. Cannot list available strategies."
                action_performed = {"command": "available_strategies_error"}
                
        elif message_lower == "start trading":
            logger.debug("ChatManager: Processing start trading command")
            if self.trading_engine:
                try:
                    if not self.trading_engine.running:
                        await self.trading_engine.start()
                        response_text = "Trading engine started."
                    else:
                        response_text = "Trading engine is already running."
                    action_performed = {"command": "start_trading"}
                except Exception as e:
                    logger.error(f"ChatManager: Error starting trading: {e}")
                    response_text = f"Error starting trading: {str(e)}"
                    action_performed = {"command": "start_trading_error", "error": str(e)}
            else:
                response_text = "Trading engine not initialized. Cannot start trading."
                action_performed = {"command": "start_trading_error"}
                
        elif message_lower == "stop trading":
            logger.debug("ChatManager: Processing stop trading command")
            if self.trading_engine:
                try:
                    if self.trading_engine.running:
                        await self.trading_engine.stop()
                        response_text = "Trading engine stopped."
                    else:
                        response_text = "Trading engine is not running."
                    action_performed = {"command": "stop_trading"}
                except Exception as e:
                    logger.error(f"ChatManager: Error stopping trading: {e}")
                    response_text = f"Error stopping trading: {str(e)}"
                    action_performed = {"command": "stop_trading_error", "error": str(e)}
            else:
                response_text = "Trading engine not initialized. Cannot stop trading."
                action_performed = {"command": "stop_trading_error"}
                
        elif message_lower == "train models":
            response_text = "ML model training initiated. Please monitor the console where the 'train_models.py --full-train' script is run. This command only triggers, training progresses in separate process."
            action_performed = {"command": "train_models_info"}
            logger.debug("ChatManager: Processed train models command")
            
        else:
            logger.debug(f"ChatManager: Unrecognized command, trying AI: {message_lower}")
            # --- Google AI Response ---
            if GOOGLE_AI_SDK_AVAILABLE and self.model: 
                try:
                    logger.debug("ChatManager: Using Google AI for response")
                    history_for_ai = []
                    for user_msg, bot_resp in self.chat_history:
                        history_for_ai.append({"role": "user", "parts": [user_msg]})
                        history_for_ai.append({"role": "model", "parts": [bot_resp]})

                    chat = self.model.start_chat(history=history_for_ai)
                    ai_response = await chat.send_message(message)
                    response_text = ai_response.text
                    action_performed = {"command": "ai_response"}
                    logger.debug(f"ChatManager: AI response generated: {response_text[:50]}...")

                except Exception as e:
                    logger.error(f"Google AI failed to generate response: {e}", exc_info=True)
                    response_text = f"I'm sorry, I couldn't process that with AI. (Error: {e}) Please try simpler commands or configure Google AI."
                    action_performed = {"command": "ai_failed", "error": str(e)}
            else:
                logger.debug("ChatManager: No AI available, using fallback response")
                response_text = f"I received: '{message}'. Type 'help' for available commands or configure Google AI for intelligent responses!"
                action_performed = {"command": "unrecognized", "message": message}

        # Update chat history
        if action_performed and action_performed.get("command") not in ["ai_failed", "unrecognized"]:
             self.chat_history.append((message, response_text))
             if len(self.chat_history) > self.max_history_length:
                 self.chat_history.pop(0)

        result = {
            "response": response_text,
            "action": action_performed
        }
        
        logger.debug(f"ChatManager: Final response: {result}")
        return result