# core/chat_bot.py
import logging
from typing import Optional, Dict, Any

from ai.gemini_ai import GeminiAI
from core.trading_engine import IndustrialTradingEngine
from core.config import settings # Global settings for GOOGLE_AI_ENABLED

logger = logging.getLogger(__name__)

class TradingChatBot:
    """Handles chat interactions, providing information and AI-powered responses."""
    def __init__(self, trading_engine: IndustrialTradingEngine, gemini_ai: Optional[GeminiAI]):
        self.engine = trading_engine
        self.gemini_ai = gemini_ai

    async def process_message(self, message: str) -> str:
        """Processes a chat message and returns a response."""
        message_lower = message.lower().strip()

        # Handle basic commands first for quick, reliable responses
        if message_lower in ['status', 'help', 'positions', 'start', 'stop', 'market', 'models']:
            return self._handle_basic_commands(message_lower)
        
        # Specific command for AI market analysis
        if message_lower == 'analyze' or 'market analysis' in message_lower:
            return await self._handle_ai_market_analysis()

        # For more complex queries, use AI if available
        if settings.GOOGLE_AI_ENABLED and self.gemini_ai:
            try:
                context = self._get_trading_context()
                response = await self.gemini_ai.generate_response(message, context)
                return f"🤖 AI Assistant: {response}"
            except Exception as e:
                logger.warning(f"AI response failed: {e}. Falling back to basic.", exc_info=False)
                return self._handle_fallback_response(message)
        else:
            return self._handle_fallback_response(message)

    def _handle_basic_commands(self, command: str) -> str:
        """Handles predefined basic commands without AI involvement."""
        if command == 'status':
            metrics = self.engine.get_performance_metrics()
            return f"""📊 Portfolio Status:
💰 Total Value: ${metrics['total_value']:.2f}
💵 Cash: ${metrics['cash_balance']:.2f}
📈 P&L: ${metrics['unrealized_pnl']:.2f}
🎯 Profit: ${metrics['total_profit']:.2f}
📊 Positions: {metrics['num_positions']}
🤖 Status: {'🟢 RUNNING' if self.engine.running else '🔴 STOPPED'}"""

        elif command == 'help':
            ai_status_text = "🤖 AI Enhanced" if settings.GOOGLE_AI_ENABLED else "📝 Basic Mode"
            return f"""🤖 Trading Bot Commands ({ai_status_text}):
• status - Portfolio overview
• positions - View open positions
• market - Current market data
• start/stop - Control trading
• models - List trained ML models
• analyze - AI market analysis (if AI enabled)
• Ask any trading question for AI analysis!"""

        elif command == 'positions':
            if not self.engine.positions:
                return "📊 No open positions currently."
            response_lines = ["📊 Current Positions:"]
            for symbol, pos in self.engine.positions.items():
                response_lines.append(f"• {symbol}: {pos['amount']:.6f} @ ${pos['entry_price']:.2f} PnL: ${pos['unrealized_pnl']:.2f}")
            return "\n".join(response_lines)

        elif command == 'market':
            if not self.engine.current_market_data:
                return "📈 No market data available yet. Data feed might be starting or not running."
            response_lines = ["📈 Live Market Prices:"]
            # Limit to top 5 for brevity in chat
            for symbol, data in list(self.engine.current_market_data.items())[:5]:
                price = data.get('price', 0)
                change = data.get('change_24h', 0)
                response_lines.append(f"• {symbol}: ${price:.2f} ({change:+.1f}%)")
            return "\n".join(response_lines)

        elif command == 'start':
            self.engine.start()
            return "🚀 Trading engine started! Bot is now monitoring markets."

        elif command == 'stop':
            self.engine.stop()
            return "⏹️ Trading engine stopped. Bot is in monitoring mode only."
        
        elif command == 'models':
            status = self.engine.ml_engine.get_model_status()
            if not status:
                return "🧠 No models trained yet. Use the ML Training section in dashboard to train models!"
            response_lines = ["🧠 Trained Models:"]
            for model_key, info in status.items():
                accuracy_val = info.get('metric_value')
                accuracy_str = "N/A"
                if isinstance(accuracy_val, (float, int)):
                    if info.get('metric_name') == 'Accuracy':
                        accuracy_str = f"{accuracy_val:.1%}"
                    elif info.get('metric_name') == 'R²':
                        accuracy_str = f"{accuracy_val:.3f}"
                    else:
                        accuracy_str = str(accuracy_val)

                response_lines.append(f"• {model_key} ({info.get('model_type', 'Unknown')}): {accuracy_str} {info.get('metric_name', '')}")
            return "\n".join(response_lines)

        return "Unknown command. Type 'help' for available commands."
    
    async def _handle_ai_market_analysis(self) -> str:
        """Handles AI market analysis request."""
        if settings.GOOGLE_AI_ENABLED and self.gemini_ai:
            if not self.engine.current_market_data:
                return "📊 I need more market data to perform an AI analysis. Please wait a moment."
            try:
                analysis = await self.gemini_ai.analyze_market_sentiment(self.engine.current_market_data)
                return f"🧠 AI Market Analysis:\n{analysis}"
            except Exception as e:
                logger.error(f"AI market analysis failed: {e}", exc_info=True)
                return f"❌ AI market analysis temporarily unavailable: {str(e)}"
        else:
            return "📊 For advanced AI analysis, please configure Google AI API key in your .env file."

    def _get_trading_context(self) -> str:
        """Gathers current trading context for AI to use."""
        try:
            metrics = self.engine.get_performance_metrics()
            ml_status = self.engine.ml_engine.get_model_status()
            active_strategies = self.engine.list_active_strategies() # Get raw config

            context = f"""Trading Bot Status:
- Portfolio Value: ${metrics['total_value']:.2f}
- P&L: ${metrics['unrealized_pnl']:.2f}
- Positions: {metrics['num_positions']}
- Status: {'RUNNING' if self.engine.running else 'STOPPED'}
- Trained ML Models: {len(ml_status)}
- Active Strategies: {len(active_strategies)}
- Available Symbols: {len(self.engine.data_fetcher.symbols)}
Current top 3 market prices:
"""
            for symbol, data in list(self.engine.current_market_data.items())[:3]:
                context += f"- {symbol}: ${data['price']:.2f} ({data['change_24h']:+.1f}%)\n"

            # Add more detail from active strategies
            if active_strategies:
                context += "\nDetails of Active Strategies (Config):\n"
                for i, strat_entry in enumerate(active_strategies):
                    context += f"  {i+1}. Name: {strat_entry.get('name')}, Symbol: {strat_entry.get('config', {}).get('symbol')}, Model: {strat_entry.get('config', {}).get('model_type')}\n"

            return context
        except Exception as e:
            logger.warning(f"Failed to get trading context for AI: {e}", exc_info=False)
            return "Trading bot operational, but detailed context for AI is unavailable."

    def _handle_fallback_response(self, message: str) -> str:
        """Provides a simple fallback response when AI is not available or command is unrecognized."""
        message_lower = message.lower()

        if 'train' in message_lower or 'ml' in message_lower:
            return """🧠 Available ML Models:
• neural_network - Deep learning price prediction
• lorentzian - Advanced k-NN classification
• social_sentiment - Social media analysis
• risk_assessment - Portfolio risk modeling

Use the ML Training section in dashboard to train models!"""
        elif any(word in message_lower for word in ['analyze', 'analysis', 'sentiment']):
            return "📊 For advanced AI analysis, please configure Google AI API key in your .env file."
        else:
            return f"💭 I received: '{message}'. Type 'help' for available commands or configure Google AI for intelligent responses!"