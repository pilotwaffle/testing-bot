# ai/chat_manager.py - Enhanced Version
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import re
from dataclasses import dataclass, asdict
from enum import Enum

from .gemini_ai import GeminiAI
from .vector_db import VectorDB
from core.chat_bot import ChatBot

logger = logging.getLogger(__name__)

class MessageType(Enum):
    TEXT = "text"
    TRADE_SIGNAL = "trade_signal"
    ANALYSIS = "analysis"
    NOTIFICATION = "notification"
    COMMAND_RESULT = "command_result"
    ERROR = "error"

class Intent(Enum):
    TRADING_QUERY = "trading_query"
    PORTFOLIO_QUERY = "portfolio_query"
    ANALYSIS_REQUEST = "analysis_request"
    STRATEGY_MANAGEMENT = "strategy_management"
    GENERAL_CHAT = "general_chat"
    HELP_REQUEST = "help_request"
    LEARNING_QUERY = "learning_query"

@dataclass
class ChatMessage:
    content: str
    sender: str
    timestamp: datetime
    message_type: MessageType = MessageType.TEXT
    intent: Optional[Intent] = None
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class UserPreferences:
    communication_style: str = "detailed"  # brief, detailed, technical
    risk_tolerance: str = "medium"  # low, medium, high
    favorite_symbols: List[str] = None
    preferred_timeframes: List[str] = None
    notification_preferences: Dict[str, bool] = None
    response_format: str = "conversational"  # conversational, structured, bullet_points

    def __post_init__(self):
        if self.favorite_symbols is None:
            self.favorite_symbols = []
        if self.preferred_timeframes is None:
            self.preferred_timeframes = ["1h", "4h", "1d"]
        if self.notification_preferences is None:
            self.notification_preferences = {
                "trade_alerts": True,
                "market_updates": True,
                "strategy_updates": True,
                "risk_warnings": True
            }

class ConversationMemory:
    def __init__(self, max_short_term: int = 25, max_long_term_summaries: int = 10):
        self.short_term = deque(maxlen=max_short_term)
        self.long_term_summaries = deque(maxlen=max_long_term_summaries)
        self.topic_threads = {}
        self.session_start = datetime.now()
        
    def add_message(self, message: ChatMessage):
        self.short_term.append(message)
        self._update_topic_threads(message)
        
    def _update_topic_threads(self, message: ChatMessage):
        if message.intent:
            intent_name = message.intent.value
            if intent_name not in self.topic_threads:
                self.topic_threads[intent_name] = deque(maxlen=5)
            self.topic_threads[intent_name].append(message)
    
    def get_relevant_context(self, query_intent: Intent, limit: int = 10) -> List[ChatMessage]:
        # Get messages from relevant topic thread
        relevant_messages = []
        
        if query_intent and query_intent.value in self.topic_threads:
            relevant_messages.extend(list(self.topic_threads[query_intent.value]))
        
        # Add recent general messages
        recent_messages = list(self.short_term)[-limit:]
        for msg in recent_messages:
            if msg not in relevant_messages:
                relevant_messages.append(msg)
                
        return sorted(relevant_messages, key=lambda x: x.timestamp)[-limit:]
    
    def create_summary(self) -> str:
        if len(self.short_term) < 5:
            return ""
            
        messages = list(self.short_term)
        summary = f"Session started: {self.session_start.strftime('%H:%M')}\n"
        summary += f"Topics discussed: {', '.join(self.topic_threads.keys())}\n"
        summary += f"Total messages: {len(messages)}"
        return summary

class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            Intent.TRADING_QUERY: [
                r'\b(buy|sell|trade|order|position|entry|exit|long|short)\b',
                r'\b(market|limit|stop)\s*(order|price)\b',
                r'\b(execute|place|cancel)\s*(trade|order)\b'
            ],
            Intent.PORTFOLIO_QUERY: [
                r'\b(portfolio|balance|holdings|positions|value|worth)\b',
                r'\b(performance|profit|loss|pnl|returns)\b',
                r'\b(allocation|diversification|exposure)\b'
            ],
            Intent.ANALYSIS_REQUEST: [
                r'\b(analyze|analysis|predict|forecast|trend|pattern)\b',
                r'\b(technical|fundamental|sentiment)\s*analysis\b',
                r'\b(support|resistance|indicator|signal)\b'
            ],
            Intent.STRATEGY_MANAGEMENT: [
                r'\b(strategy|strategies|activate|deactivate|enable|disable)\b',
                r'\b(backtest|optimize|parameters|settings)\b',
                r'\b(ml|model|train|retrain)\b'
            ],
            Intent.HELP_REQUEST: [
                r'\b(help|how|what|explain|tutorial|guide)\b',
                r'\b(commands|options|features|capabilities)\b'
            ],
            Intent.LEARNING_QUERY: [
                r'\b(learn|teach|understand|clarify|definition)\b',
                r'\b(why|when|where|which|concept)\b'
            ]
        }
    
    def classify(self, message: str) -> Intent:
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        return Intent.GENERAL_CHAT

class EnhancedChatManager:
    def __init__(self, trading_engine, ml_engine, data_fetcher, notification_manager=None):
        self.trading_engine = trading_engine
        self.ml_engine = ml_engine
        self.data_fetcher = data_fetcher
        self.notification_manager = notification_manager
        
        # Initialize AI components
        self.gemini_ai = GeminiAI()
        self.vector_db = VectorDB()
        self.chat_bot = ChatBot()
        
        # Enhanced memory and intelligence
        self.memory = ConversationMemory()
        self.intent_classifier = IntentClassifier()
        self.user_preferences = UserPreferences()
        
        # Performance tracking
        self.response_times = deque(maxlen=100)
        self.satisfaction_scores = deque(maxlen=50)
        
        # Command handlers
        self.command_handlers = {
            "status": self._handle_status_command,
            "portfolio": self._handle_portfolio_command,
            "positions": self._handle_positions_command,
            "market": self._handle_market_command,
            "analyze": self._handle_analyze_command,
            "help": self._handle_help_command,
            "strategies": self._handle_strategies_command,
            "risk": self._handle_risk_command,
            "settings": self._handle_settings_command,
            "history": self._handle_history_command
        }
        
        logger.info("Enhanced ChatManager initialized with advanced AI features")

    async def process_message(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """Enhanced message processing with intent classification and personalization"""
        start_time = datetime.now()
        
        try:
            # Classify intent
            intent = self.intent_classifier.classify(message)
            
            # Create chat message object
            chat_msg = ChatMessage(
                content=message,
                sender="user",
                timestamp=start_time,
                intent=intent
            )
            
            # Add to memory
            self.memory.add_message(chat_msg)
            
            # Check for direct commands
            if message.startswith('/'):
                response = await self._handle_slash_command(message)
                response_type = MessageType.COMMAND_RESULT
            else:
                # Process with AI
                response = await self._process_ai_message(message, intent)
                response_type = MessageType.TEXT
            
            # Personalize response
            response = self._personalize_response(response)
            
            # Create response message
            response_msg = ChatMessage(
                content=response,
                sender="bot",
                timestamp=datetime.now(),
                message_type=response_type,
                intent=intent
            )
            
            # Add response to memory
            self.memory.add_message(response_msg)
            
            # Track performance
            response_time = (datetime.now() - start_time).total_seconds()
            self.response_times.append(response_time)
            
            # Check for proactive insights
            insights = await self._generate_proactive_insights()
            
            return {
                "response": response,
                "message_type": response_type.value,
                "intent": intent.value if intent else None,
                "response_time": response_time,
                "proactive_insights": insights,
                "suggestions": self._get_command_suggestions(message)
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I'm sorry, I encountered an error processing your message. Please try again.",
                "message_type": MessageType.ERROR.value,
                "error": str(e)
            }

    async def _process_ai_message(self, message: str, intent: Intent) -> str:
        """Process message with AI based on intent"""
        
        # Build enhanced context
        context = await self._build_enhanced_context(intent)
        
        # Get relevant conversation history
        relevant_history = self.memory.get_relevant_context(intent)
        history_context = self._format_conversation_history(relevant_history)
        
        # Enhanced prompt based on intent
        enhanced_prompt = self._build_intent_specific_prompt(message, intent, context, history_context)
        
        try:
            # Try Gemini AI first
            if self.gemini_ai:
                response = await self.gemini_ai.chat(enhanced_prompt)
                if response:
                    return response
        except Exception as e:
            logger.warning(f"Gemini AI failed: {e}")
        
        # Fallback to local chat bot with enhanced responses
        return await self._enhanced_fallback_response(message, intent, context)

    async def _build_enhanced_context(self, intent: Intent = None) -> str:
        """Build rich trading context based on current bot state"""
        try:
            # Get comprehensive status
            status = await self._get_comprehensive_status()
            
            # Build context based on intent
            context_parts = [
                "🤖 **Trading Bot Assistant Context**",
                f"📊 **Portfolio**: ${status.get('total_value', 0):.2f} ({status.get('change_24h', 0):+.2%} 24h)",
                f"🎯 **Risk Level**: {status.get('risk_level', 'Unknown')}/10",
                f"⚡ **Active Strategies**: {len(status.get('active_strategies', []))} running",
                f"📈 **Market Sentiment**: {status.get('market_sentiment', 'Neutral')}",
                f"🧠 **ML Models**: {status.get('ml_models_loaded', 0)} loaded and ready"
            ]
            
            # Add intent-specific context
            if intent == Intent.TRADING_QUERY:
                context_parts.extend([
                    f"💰 **Available Cash**: ${status.get('available_cash', 0):.2f}",
                    f"📊 **Open Positions**: {len(status.get('positions', {}))}",
                    f"🎲 **Max Risk per Trade**: {status.get('max_risk_per_trade', 'Not set')}"
                ])
            elif intent == Intent.PORTFOLIO_QUERY:
                context_parts.extend([
                    f"📈 **Today's P&L**: ${status.get('pnl_today', 0):.2f}",
                    f"🏆 **Best Strategy**: {status.get('best_strategy', 'None')}",
                    f"📊 **Win Rate**: {status.get('win_rate', 0):.1%}"
                ])
            elif intent == Intent.ANALYSIS_REQUEST:
                context_parts.extend([
                    f"📊 **Market Data**: {len(status.get('market_data', {}))} symbols",
                    f"🔍 **Analysis Models**: {status.get('analysis_models', 'Available')}",
                    f"⏰ **Last Analysis**: {status.get('last_analysis_time', 'Never')}"
                ])
            
            # Add recent alerts if any
            alerts = status.get('active_alerts', [])
            if alerts:
                context_parts.append(f"🚨 **Active Alerts**: {len(alerts)} requiring attention")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return "🤖 Trading Bot Assistant (Limited context available)"

    def _build_intent_specific_prompt(self, message: str, intent: Intent, context: str, history: str) -> str:
        """Build enhanced prompts based on intent classification"""
        
        base_prompt = f"""You are an advanced AI trading assistant with deep knowledge of cryptocurrency markets, trading strategies, and risk management.

Current Trading Context:
{context}

Recent Conversation:
{history}

User's Message: "{message}"
Intent Classification: {intent.value if intent else 'general_chat'}

Response Guidelines:
- Be helpful, accurate, and professional
- Use trading terminology appropriately
- Provide actionable insights when possible
- Include relevant data from the context
- If suggesting trades, always include risk warnings
- Format responses clearly with emojis for better readability"""

        # Intent-specific enhancements
        if intent == Intent.TRADING_QUERY:
            base_prompt += """

TRADING FOCUS:
- Analyze current market conditions before suggesting trades
- Always mention risk management
- Include specific entry/exit strategies if applicable
- Reference available cash and position limits
- Suggest position sizing based on risk tolerance"""

        elif intent == Intent.PORTFOLIO_QUERY:
            base_prompt += """

PORTFOLIO FOCUS:
- Provide detailed performance analysis
- Compare against benchmarks if possible
- Identify strengths and weaknesses in current allocation
- Suggest rebalancing opportunities
- Highlight top performing and underperforming assets"""

        elif intent == Intent.ANALYSIS_REQUEST:
            base_prompt += """

ANALYSIS FOCUS:
- Use technical analysis concepts
- Reference chart patterns and indicators
- Include both bullish and bearish scenarios
- Provide confidence levels for predictions
- Suggest specific timeframes for analysis"""

        elif intent == Intent.STRATEGY_MANAGEMENT:
            base_prompt += """

STRATEGY FOCUS:
- Explain strategy logic and parameters
- Provide performance metrics
- Suggest optimization opportunities
- Include backtesting insights
- Recommend when to activate/deactivate strategies"""

        return base_prompt

    async def _enhanced_fallback_response(self, message: str, intent: Intent, context: str) -> str:
        """Enhanced fallback responses based on intent"""
        
        fallback_responses = {
            Intent.TRADING_QUERY: "I understand you're asking about trading. Let me help you with that based on current market conditions.",
            Intent.PORTFOLIO_QUERY: "I can help you analyze your portfolio performance and provide insights.",
            Intent.ANALYSIS_REQUEST: "I'll provide you with market analysis based on available data.",
            Intent.STRATEGY_MANAGEMENT: "Let me help you with strategy management and optimization.",
            Intent.HELP_REQUEST: "Here are the available commands and features I can help you with:",
            Intent.GENERAL_CHAT: "I'm here to help with your trading questions and analysis."
        }
        
        base_response = fallback_responses.get(intent, "I'm here to help with your trading needs.")
        
        # Add context-aware information
        try:
            status = await self._get_comprehensive_status()
            
            if intent == Intent.TRADING_QUERY:
                return f"{base_response}\n\n📊 Current portfolio value: ${status.get('total_value', 0):.2f}\n💰 Available for trading: ${status.get('available_cash', 0):.2f}\n🎯 Active strategies: {len(status.get('active_strategies', []))}"
                
            elif intent == Intent.PORTFOLIO_QUERY:
                return f"{base_response}\n\n📈 Portfolio Performance:\n• Total Value: ${status.get('total_value', 0):.2f}\n• 24h Change: {status.get('change_24h', 0):+.2%}\n• Open Positions: {len(status.get('positions', {}))}"
                
            elif intent == Intent.HELP_REQUEST:
                return self._get_help_text()
        
        except Exception as e:
            logger.error(f"Error in fallback response: {e}")
        
        return base_response

    async def _handle_slash_command(self, message: str) -> str:
        """Handle slash commands with enhanced functionality"""
        parts = message[1:].split()
        command = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        if command in self.command_handlers:
            return await self.command_handlers[command](args)
        else:
            return f"Unknown command: /{command}\n\nType /help to see available commands."

    async def _handle_status_command(self, args: List[str]) -> str:
        """Enhanced status command with more details"""
        try:
            status = await self._get_comprehensive_status()
            
            return f"""🤖 **Trading Bot Status**
            
📊 **Portfolio Overview**
• Total Value: ${status.get('total_value', 0):.2f}
• 24h Change: {status.get('change_24h', 0):+.2%}
• Available Cash: ${status.get('available_cash', 0):.2f}

⚡ **System Status**
• Bot Status: {'🟢 Running' if status.get('running', False) else '🔴 Stopped'}
• Active Strategies: {len(status.get('active_strategies', []))}
• ML Models: {status.get('ml_models_loaded', 0)} loaded
• Market Feeds: {len(status.get('market_data', {}))} symbols

🎯 **Risk Management**
• Risk Level: {status.get('risk_level', 'Unknown')}/10
• Open Positions: {len(status.get('positions', {}))}
• Max Drawdown: {status.get('max_drawdown', 'N/A')}

⚠️ **Alerts**: {len(status.get('active_alerts', []))} active"""

        except Exception as e:
            return f"Error getting status: {e}"

    async def _handle_analyze_command(self, args: List[str]) -> str:
        """Enhanced analyze command with ML integration"""
        try:
            symbol = args[0] if args else "BTC/USDT"
            timeframe = args[1] if len(args) > 1 else "1h"
            
            # Get AI analysis
            if self.ml_engine:
                analysis = await self.ml_engine.analyze_symbol(symbol, timeframe)
                if analysis:
                    return f"""🔍 **AI Analysis for {symbol}**
                    
📊 **Technical Analysis**
• Trend: {analysis.get('trend', 'Unknown')}
• Signal: {analysis.get('signal', 'Neutral')}
• Confidence: {analysis.get('confidence', 0):.1%}

📈 **Key Levels**
• Support: ${analysis.get('support', 'N/A')}
• Resistance: ${analysis.get('resistance', 'N/A')}
• Current Price: ${analysis.get('current_price', 'N/A')}

🎯 **Recommendation**
{analysis.get('recommendation', 'No specific recommendation at this time.')}

⚠️ **Risk Assessment**: {analysis.get('risk_level', 'Medium')}"""
            
            return f"Analysis for {symbol} on {timeframe} timeframe requested. ML analysis temporarily unavailable."
            
        except Exception as e:
            return f"Error performing analysis: {e}"

    def _personalize_response(self, response: str) -> str:
        """Personalize response based on user preferences"""
        try:
            if self.user_preferences.communication_style == "brief":
                # Summarize long responses
                if len(response) > 300:
                    lines = response.split('\n')
                    key_lines = [line for line in lines if any(keyword in line.lower() 
                                for keyword in ['total', 'value', 'profit', 'loss', 'signal', 'alert'])]
                    return '\n'.join(key_lines[:5]) + "\n\n💡 Type /help for more details"
            
            elif self.user_preferences.communication_style == "technical":
                # Add technical details
                if "analysis" in response.lower():
                    response += "\n\n🔧 Technical note: Analysis includes RSI, MACD, Bollinger Bands, and custom ML indicators."
            
            # Add preferred symbols context
            if self.user_preferences.favorite_symbols:
                symbols_mentioned = any(symbol in response for symbol in self.user_preferences.favorite_symbols)
                if not symbols_mentioned and len(response) < 200:
                    fav_symbols = ", ".join(self.user_preferences.favorite_symbols[:3])
                    response += f"\n\n💡 Your favorite symbols: {fav_symbols}"
            
        except Exception as e:
            logger.error(f"Error personalizing response: {e}")
        
        return response

    async def _generate_proactive_insights(self) -> List[str]:
        """Generate proactive insights based on current market conditions"""
        insights = []
        
        try:
            status = await self._get_comprehensive_status()
            
            # Market opportunity detection
            if status.get('market_volatility', 0) > 0.05:
                insights.append("🔥 High volatility detected - consider adjusting position sizes")
            
            # Risk warnings
            if status.get('risk_level', 5) > 7:
                insights.append("⚠️ Portfolio risk elevated - consider reducing exposure")
            
            # Performance insights
            if status.get('win_rate', 0) < 0.4:
                insights.append("📊 Win rate below 40% - review strategy parameters")
            
            # Strategy suggestions
            active_strategies = len(status.get('active_strategies', []))
            if active_strategies == 0:
                insights.append("🎯 No active strategies - consider enabling trend-following or mean-reversion")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights

    def _get_command_suggestions(self, message: str) -> List[str]:
        """Get relevant command suggestions based on message content"""
        suggestions = []
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['status', 'running', 'bot']):
            suggestions.append("/status - Get comprehensive bot status")
        
        if any(word in message_lower for word in ['portfolio', 'balance', 'holdings']):
            suggestions.append("/portfolio - View portfolio details")
        
        if any(word in message_lower for word in ['analyze', 'analysis', 'chart']):
            suggestions.append("/analyze BTC/USDT - Get AI analysis")
        
        if any(word in message_lower for word in ['strategy', 'strategies']):
            suggestions.append("/strategies - Manage trading strategies")
        
        if any(word in message_lower for word in ['risk', 'danger', 'safe']):
            suggestions.append("/risk - View risk assessment")
        
        if any(word in message_lower for word in ['help', 'commands', 'what']):
            suggestions.append("/help - Show all available commands")
        
        return suggestions[:3]  # Limit to 3 suggestions

    async def _get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive trading bot status"""
        try:
            # This would integrate with your actual trading engine
            if hasattr(self.trading_engine, 'get_status'):
                return await self.trading_engine.get_status()
            
            # Fallback mock data for demonstration
            return {
                'running': True,
                'total_value': 10000.00,
                'change_24h': 0.025,
                'available_cash': 2500.00,
                'risk_level': 6,
                'active_strategies': ['momentum', 'mean_reversion'],
                'ml_models_loaded': 3,
                'market_data': {'BTC/USDT': {}, 'ETH/USDT': {}},
                'positions': {'BTC/USDT': {}},
                'active_alerts': [],
                'market_sentiment': 'Bullish',
                'pnl_today': 150.00,
                'best_strategy': 'momentum',
                'win_rate': 0.65,
                'market_volatility': 0.03,
                'max_drawdown': '5.2%',
                'last_analysis_time': '2 minutes ago'
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {}

    def _format_conversation_history(self, messages: List[ChatMessage], max_length: int = 500) -> str:
        """Format conversation history for context"""
        if not messages:
            return "No recent conversation history."
        
        formatted = []
        total_length = 0
        
        for msg in reversed(messages[-10:]):  # Last 10 messages
            msg_text = f"{msg.sender}: {msg.content[:100]}..." if len(msg.content) > 100 else f"{msg.sender}: {msg.content}"
            
            if total_length + len(msg_text) > max_length:
                break
                
            formatted.insert(0, msg_text)
            total_length += len(msg_text)
        
        return "\n".join(formatted)

    def _get_help_text(self) -> str:
        """Get comprehensive help text"""
        return """🤖 **Trading Bot Assistant Commands**

**📊 Portfolio & Status**
• `/status` - Complete bot status and metrics
• `/portfolio` - Detailed portfolio analysis
• `/positions` - Current open positions
• `/risk` - Risk assessment and recommendations

**📈 Analysis & Trading**
• `/analyze [SYMBOL] [TIMEFRAME]` - AI market analysis
• `/market` - Current market overview
• `/strategies` - Manage trading strategies

**⚙️ Settings & Help**
• `/settings` - Configure preferences
• `/history` - View conversation history
• `/help` - Show this help message

**💡 Natural Language**
You can also ask questions naturally:
• "What's my portfolio performance?"
• "Should I buy Bitcoin now?"
• "Analyze Ethereum on 4h timeframe"
• "What are the current risks?"

**🎯 Quick Tips**
• Type `/` to see command suggestions
• Use voice input with the microphone button
• Set your preferences with `/settings`"""

    # Additional handler methods would go here...
    async def _handle_portfolio_command(self, args: List[str]) -> str:
        """Handle portfolio command"""
        # Implementation here
        pass
    
    async def _handle_positions_command(self, args: List[str]) -> str:
        """Handle positions command"""
        # Implementation here
        pass
    
    async def _handle_market_command(self, args: List[str]) -> str:
        """Handle market command"""
        # Implementation here
        pass
    
    async def _handle_strategies_command(self, args: List[str]) -> str:
        """Handle strategies command"""
        # Implementation here
        pass
    
    async def _handle_risk_command(self, args: List[str]) -> str:
        """Handle risk command"""
        # Implementation here
        pass
    
    async def _handle_settings_command(self, args: List[str]) -> str:
        """Handle settings command"""
        # Implementation here
        pass
    
    async def _handle_history_command(self, args: List[str]) -> str:
        """Handle history command"""
        # Implementation here
        pass