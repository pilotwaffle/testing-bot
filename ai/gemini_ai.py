"""
File: ai/gemini_ai.py
Location: E:\Trade Chat Bot\G Trading Bot\ai\gemini_ai.py

Google Gemini AI Integration for Trading Bot - FIXED VERSION
"""
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class GeminiAI:
    """Google Gemini AI integration for enhanced chat responses - FIXED"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini AI client with working model"""
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        self.enabled = bool(self.api_key)
        self.model_name = "gemini-1.5-flash"  # Updated to working model
        self.client = None
        
        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
                logger.info(f"✅ Gemini AI initialized successfully with model: {self.model_name}")
            except ImportError:
                logger.warning("❌ google-generativeai not installed. Install with: pip install google-generativeai")
                self.enabled = False
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini AI: {e}")
                self.enabled = False
        else:
            logger.warning("⚠️ Gemini AI disabled - no API key provided")
    
    async def chat(self, message: str, context: Optional[str] = None) -> Optional[str]:
        """Send message to Gemini AI and get response"""
        if not self.enabled or not self.client:
            return None
        
        try:
            # Build enhanced prompt for trading bot context
            prompt = self._build_prompt(message, context)
            
            # Generate response
            response = self.client.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("⚠️ Empty response from Gemini AI")
                return None
                
        except Exception as e:
            logger.error(f"❌ Gemini AI chat error: {e}")
            return None
    
    def _build_prompt(self, message: str, context: Optional[str] = None) -> str:
        """Build enhanced prompt for trading bot context"""
        base_prompt = """You are an advanced AI assistant for a cryptocurrency trading bot. You help users with:

- Portfolio analysis and performance metrics
- Market analysis and trading insights  
- Risk management and strategy advice
- Technical analysis and market trends
- General trading questions and education
- ML model training and optimization

Be helpful, accurate, and professional. Use trading terminology appropriately.
Always include risk warnings when discussing trades or investments.
Format responses clearly with emojis for better readability.
Keep responses concise but informative.

"""
        
        if context:
            base_prompt += f"\nCurrent Trading Context:\n{context}\n"
        
        base_prompt += f"\nUser Question: {message}\n"
        base_prompt += "\nAssistant Response:"
        
        return base_prompt
    
    def is_available(self) -> bool:
        """Check if Gemini AI is available and working"""
        if not self.enabled or not self.client:
            return False
        
        try:
            # Quick test to verify it's working
            response = self.client.generate_content("Test")
            return bool(response and response.text)
        except:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get Gemini AI status"""
        return {
            'enabled': self.enabled,
            'model': self.model_name,
            'api_key_configured': bool(self.api_key),
            'client_initialized': self.client is not None,
            'working': self.is_available()
        }
