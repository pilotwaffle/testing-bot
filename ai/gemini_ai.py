# ai/gemini_ai.py
import logging
from typing import Dict, Any, Optional

import asyncio # For asyncio.to_thread

logger = logging.getLogger(__name__)

# Import vector_db client (this assumes it's set up in ai/vector_db.py)
from ai.vector_db import VectorDBClient # Use Specific import, not from .


class GeminiAI:
    """Handles communication with the Google Gemini AI API."""
    def __init__(self, api_key: str, vector_db_client: Optional[VectorDBClient] = None): # Corrected type hint to Optional[VectorDBClient]
        if not api_key:
            raise ValueError("Google AI API Key is required for GeminiAI.")
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.session = None # aiohttp ClientSession
        self.vector_db = vector_db_client # Assign vector DB client

    async def _get_session(self):
        """Lazily create aiohttp ClientSession."""
        if self.session is None or self.session.closed:
            try:
                import aiohttp
                self.session = aiohttp.ClientSession()
            except ImportError:
                raise ImportError("aiohttp is required for GeminiAI. Please install it: pip install aiohttp")
        return self.session
        
    async def generate_response(self, prompt: str, bot_context: str = "", query_vector_db: bool = True) -> str:
        """
        Generates an AI response using the Gemini API.
        Can optionally query the vector database for additional context.
        """
        try:
            session = await self._get_session()
            url = f"{self.base_url}?key={self.api_key}"

            retrieved_context = ""
            if query_vector_db and self.vector_db and self.vector_db.is_ready:
                # Query vector DB to retrieve relevant documents (e.g., news, history)
                # Ensure query_documents is awaited if it's async
                rag_docs = await self.vector_db.query_documents(prompt, n_results=5)
                if rag_docs:
                    # Format documents for Gemini as direct text, or better, structured JSON/markdown
                    retrieved_context = "\n\nRelevant Knowledge Base Articles:\n" + \
                                        "\n---\n".join([doc['document'] for doc in rag_docs])

            enhanced_prompt = f"""You are an AI assistant for an industrial crypto trading bot.
Here is general context about the bot or market: {bot_context}
{retrieved_context} # Add retrieved context from vector DB if any

User Query: {prompt}

Provide a helpful, concise response about trading, market analysis, or bot operations.
Keep responses under 200 words and focus on actionable insights."""

            payload = {
                "contents": [
                    {"parts": [{"text": enhanced_prompt}]}
                ]
            }

            headers = {'Content-Type': 'application/json'}

            async with session.post(url, json=payload, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'candidates' in data and len(data['candidates']) > 0 and 'content' in data['candidates'][0] and 'parts' in data['candidates'][0]['content'] and len(data['candidates'][0]['content']['parts']) > 0:
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        return content.strip()
                    else:
                        logger.warning(f"Gemini API returned no candidates or content: {data}")
                        return "I couldn't generate a response. The AI model might have found the request unsafe or unanswerable."
                else:
                    error_detail = await response.text()
                    logger.error(f"Gemini API error (Status: {response.status}): {error_detail}")
                    return f"API Error {response.status}. Please check your connection or API key. Detail: {error_detail[:100]}"

        except ImportError:
            return "AI service requires aiohttp. Please install it with 'pip install aiohttp'."
        except asyncio.TimeoutError: # Catch asyncio.TimeoutError for aiohttp
            return "AI service timed out. The request took too long to complete."
        except Exception as e:
            logger.error(f"Gemini AI general error: {e}", exc_info=True)
            return f"AI service temporarily unavailable: {str(e)}"

    async def analyze_market_sentiment(self, market_data: Dict[str, Any]) -> str:
        """Analyzes market data and provides sentiment using AI."""
        # This will also benefit from vector DB context
        # The `generate_response` method is updated to include query_vector_db flag

        # Prepare market data summary (limit to top 5 for concise prompt)
        market_summary_lines = []
        for symbol, data in list(market_data.items())[:5]:
            change = data.get('change_24h', 0)
            price = data.get('price', 0)
            market_summary_lines.append(f"{symbol}: ${price:.2f} ({change:+.1f}%)")

        market_text = "\n".join(market_summary_lines)
        if not market_text:
            return "No recent market data available for analysis."

        prompt = f"""Analyze the following crypto market data and provide sentiment analysis:

{market_text}

Provide:
1. Overall market sentiment (bullish/bearish/neutral)
2. Key observations for a crypto trader
3. Potential short-term implications
4. Risk level assessment for these assets

Keep the response concise, under 150 words."""

        return await self.generate_response(prompt, "Market Analysis from live data", query_vector_db=True)

    async def close(self):
        """Closes the aiohttp ClientSession."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None # Clear the session reference
            logger.info("aiohttp session for GeminiAI closed.")