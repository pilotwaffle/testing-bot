# ml/models.py
from pydantic import BaseModel
from typing import Optional

class MLTrainRequest(BaseModel):
    """Request model for ML training API."""
    model_type: str # e.g., "neural_network", "lorentzian", "social_sentiment", "risk_assessment"
    symbol: str = "BTC/USDT" # Crypto pair to train on

class ChatMessage(BaseModel):
    """Request model for chat API."""
    message: str