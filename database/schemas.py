from pydantic import BaseModel, Field
from typing import Optional

class MLTrainRequest(BaseModel):
    """
    Request model for ML training API.
    """
    model_type: str = Field(..., description='Model type, e.g., "neural_network", "lorentzian", "social_sentiment", "risk_assessment"')
    symbol: str = Field("BTC/USDT", description="Crypto pair to train on")
    epochs: Optional[int] = Field(None, description="Number of training epochs")
    learning_rate: Optional[float] = Field(None, description="Learning rate for training")
    dataset_path: Optional[str] = Field(None, description="Path to training data")
    additional_params: Optional[dict] = Field(None, description="Other training parameters")

class ChatMessage(BaseModel):
    """
    Request model for chat API.
    """
    message: str = Field(..., description="The user's chat message")