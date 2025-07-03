from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

# Create router immediately
router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: str

@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    return ChatResponse(
        response=f"Echo: {request.message}",
        timestamp=datetime.now().isoformat()
    )

@router.get("/health")
async def chat_health():
    return {"status": "healthy"}
