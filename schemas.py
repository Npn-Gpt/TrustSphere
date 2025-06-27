from pydantic import BaseModel, Field
from typing import List, Optional

class TrustRequest(BaseModel):
    text: str
    facial_emotion: Optional[str] = None  # Updated for broader compatibility
    audio_emotion: Optional[str] = None  # For audio-based emotion
    audio_weight: Optional[float] = None  # For audio influence weight

class TrustResponse(BaseModel):
    sentiment: str
    sentiment_score: float = Field(..., ge=0.0, le=1.0)
    embedding: List[float]
    trust_score: float = Field(..., ge=0.0, le=100.0)
    trust_reason: str
    is_spam: bool
    spam_confidence: float = Field(..., ge=0.0, le=1.0)
    facial_emotion: str = "N/A"
    audio_emotion: Optional[str] = None
    audio_weight: Optional[float] = None
