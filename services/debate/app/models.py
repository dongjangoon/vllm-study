from pydantic import BaseModel, Field
from typing import List, Optional


class DebateRequest(BaseModel):
    topic: str = Field(..., description="토론 주제")
    rounds: int = Field(default=3, ge=1, le=10, description="토론 라운드 수")
    language: str = Field(default="ko", description="응답 언어 (ko/en)")


class DebateTurn(BaseModel):
    round: int
    agent: str
    stance: str
    message: str


class DebateResponse(BaseModel):
    topic: str
    rounds: int
    turns: List[DebateTurn]
    trace_id: Optional[str] = None
    langfuse_url: Optional[str] = None
