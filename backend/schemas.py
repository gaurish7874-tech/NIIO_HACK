from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str
    status: str
    single_session: bool = True


class AnalysisResponse(BaseModel):
    session_id: str
    vitals: dict[str, Any]
    triage: dict[str, Any]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    response: str


class ClarificationRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)


class CalibrationResponse(BaseModel):
    session_id: str
    baseline: dict[str, Any]
    readings: int
    profile: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    ml_layers: dict[str, str]
