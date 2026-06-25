"""API request schemas for the OneVoice web service."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str
    direction: str = "vi2en"
    speak: bool = False
    emotion: str = "neutral"


class SpeakRequest(BaseModel):
    text: str
    direction: str = "vi2en"
    emotion: str = "neutral"
    original_text: Optional[str] = None
    session_id: Optional[str] = None
    segment_id: Optional[int | str] = None
    request_id: Optional[str] = None


class PrepareTtsRequest(BaseModel):
    direction: str = "vi2en"
    session_id: Optional[str] = None


class ClientLogRequest(BaseModel):
    session_id: str
    event: str
    segment_id: Optional[int | str] = None
    details: dict = Field(default_factory=dict)
