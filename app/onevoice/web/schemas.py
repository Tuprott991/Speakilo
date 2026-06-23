"""API request schemas for the OneVoice web service."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


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
