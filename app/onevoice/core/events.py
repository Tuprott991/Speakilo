"""Typed contracts shared by realtime pipeline and web transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SegmentMetadata:
    """Metadata carried alongside one speech segment."""

    direction: str = "vi2en"
    request_id: str | None = None
    created_at: float | None = None


@dataclass(slots=True)
class PipelineResult:
    """Result emitted after a speech segment is recognized and translated."""

    source_text: str
    translated_text: str
    direction: str
    request_id: str | None = None
    emotion: str = "neutral"
    event: str = "speech"
    audio_duration_s: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "direction": self.direction,
            "emotion": self.emotion,
            "event": self.event,
            "audio_duration_s": self.audio_duration_s,
            "latency_ms": self.latency_ms,
        }
