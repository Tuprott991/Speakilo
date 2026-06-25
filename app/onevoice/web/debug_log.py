"""Structured debug logging for browser/backend stream coordination."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from onevoice.config.settings import PROJECT_ROOT


LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "stream-debug.jsonl"
_LOCK = threading.Lock()


def write_debug_event(
    *,
    side: str,
    event: str,
    session_id: str | None = None,
    segment_id: int | str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Append one compact JSONL event for later stream analysis."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "side": side,
        "event": event,
        "session_id": session_id,
        "segment_id": segment_id,
        "details": _json_safe(details or {}),
    }
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def read_recent_debug_events(limit: int = 300) -> list[dict[str, Any]]:
    """Read the most recent debug events without loading huge logs forever."""
    if not LOG_PATH.exists():
        return []
    limit = max(1, min(int(limit), 2000))
    with LOG_PATH.open("r", encoding="utf-8") as file:
        lines = file.readlines()[-limit:]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
