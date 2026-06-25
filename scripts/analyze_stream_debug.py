"""Analyze correlated UI/backend stream debug logs.

Run after reproducing a UI stuck/missing-result case:
    python scripts/analyze_stream_debug.py --session latest
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


LOG_PATH = Path("logs/stream-debug.jsonl")


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Debug log not found: {path}")
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def choose_session(events: list[dict[str, Any]], session: str) -> str:
    sessions = [event.get("session_id") for event in events if event.get("session_id")]
    if not sessions:
        raise RuntimeError("No session_id values found in debug log.")
    if session != "latest":
        return session
    return sessions[-1]


def analyze(events: list[dict[str, Any]], session_id: str) -> None:
    session_events = [event for event in events if event.get("session_id") == session_id]
    if not session_events:
        raise RuntimeError(f"No events found for session: {session_id}")

    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in session_events:
        segment_id = event.get("segment_id")
        if segment_id is not None:
            by_segment[str(segment_id)].append(event)

    print(f"Session: {session_id}")
    print(f"Events : {len(session_events)}")
    print(f"UI     : {sum(1 for e in session_events if e.get('side') == 'ui')}")
    print(f"Backend: {sum(1 for e in session_events if e.get('side') == 'backend')}")
    print()

    if not by_segment:
        print("No segment events found. The browser may not have triggered VAD endpointing.")
    for segment_id, segment_events in sorted(by_segment.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        names = [event["event"] for event in segment_events]
        print(f"Segment {segment_id}: {' -> '.join(names)}")

        warnings = []
        if "segment_processing_result" in names and "send_result" not in names:
            warnings.append("backend produced ASR/MT result but did not log send_result")
        if "send_result" in names and "ws_message_result" not in names:
            warnings.append("backend sent result but UI did not log receiving it")
        if "ws_message_result" in names and "render_result" not in names:
            warnings.append("UI received result but did not render it")
        if "segment_processing_ui" in names and "segment_done_ui" not in names:
            warnings.append("UI entered processing state but never cleared segment")
        if "processing_watchdog_timeout" in names:
            warnings.append("UI watchdog cleared a stuck processing segment")
        if "send_failed" in names:
            warnings.append("backend WebSocket send failed")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    print()

    print("Last 20 events:")
    for event in session_events[-20:]:
        ts = event.get("ts")
        side = event.get("side")
        name = event.get("event")
        segment_id = event.get("segment_id")
        details = event.get("details", {})
        print(f"  {ts:.3f} {side:7} {name:32} seg={segment_id} details={details}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze OneVoice UI/backend stream debug logs.")
    parser.add_argument("--log", default=str(LOG_PATH))
    parser.add_argument("--session", default="latest", help="Session id or 'latest'.")
    args = parser.parse_args()

    events = load_events(Path(args.log))
    session_id = choose_session(events, args.session)
    analyze(events, session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
