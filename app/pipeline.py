"""Compatibility entrypoint for the production OneVoice package."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from onevoice.realtime.pipeline import OneVoicePipeline, main

__all__ = ["OneVoicePipeline", "main"]


if __name__ == "__main__":
    main()
