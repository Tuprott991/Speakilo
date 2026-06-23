"""Compatibility entrypoint for Uvicorn.

Run:
    conda activate onevoice
    python -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from onevoice.web.app import app, service

__all__ = ["app", "service"]
