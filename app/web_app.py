"""Compatibility entrypoint for Uvicorn.

Run:
    conda activate onevoice
    python -m uvicorn app.web_app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

# On Windows, load the CPU ONNX Runtime DLLs before CUDA PyTorch. Loading them
# in the opposite order can resolve incompatible CUDA-adjacent DLL symbols and
# terminate the process before FastAPI starts.
import onnxruntime  # noqa: F401

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from onevoice.web.app import app, service

__all__ = ["app", "service"]
