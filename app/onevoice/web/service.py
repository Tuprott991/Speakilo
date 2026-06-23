"""Application service that owns model lifecycle and pipeline execution."""

from __future__ import annotations

import asyncio
import io
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import HTTPException

from onevoice.adapters.text import normalize
from onevoice.adapters.translation import Translator
from onevoice.adapters.tts import TTSEngine
from onevoice.config import load_config
from onevoice.realtime.pipeline import OneVoicePipeline


class OneVoiceService:
    """High-level service boundary for the local edge application."""

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.pipeline = OneVoicePipeline(config_path=config_path, direction="vi2en")
        self.translator: Translator = self.pipeline.translator
        self.tts: Optional[TTSEngine] = None
        self.ready = False
        self.loading_error: Optional[str] = None
        self._mt_lock = threading.Lock()
        self._tts_lock = threading.Lock()

    def load(self):
        """Load and warm up all models needed by the realtime UI."""
        if self.ready:
            return
        try:
            start = time.perf_counter()
            self.pipeline.start_workers(use_capture=False)
            self._warmup_translation()
            self.ready = True
            print(f"[Web] OneVoice models ready in {time.perf_counter() - start:.1f}s")
        except Exception as exc:
            self.loading_error = str(exc)
            print(f"[Web] Model loading failed: {exc}")
            raise

    def status(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.loading_error,
            "tts_loaded": self.tts is not None,
            "directions": ["vi2en", "en2vi"],
            "sample_rate": self.cfg["audio"]["sample_rate"],
        }

    def translate_text(self, text: str, direction: str, emotion: str = "neutral") -> dict:
        self.ensure_ready()
        lang = "vi" if direction == "vi2en" else "en"
        normalized = normalize(text, lang=lang)
        with self._mt_lock:
            start = time.perf_counter()
            translated = self.translator.translate(normalized, direction=direction)
            mt_ms = (time.perf_counter() - start) * 1000
        return {
            "source_text": normalized,
            "translated_text": translated,
            "direction": direction,
            "emotion": emotion,
            "latency_ms": {"mt": round(mt_ms, 1), "total": round(mt_ms, 1)},
        }

    def process_audio(self, audio: np.ndarray, direction: str) -> dict:
        self.ensure_ready()
        request_id = uuid.uuid4().hex
        self.pipeline.submit_audio(audio, direction=direction, request_id=request_id)
        result = self.pipeline.get_result(request_id=request_id, timeout=60.0)
        if result is None:
            raise HTTPException(status_code=504, detail="Pipeline timed out before producing a translation")
        if result.get("error"):
            raise HTTPException(status_code=422, detail=result["error"])
        return result

    async def process_audio_async(self, audio: np.ndarray, direction: str) -> dict:
        return await asyncio.to_thread(self.process_audio, audio, direction)

    def synthesize_wav(
        self,
        text: str,
        direction: str,
        emotion: str = "neutral",
        original_text: str | None = None,
    ) -> bytes:
        self.ensure_ready()
        tts = self._ensure_tts()
        audio, sample_rate = tts.synthesize(
            text,
            direction=direction,
            emotion=emotion,
            original_text=original_text,
        )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        buffer = io.BytesIO()
        sf.write(buffer, audio.astype(np.float32), sample_rate, format="WAV")
        return buffer.getvalue()

    def ensure_ready(self):
        if not self.ready:
            raise HTTPException(status_code=503, detail=self.loading_error or "Models are still loading")

    def _ensure_tts(self) -> TTSEngine:
        with self._tts_lock:
            if self.tts is None:
                self.tts = TTSEngine(self.cfg)
                self.tts.load()
        return self.tts

    def _warmup_translation(self):
        self.translator.translate("xin chao", direction="vi2en")
        self.translator.translate("hello", direction="en2vi")
