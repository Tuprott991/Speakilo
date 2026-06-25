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
        self._tts_inference_lock = threading.Lock()
        self._tts_warmup_started = False

    def load(self):
        """Load and warm up all models needed by the realtime UI."""
        if self.ready:
            return
        try:
            start = time.perf_counter()
            self.pipeline.start_workers(use_capture=False)
            self._warmup_translation()
            self.ready = True
            self._start_tts_background_warmup()
            print(f"[Web] OneVoice models ready in {time.perf_counter() - start:.1f}s")
        except Exception as exc:
            self.loading_error = str(exc)
            print(f"[Web] Model loading failed: {exc}")
            raise

    def status(self) -> dict:
        runtime_status = self.tts.runtime_status() if self.tts is not None else {}
        return {
            "ready": self.ready,
            "error": self.loading_error,
            "tts_loaded": bool(self.tts is not None and self.tts.loaded),
            "tts_warmup_started": self._tts_warmup_started,
            "tts_ui_enabled_by_default": bool(
                self.cfg.get("tts", {}).get("ui_enabled_by_default", False)
            ),
            "tts_max_pending_chunks": int(
                self.cfg.get("tts", {}).get("max_pending_chunks", 8)
            ),
            "tts_engines": {
                "vi": self.cfg.get("tts", {}).get("vi_engine", "kokoro"),
                "en": self.cfg.get("tts", {}).get("en_engine", "kokoro"),
                "kokoro_vi_version": self.cfg.get("tts", {}).get("kokoro_vi", {}).get("version"),
                "kokoro_en_version": self.cfg.get("tts", {}).get("kokoro_en", {}).get("version"),
                "kokoro_en_onnx_version": self.cfg.get("tts", {}).get("kokoro_en", {}).get("onnx_version"),
                "kokoro_en_backend": self.cfg.get("tts", {}).get("kokoro_en", {}).get("backend", "auto"),
            },
            "tts_runtime": runtime_status,
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

    def transcribe_partial(self, audio: np.ndarray, direction: str) -> dict:
        """Run a low-rate ASR snapshot for unstable live text preview."""
        self.ensure_ready()
        start = time.perf_counter()
        clean = self.pipeline.denoiser.denoise(audio)
        denoise_ms = (time.perf_counter() - start) * 1000
        asr_start = time.perf_counter()
        result = self.pipeline.asr.transcribe(clean, direction=direction)
        asr_ms = (time.perf_counter() - asr_start) * 1000
        if result.get("text"):
            result["text"] = normalize(result["text"], lang=result["lang"])
        result["latency_ms"] = {
            "denoise": round(denoise_ms, 1),
            "asr": round(asr_ms, 1),
            "total": round(denoise_ms + asr_ms, 1),
        }
        result["audio_duration_s"] = round(len(audio) / self.cfg["audio"]["sample_rate"], 3)
        return result

    async def transcribe_partial_async(self, audio: np.ndarray, direction: str) -> dict:
        return await asyncio.to_thread(self.transcribe_partial, audio, direction)

    async def translate_text_async(self, text: str, direction: str, emotion: str = "neutral") -> dict:
        return await asyncio.to_thread(self.translate_text, text, direction, emotion)

    def synthesize_wav(
        self,
        text: str,
        direction: str,
        emotion: str = "neutral",
        original_text: str | None = None,
    ) -> bytes:
        wav, _metrics = self.synthesize_wav_with_metrics(
            text,
            direction=direction,
            emotion=emotion,
            original_text=original_text,
        )
        return wav

    def synthesize_wav_with_metrics(
        self,
        text: str,
        direction: str,
        emotion: str = "neutral",
        original_text: str | None = None,
    ) -> tuple[bytes, dict]:
        self.ensure_ready()
        tts = self._ensure_tts()
        total_start = time.perf_counter()
        with self._tts_inference_lock:
            inference_start = time.perf_counter()
            audio, sample_rate = tts.synthesize(
                text,
                direction=direction,
                emotion=emotion,
                original_text=original_text,
            )
            inference_wall_ms = (time.perf_counter() - inference_start) * 1000
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        buffer = io.BytesIO()
        sf.write(buffer, audio.astype(np.float32), sample_rate, format="WAV")
        metrics = dict(tts.last_metrics)
        metrics.update(
            {
                "direction": direction,
                "inference_wall_ms": round(inference_wall_ms, 1),
                "response_total_ms": round((time.perf_counter() - total_start) * 1000, 1),
                "sample_rate": sample_rate,
            }
        )
        return buffer.getvalue(), metrics

    def prepare_tts(self, direction: str) -> dict:
        self.ensure_ready()
        started = time.perf_counter()
        tts = self._ensure_tts()
        with self._tts_inference_lock:
            tts.prepare(direction)
        return {
            "ready": True,
            "direction": direction,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "runtime": tts.runtime_status(),
        }

    def ensure_ready(self):
        if not self.ready:
            raise HTTPException(status_code=503, detail=self.loading_error or "Models are still loading")

    def _ensure_tts(self) -> TTSEngine:
        with self._tts_lock:
            if self.tts is None:
                self.tts = TTSEngine(self.cfg)
                self.tts.load()
        return self.tts

    def _start_tts_background_warmup(self):
        if not self.cfg.get("tts", {}).get("background_warmup", True):
            return
        if self._tts_warmup_started:
            return
        self._tts_warmup_started = True

        def warmup():
            try:
                self._ensure_tts()
            except Exception as exc:
                print(f"[Web] TTS background warmup failed: {exc}")

        threading.Thread(target=warmup, name="onevoice-tts-warmup", daemon=True).start()

    def _warmup_translation(self):
        self.translator.translate("xin chao", direction="vi2en")
        self.translator.translate("hello", direction="en2vi")
