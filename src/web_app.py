"""
FastAPI wrapper for the OneVoice PoC.

Run:
    conda run -n aic uvicorn src.web_app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from pipeline import OneVoicePipeline
from translation.mt_engine import Translator
from tts.tts_engine import TTSEngine
from utils.text_normalizer import normalize

try:
    from silero_vad import get_speech_timestamps, load_silero_vad
    import torch
except Exception:
    get_speech_timestamps = None
    load_silero_vad = None
    torch = None


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


class StreamingVadSession:
    """Server-side utterance endpointing for PCM16 websocket audio."""

    def __init__(self, cfg: dict):
        audio_cfg = cfg["audio"]
        self.sample_rate = int(audio_cfg["sample_rate"])
        self.threshold = float(audio_cfg.get("vad_threshold", 0.5))
        self.window_samples = int(self.sample_rate * 0.256)
        self.pre_roll_samples = int(self.sample_rate * 0.35)
        self.min_speech_samples = int(self.sample_rate * 0.45)
        self.min_silence_samples = int(self.sample_rate * 0.45)
        self.max_segment_samples = int(self.sample_rate * 8.0)
        self.analysis_buffer = np.array([], dtype=np.float32)
        self.pre_roll = np.array([], dtype=np.float32)
        self.segment = []
        self.segment_samples = 0
        self.silence_samples = 0
        self.speaking = False
        self.energy_floor = 0.010
        self.vad_model = load_silero_vad(onnx=True) if load_silero_vad else None

    def push_pcm16(self, payload: bytes) -> list[np.ndarray]:
        if not payload:
            return []
        audio = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
        return self.push_audio(audio)

    def push_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        completed = []
        self.analysis_buffer = np.concatenate([self.analysis_buffer, audio])

        while len(self.analysis_buffer) >= self.window_samples:
            window = self.analysis_buffer[: self.window_samples]
            self.analysis_buffer = self.analysis_buffer[self.window_samples :]
            has_speech = self._is_speech(window)
            if has_speech:
                self.silence_samples = 0
                if not self.speaking:
                    self.speaking = True
                    self.segment = []
                    self.segment_samples = 0
                    if len(self.pre_roll):
                        self._append(self.pre_roll.copy())
                self._append(window)
            elif self.speaking:
                self.silence_samples += len(window)
                self._append(window)
                if self.silence_samples >= self.min_silence_samples:
                    segment = self._flush()
                    if segment is not None:
                        completed.append(segment)
            else:
                self.pre_roll = np.concatenate([self.pre_roll, window])[-self.pre_roll_samples:]

            if self.speaking and self.segment_samples >= self.max_segment_samples:
                segment = self._flush()
                if segment is not None:
                    completed.append(segment)

        return completed

    def flush(self) -> Optional[np.ndarray]:
        if self.speaking:
            if len(self.analysis_buffer):
                self._append(self.analysis_buffer)
                self.analysis_buffer = np.array([], dtype=np.float32)
            return self._flush()
        return None

    def _is_speech(self, window: np.ndarray) -> bool:
        # Energy gate avoids Silero firing on near-silence and fan hum.
        if float(np.sqrt(np.mean(np.square(window)) + 1e-9)) < self.energy_floor:
            return False
        if self.vad_model is None or get_speech_timestamps is None or torch is None:
            return True
        speech = get_speech_timestamps(
            torch.from_numpy(window),
            self.vad_model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=80,
            speech_pad_ms=0,
        )
        return bool(speech)

    def _append(self, audio: np.ndarray):
        self.segment.append(audio)
        self.segment_samples += len(audio)

    def _flush(self) -> Optional[np.ndarray]:
        if not self.segment:
            self._reset()
            return None
        segment = np.concatenate(self.segment).astype(np.float32)
        self._reset()
        if len(segment) < self.min_speech_samples:
            return None
        return segment

    def _reset(self):
        self.segment = []
        self.segment_samples = 0
        self.silence_samples = 0
        self.speaking = False


class OneVoiceService:
    def __init__(self):
        self.cfg = load_config()
        self.pipeline = OneVoicePipeline(config_path="config/config.yaml", direction="vi2en")
        self.translator: Translator = self.pipeline.translator
        self.tts: Optional[TTSEngine] = None
        self.ready = False
        self.loading_error: Optional[str] = None
        self._lock = threading.Lock()
        self._tts_lock = threading.Lock()

    def load(self):
        try:
            t0 = time.perf_counter()
            self.pipeline.start_workers(use_capture=False)
            self._warmup()
            elapsed = time.perf_counter() - t0
            self.ready = True
            print(f"[Web] OneVoice models ready in {elapsed:.1f}s")
        except Exception as exc:
            self.loading_error = str(exc)
            print(f"[Web] Model loading failed: {exc}")
            raise

    def _warmup(self):
        self.translator.translate("xin chao", direction="vi2en")
        self.translator.translate("hello", direction="en2vi")

    def ensure_ready(self):
        if not self.ready:
            raise HTTPException(status_code=503, detail=self.loading_error or "Models are still loading")

    def ensure_tts(self):
        with self._tts_lock:
            if self.tts is None:
                self.tts = TTSEngine(self.cfg)
                self.tts.load()
        return self.tts

    def translate_text(self, text: str, direction: str, emotion: str = "neutral") -> dict:
        self.ensure_ready()
        lang = "vi" if direction == "vi2en" else "en"
        normalized = normalize(text, lang=lang)
        with self._lock:
            t0 = time.perf_counter()
            translated = self.translator.translate(normalized, direction=direction)
            mt_ms = (time.perf_counter() - t0) * 1000
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

    def synthesize_wav(self, text: str, direction: str, emotion: str = "neutral", original_text: str | None = None) -> bytes:
        self.ensure_ready()
        tts = self.ensure_tts()
        audio, sr = tts.synthesize(text, direction=direction, emotion=emotion, original_text=original_text)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        buf = io.BytesIO()
        sf.write(buf, audio.astype(np.float32), sr, format="WAV")
        return buf.getvalue()


service = OneVoiceService()
app = FastAPI(title="OneVoice Edge PoC")


@app.on_event("startup")
def startup():
    service.load()


@app.get("/api/status")
def status():
    return {
        "ready": service.ready,
        "error": service.loading_error,
        "tts_loaded": service.tts is not None,
        "directions": ["vi2en", "en2vi"],
    }


@app.post("/api/translate-text")
def translate_text(req: TranslateRequest):
    result = service.translate_text(req.text, req.direction, emotion=req.emotion)
    if req.speak:
        result["audio_url"] = "/api/speak"
    return result


@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    direction: str = Form("vi2en"),
):
    audio = await decode_upload(file)
    return service.process_audio(audio, direction=direction)


@app.websocket("/ws/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    session = StreamingVadSession(service.cfg)
    direction = "vi2en"
    segment_queue: asyncio.Queue[tuple[np.ndarray | None, str | None]] = asyncio.Queue(maxsize=4)
    sender_lock = asyncio.Lock()

    async def safe_send(payload: dict):
        async with sender_lock:
            await websocket.send_json(payload)

    async def segment_worker():
        while True:
            segment, segment_direction = await segment_queue.get()
            if segment is None:
                segment_queue.task_done()
                return
            try:
                await safe_send({"type": "segment", "status": "processing"})
                result = await service.process_audio_async(segment, segment_direction or "vi2en")
                await safe_send({"type": "result", "data": result})
            except HTTPException as exc:
                await safe_send({"type": "segment", "status": "ignored", "detail": exc.detail})
            except Exception as exc:
                await safe_send({"type": "error", "detail": str(exc)})
            finally:
                segment_queue.task_done()

    worker_task = asyncio.create_task(segment_worker())

    async def enqueue_segment(segment: np.ndarray, segment_direction: str):
        if segment_queue.full():
            try:
                segment_queue.get_nowait()
                segment_queue.task_done()
                await safe_send({"type": "segment", "status": "dropped", "detail": "Dropped stale segment under load"})
            except asyncio.QueueEmpty:
                pass
        await segment_queue.put((segment, segment_direction))

    try:
        await safe_send({"type": "ready"})
        while True:
            message = await websocket.receive()
            if "text" in message and message["text"] is not None:
                data = json.loads(message["text"])
                if data.get("type") == "config":
                    direction = data.get("direction", direction)
                    await safe_send({"type": "config", "direction": direction})
                elif data.get("type") == "stop":
                    segment = session.flush()
                    if segment is not None:
                        await enqueue_segment(segment, direction)
                continue

            payload = message.get("bytes")
            if payload is None:
                continue
            for segment in session.push_pcm16(payload):
                await enqueue_segment(segment, direction)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await segment_queue.put((None, None))
        except Exception:
            pass
        worker_task.cancel()


@app.post("/api/speak")
def speak(req: SpeakRequest):
    wav = service.synthesize_wav(
        req.text,
        direction=req.direction,
        emotion=req.emotion,
        original_text=req.original_text,
    )
    return Response(content=wav, media_type="audio/wav")


async def decode_upload(file: UploadFile) -> np.ndarray:
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        input_path = tmp.name
    try:
        try:
            import librosa

            audio, _ = librosa.load(input_path, sr=service.cfg["audio"]["sample_rate"], mono=True)
            return audio.astype(np.float32)
        except Exception:
            from pydub import AudioSegment

            seg = AudioSegment.from_file(input_path)
            seg = seg.set_channels(1).set_frame_rate(service.cfg["audio"]["sample_rate"])
            samples = np.array(seg.get_array_of_samples()).astype(np.float32)
            samples /= max(float(1 << (8 * seg.sample_width - 1)), 1.0)
            return samples
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass


STATIC_DIR = SRC_DIR / "web_static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
