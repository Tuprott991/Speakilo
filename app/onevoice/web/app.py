"""FastAPI app for the OneVoice Edge web experience."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from onevoice.config.settings import PROJECT_ROOT
from onevoice.realtime.vad import StreamingVadSession
from onevoice.web.audio import decode_upload
from onevoice.web.schemas import SpeakRequest, TranslateRequest
from onevoice.web.service import OneVoiceService


service = OneVoiceService(config_path=PROJECT_ROOT / "config" / "config.yaml")
app = FastAPI(title="OneVoice Edge")


@app.on_event("startup")
def startup():
    service.load()


@app.get("/api/status")
def status():
    return service.status()


@app.post("/api/translate-text")
def translate_text(req: TranslateRequest):
    result = service.translate_text(req.text, req.direction, emotion=req.emotion)
    if req.speak:
        result["audio_url"] = "/api/speak"
    return result


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...), direction: str = Form("vi2en")):
    audio = await decode_upload(file, sample_rate=service.cfg["audio"]["sample_rate"])
    return service.process_audio(audio, direction=direction)


@app.post("/api/speak")
def speak(req: SpeakRequest):
    wav = service.synthesize_wav(
        req.text,
        direction=req.direction,
        emotion=req.emotion,
        original_text=req.original_text,
    )
    return Response(content=wav, media_type="audio/wav")


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

    async def enqueue_segment(segment: np.ndarray, segment_direction: str):
        if segment_queue.full():
            try:
                segment_queue.get_nowait()
                segment_queue.task_done()
                await safe_send({"type": "segment", "status": "dropped", "detail": "Dropped stale segment under load"})
            except asyncio.QueueEmpty:
                pass
        await segment_queue.put((segment, segment_direction))

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


STATIC_DIR = PROJECT_ROOT / "src" / "web_static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
