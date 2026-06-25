"""FastAPI app for the OneVoice Edge web experience."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from onevoice.config.settings import PROJECT_ROOT
from onevoice.realtime.incremental import StableTextBuffer
from onevoice.realtime.vad import StreamingVadSession
from onevoice.web.audio import decode_upload
from onevoice.web.debug_log import read_recent_debug_events, write_debug_event
from onevoice.web.schemas import ClientLogRequest, PrepareTtsRequest, SpeakRequest, TranslateRequest
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
    started = time.perf_counter()
    wav, metrics = service.synthesize_wav_with_metrics(
        req.text,
        direction=req.direction,
        emotion=req.emotion,
        original_text=req.original_text,
    )
    metrics["endpoint_ms"] = round((time.perf_counter() - started) * 1000, 1)
    metrics["request_id"] = req.request_id
    write_debug_event(
        side="backend",
        event="tts_synthesized",
        session_id=req.session_id,
        segment_id=req.segment_id,
        details=metrics,
    )
    headers = {
        "X-OneVoice-TTS-Engine": str(metrics.get("engine", "unknown")),
        "X-OneVoice-TTS-Inference-Ms": str(metrics.get("inference_ms", metrics.get("inference_wall_ms", 0))),
        "X-OneVoice-TTS-RTF": str(metrics.get("rtf", 0)),
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


@app.post("/api/tts/prepare")
def prepare_tts(req: PrepareTtsRequest):
    result = service.prepare_tts(req.direction)
    write_debug_event(
        side="backend",
        event="tts_prepared",
        session_id=req.session_id,
        details=result,
    )
    return result


@app.post("/api/client-log")
def client_log(req: ClientLogRequest):
    write_debug_event(
        side="ui",
        event=req.event,
        session_id=req.session_id,
        segment_id=req.segment_id,
        details=req.details,
    )
    return {"ok": True}


@app.get("/api/debug-log/latest")
def latest_debug_log(lines: int = 300):
    return {"events": read_recent_debug_events(lines)}


@app.websocket("/ws/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("client_session") or uuid.uuid4().hex
    session = StreamingVadSession(service.cfg)
    stream_cfg = service.cfg.get("streaming", {})
    direction = "vi2en"
    segment_queue: asyncio.Queue[tuple[int | None, np.ndarray | None, str | None]] = asyncio.Queue(maxsize=4)
    sender_lock = asyncio.Lock()
    next_segment_id = 0
    active_segment_id: int | None = None
    stable_buffers: dict[int, StableTextBuffer] = {}
    frame_count = 0
    byte_count = 0
    started_at = time.perf_counter()
    partial_enabled = bool(stream_cfg.get("partial_asr_enabled", True))
    partial_interval_s = max(0.35, float(stream_cfg.get("partial_interval_ms", 900)) / 1000.0)
    partial_min_samples = int(service.cfg["audio"]["sample_rate"] * float(stream_cfg.get("partial_min_audio_ms", 900)) / 1000.0)
    partial_max_samples = int(service.cfg["audio"]["sample_rate"] * float(stream_cfg.get("partial_max_audio_s", 4.0)))

    write_debug_event(
        side="backend",
        event="ws_connected",
        session_id=session_id,
        details={"client": websocket.client.host if websocket.client else None},
    )

    async def safe_send(payload: dict) -> bool:
        async with sender_lock:
            try:
                await websocket.send_json(payload)
                write_debug_event(
                    side="backend",
                    event=f"send_{payload.get('type')}_{payload.get('status', '')}".rstrip("_"),
                    session_id=session_id,
                    segment_id=payload.get("segment_id"),
                    details={
                        "payload_type": payload.get("type"),
                        "status": payload.get("status"),
                        "has_data": bool(payload.get("data")),
                    },
                )
                return True
            except Exception as exc:
                write_debug_event(
                    side="backend",
                    event="send_failed",
                    session_id=session_id,
                    segment_id=payload.get("segment_id"),
                    details={"error": str(exc), "payload_type": payload.get("type")},
                )
                return False

    def allocate_segment_id() -> int:
        nonlocal next_segment_id
        next_segment_id += 1
        return next_segment_id

    def active_or_new_segment_id() -> int:
        nonlocal active_segment_id
        if active_segment_id is None:
            active_segment_id = allocate_segment_id()
        return active_segment_id

    def get_stable_buffer(segment_id: int) -> StableTextBuffer:
        if segment_id not in stable_buffers:
            stable_buffers[segment_id] = StableTextBuffer(
                stable_repeats=int(stream_cfg.get("stable_repeats", 2)),
                holdback_words=int(stream_cfg.get("stable_holdback_words", 2)),
                min_commit_words=int(stream_cfg.get("stable_min_commit_words", 3)),
            )
        return stable_buffers[segment_id]

    async def translate_commit(segment_id: int, source_text: str, segment_direction: str):
        if not source_text:
            return
        await safe_send({"type": "commit", "segment_id": segment_id, "source_text": source_text})
        translated = await service.translate_text_async(source_text, segment_direction)
        await safe_send(
            {
                "type": "translation_partial",
                "segment_id": segment_id,
                "source_text": source_text,
                "translated_text": translated["translated_text"],
                "direction": segment_direction,
                "latency_ms": translated.get("latency_ms", {}),
            }
        )

    async def enqueue_segment(segment: np.ndarray, segment_direction: str):
        nonlocal active_segment_id
        segment_id = active_or_new_segment_id()
        active_segment_id = None
        if segment_queue.full():
            try:
                dropped_segment_id, _, _ = segment_queue.get_nowait()
                segment_queue.task_done()
                await safe_send(
                    {
                        "type": "segment",
                        "status": "dropped",
                        "segment_id": dropped_segment_id,
                        "detail": "Dropped stale segment under load",
                    }
                )
            except asyncio.QueueEmpty:
                pass
        await safe_send(
            {
                "type": "segment",
                "status": "accepted",
                "segment_id": segment_id,
                "audio_duration_s": round(len(segment) / service.cfg["audio"]["sample_rate"], 3),
            }
        )
        write_debug_event(
            side="backend",
            event="segment_enqueued",
            session_id=session_id,
            segment_id=segment_id,
            details={
                "direction": segment_direction,
                "samples": len(segment),
                "queue_size": segment_queue.qsize(),
            },
        )
        await segment_queue.put((segment_id, segment, segment_direction))

    async def segment_worker():
        while True:
            segment_id, segment, segment_direction = await segment_queue.get()
            if segment is None:
                segment_queue.task_done()
                return
            try:
                started = time.perf_counter()
                write_debug_event(
                    side="backend",
                    event="segment_processing_start",
                    session_id=session_id,
                    segment_id=segment_id,
                    details={"direction": segment_direction, "samples": len(segment)},
                )
                await safe_send({"type": "segment", "status": "processing", "segment_id": segment_id})
                result = await service.process_audio_async(segment, segment_direction or "vi2en")
                stable_buffer = stable_buffers.pop(int(segment_id), None)
                incremental_committed = False
                if stable_buffer is not None and stable_buffer.has_commits:
                    final_commit = stable_buffer.finalize(result.get("source_text", ""))
                    if final_commit is not None:
                        await translate_commit(int(segment_id), final_commit.text, segment_direction or "vi2en")
                    incremental_committed = True
                    result["incremental_committed"] = True
                    result["committed_source_text"] = (
                        stable_buffer.committed_text if final_commit is None else final_commit.committed_text
                    )
                else:
                    result["incremental_committed"] = False
                write_debug_event(
                    side="backend",
                    event="segment_processing_result",
                    session_id=session_id,
                    segment_id=segment_id,
                    details={
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                        "source_len": len(result.get("source_text", "")),
                        "translated_len": len(result.get("translated_text", "")),
                        "incremental_committed": incremental_committed,
                        "latency_ms": result.get("latency_ms", {}),
                    },
                )
                await safe_send({"type": "result", "segment_id": segment_id, "data": result})
                await safe_send({"type": "segment", "status": "done", "segment_id": segment_id})
            except HTTPException as exc:
                write_debug_event(
                    side="backend",
                    event="segment_ignored",
                    session_id=session_id,
                    segment_id=segment_id,
                    details={"detail": exc.detail},
                )
                await safe_send(
                    {
                        "type": "segment",
                        "status": "ignored",
                        "segment_id": segment_id,
                        "detail": exc.detail,
                    }
                )
                await safe_send({"type": "segment", "status": "done", "segment_id": segment_id})
            except Exception as exc:
                write_debug_event(
                    side="backend",
                    event="segment_error",
                    session_id=session_id,
                    segment_id=segment_id,
                    details={"error": str(exc)},
                )
                await safe_send({"type": "error", "segment_id": segment_id, "detail": str(exc)})
                await safe_send({"type": "segment", "status": "done", "segment_id": segment_id})
            finally:
                segment_queue.task_done()

    worker_task = asyncio.create_task(segment_worker())

    async def partial_worker():
        while True:
            await asyncio.sleep(partial_interval_s)
            if not partial_enabled or not session.speaking:
                continue
            segment_id = active_or_new_segment_id()
            audio = session.active_audio(max_samples=partial_max_samples)
            if len(audio) < partial_min_samples:
                continue
            try:
                result = await service.transcribe_partial_async(audio.copy(), direction)
                text = result.get("text", "").strip()
                if not text:
                    continue
                stable_buffer = get_stable_buffer(segment_id)
                commit = stable_buffer.update_partial(text)
                if commit is not None:
                    await translate_commit(segment_id, commit.text, direction)
                await safe_send(
                    {
                        "type": "partial_asr",
                        "segment_id": segment_id,
                        "text": text,
                        "unstable_text": stable_buffer.unstable_text or text,
                        "committed_text": stable_buffer.committed_text,
                        "direction": direction,
                        "latency_ms": result.get("latency_ms", {}),
                        "audio_duration_s": result.get("audio_duration_s", 0),
                    }
                )
            except Exception as exc:
                write_debug_event(
                    side="backend",
                    event="partial_asr_error",
                    session_id=session_id,
                    segment_id=segment_id,
                    details={"error": str(exc)},
                )

    partial_task = asyncio.create_task(partial_worker())

    try:
        await safe_send({"type": "ready", "session_id": session_id})
        while True:
            message = await websocket.receive()
            if "text" in message and message["text"] is not None:
                data = json.loads(message["text"])
                if data.get("type") == "config":
                    direction = data.get("direction", direction)
                    write_debug_event(
                        side="backend",
                        event="config_received",
                        session_id=session_id,
                        details={"direction": direction},
                    )
                    await safe_send({"type": "config", "direction": direction})
                elif data.get("type") == "stop":
                    write_debug_event(
                        side="backend",
                        event="stop_received",
                        session_id=session_id,
                        details={"analysis_buffer_samples": len(session.analysis_buffer)},
                    )
                    segment = session.flush()
                    if segment is not None:
                        await enqueue_segment(segment, direction)
                continue

            payload = message.get("bytes")
            if payload is None:
                continue
            frame_count += 1
            byte_count += len(payload)
            if frame_count == 1 or frame_count % 50 == 0:
                write_debug_event(
                    side="backend",
                    event="audio_frames_received",
                    session_id=session_id,
                    details={"frames": frame_count, "bytes": byte_count},
                )
            for segment in session.push_pcm16(payload):
                await enqueue_segment(segment, direction)
    except WebSocketDisconnect:
        write_debug_event(
            side="backend",
            event="ws_disconnected",
            session_id=session_id,
            details={
                "frames": frame_count,
                "bytes": byte_count,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 1),
            },
        )
    finally:
        try:
            await segment_queue.put((None, None, None))
        except Exception:
            pass
        worker_task.cancel()
        partial_task.cancel()
        write_debug_event(side="backend", event="ws_cleanup", session_id=session_id)


STATIC_DIR = PROJECT_ROOT / "app" / "web_static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
