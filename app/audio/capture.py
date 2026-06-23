"""
Audio capture and low-latency Silero VAD endpointing.

Captures microphone audio in real time, keeps a short pre-roll, and emits a
speech segment soon after silence is detected.
"""

import queue
import threading
import time as time_module

import numpy as np
import sounddevice as sd
import torch

VAD_MODEL = None
get_speech_timestamps = None


def _load_vad():
    """Load Silero VAD lazily so importing pipeline.py does not need network."""
    global VAD_MODEL, get_speech_timestamps

    if VAD_MODEL is not None and get_speech_timestamps is not None:
        return

    try:
        from silero_vad import (
            get_speech_timestamps as speech_timestamps_fn,
            load_silero_vad,
        )

        VAD_MODEL = load_silero_vad(onnx=True)
        get_speech_timestamps = speech_timestamps_fn
        return
    except ImportError:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad:master",
            model="silero_vad",
            force_reload=False,
            onnx=True,
            trust_repo=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load Silero VAD. Install the silero-vad package or "
            "connect to the internet once so torch.hub can cache "
            "snakers4/silero-vad, then rerun the pipeline."
        ) from exc

    VAD_MODEL = vad_model
    get_speech_timestamps = vad_utils[0]


class AudioCapture:
    """
    Captures microphone audio and segments it with low-latency endpointing.
    """

    def __init__(self, audio_queue: queue.Queue, config: dict):
        audio_cfg = config["audio"]
        self.q = audio_queue
        self.sample_rate = audio_cfg["sample_rate"]
        self.chunk_size = audio_cfg["chunk_size"]
        self.vad_threshold = audio_cfg["vad_threshold"]
        self.min_speech_ms = audio_cfg["vad_min_speech_ms"]
        self.vad_window_samples = int(self.sample_rate * audio_cfg.get("vad_window_ms", 256) / 1000)
        self.min_silence_samples = int(self.sample_rate * audio_cfg.get("vad_min_silence_ms", 280) / 1000)
        self.pre_roll_samples = int(self.sample_rate * audio_cfg.get("vad_pre_roll_ms", 160) / 1000)
        self.max_segment_samples = int(self.sample_rate * float(audio_cfg.get("vad_max_segment_s", 8.0)))

        self._running = False
        self._analysis_buffer = np.array([], dtype=np.float32)
        self._pre_roll = np.array([], dtype=np.float32)
        self._segment = []
        self._segment_samples = 0
        self._silence_samples = 0
        self._speaking = False
        self._last_vad_ts = 0.0

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        """Called by sounddevice for every audio chunk."""
        if status:
            print(f"[AudioCapture] Input status: {status}")

        audio = indata[:, 0].astype(np.float32)
        self._analysis_buffer = np.concatenate([self._analysis_buffer, audio])
        self._pre_roll = np.concatenate([self._pre_roll, audio])[-self.pre_roll_samples:]

        if len(self._analysis_buffer) < self.vad_window_samples:
            if self._speaking:
                self._append_segment(audio)
            return

        window = self._analysis_buffer[-self.vad_window_samples:].copy()
        self._analysis_buffer = self._analysis_buffer[-self.vad_window_samples:]
        speech_ts = get_speech_timestamps(
            torch.from_numpy(window),
            VAD_MODEL,
            threshold=self.vad_threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
            speech_pad_ms=0,
        )

        if speech_ts:
            self._last_vad_ts = time_module.perf_counter()
            self._silence_samples = 0
            if not self._speaking:
                self._speaking = True
                self._segment = []
                self._segment_samples = 0
                if len(self._pre_roll):
                    self._append_segment(self._pre_roll.copy())
            else:
                self._append_segment(audio)
        elif self._speaking:
            self._silence_samples += len(audio)
            self._append_segment(audio)
            if self._silence_samples >= self.min_silence_samples:
                self._flush_segment()

        if self._speaking and self._segment_samples >= self.max_segment_samples:
            self._flush_segment()

    def _append_segment(self, audio: np.ndarray):
        self._segment.append(audio)
        self._segment_samples += len(audio)

    def _flush_segment(self):
        if not self._segment:
            self._reset_segment()
            return

        segment = np.concatenate(self._segment).astype(np.float32)
        min_samples = int(self.sample_rate * self.min_speech_ms / 1000)
        if len(segment) >= min_samples and not self.q.full():
            self.q.put(segment)
        self._reset_segment()

    def _reset_segment(self):
        self._segment = []
        self._segment_samples = 0
        self._silence_samples = 0
        self._speaking = False

    def start(self):
        _load_vad()
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        print("[AudioCapture] Started (Silero VAD active, low-latency endpointing)")

    def _stream_loop(self):
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._callback,
        ):
            while self._running:
                sd.sleep(100)

    def stop(self):
        self._running = False
        print("[AudioCapture] Stopped")
