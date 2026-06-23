"""Streaming VAD endpointing for websocket PCM frames."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from silero_vad import get_speech_timestamps, load_silero_vad
    import torch
except Exception:
    get_speech_timestamps = None
    load_silero_vad = None
    torch = None


class StreamingVadSession:
    """Converts continuous PCM16 frames into complete speech segments."""

    def __init__(self, cfg: dict):
        audio_cfg = cfg["audio"]
        self.sample_rate = int(audio_cfg["sample_rate"])
        self.threshold = float(audio_cfg.get("vad_threshold", 0.5))
        self.window_samples = self._ms_to_samples(audio_cfg.get("vad_window_ms", 256))
        self.pre_roll_samples = self._ms_to_samples(audio_cfg.get("vad_pre_roll_ms", 160))
        self.min_speech_samples = self._ms_to_samples(audio_cfg.get("vad_min_speech_ms", 300))
        self.min_silence_samples = self._ms_to_samples(audio_cfg.get("vad_min_silence_ms", 280))
        self.speech_pad_samples = self._ms_to_samples(audio_cfg.get("vad_speech_pad_ms", 80))
        self.max_segment_samples = int(self.sample_rate * float(audio_cfg.get("vad_max_segment_s", 8.0)))
        self.analysis_buffer = np.array([], dtype=np.float32)
        self.pre_roll = np.array([], dtype=np.float32)
        self.segment: list[np.ndarray] = []
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
        completed: list[np.ndarray] = []
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
        if not self.speaking:
            return None
        if len(self.analysis_buffer):
            self._append(self.analysis_buffer)
            self.analysis_buffer = np.array([], dtype=np.float32)
        return self._flush()

    def _is_speech(self, window: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(window)) + 1e-9))
        if rms < self.energy_floor:
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

    def _ms_to_samples(self, value_ms: int | float) -> int:
        return max(1, int(self.sample_rate * float(value_ms) / 1000.0))
