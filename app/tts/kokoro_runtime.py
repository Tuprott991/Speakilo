"""Latency-aware Kokoro runtime adapters for OneVoice TTS."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 24000
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    engine: str
    elapsed_ms: float


def _add_path(path: str | Path | None) -> None:
    if not path:
        return
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    path_str = str(resolved.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _resolve_device(device: str | None) -> str:
    requested = (device or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _configure_torch_threads(cfg: dict) -> None:
    try:
        import torch

        requested = int(cfg.get("num_threads", 0))
        if requested > 0 and torch.get_num_threads() != requested:
            torch.set_num_threads(requested)
    except Exception:
        return


def _clean_text(text: str, max_chars: int) -> str:
    normalized = " ".join((text or "").split())
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].strip() or normalized[:max_chars]


def _as_float32(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


class KokoroVietnameseRuntime:
    """Adapter for app/tts/Kokoro-Vietnamese Vietnamese fine-tuned Kokoro."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._engine = None
        self._lock = threading.Lock()
        self.sample_rate = SAMPLE_RATE
        self.device = "unloaded"

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def load(self) -> None:
        if self._engine is not None:
            return
        with self._lock:
            if self._engine is not None:
                return

            _add_path(self.cfg.get("package_dir", "app/tts/Kokoro-Vietnamese/src"))
            _configure_torch_threads(self.cfg)
            from kokoro_vietnamese import KokoroVietnamese

            device = _resolve_device(self.cfg.get("device"))
            self.device = device
            self._engine = KokoroVietnamese(
                repo_id=self.cfg.get("repo_id", "contextboxai/Kokoro-Vietnamese"),
                voice=self.cfg.get("voice", "diem_trinh"),
                model_path=self.cfg.get("model_path"),
                voicepack_path=self.cfg.get("voicepack_path"),
                config_path=self.cfg.get("config_path"),
                device=device,
            )
            print(
                "[TTS VI] Kokoro Vietnamese ready "
                f"(voice={self.cfg.get('voice', 'diem_trinh')}, device={device}, "
                f"version={self.cfg.get('version', '0.1.0')})"
            )

    def synthesize(self, text: str) -> SynthesisResult:
        self.load()
        clean = _clean_text(text, int(self.cfg.get("max_chars", 240)))
        if not clean:
            return SynthesisResult(np.zeros(0, dtype=np.float32), self.sample_rate, "kokoro_vi", 0.0)

        start = time.perf_counter()
        audio, _phonemes = self._engine.synthesize(
            clean,
            speed=float(self.cfg.get("speed", 1.0)),
            crossfade_ms=int(self.cfg.get("crossfade_ms", 50)),
            normalize_peak=self.cfg.get("normalize_peak", 0.95),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SynthesisResult(_as_float32(audio), self.sample_rate, "kokoro_vi", elapsed_ms)


class KokoroEnglishRuntime:
    """Adapter for the bundled Kokoro 0.9.4 English runtime."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._pipeline = None
        self._lock = threading.Lock()
        self.sample_rate = SAMPLE_RATE
        self.device = "unloaded"

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return

            _add_path(self.cfg.get("package_dir", "app/tts/Kokoro-Vietnamese/kokoro"))
            _configure_torch_threads(self.cfg)
            from kokoro import KPipeline, __version__

            device = _resolve_device(self.cfg.get("device"))
            self.device = device
            self._pipeline = KPipeline(
                lang_code=self.cfg.get("lang_code", "a"),
                repo_id=self.cfg.get("repo_id", "hexgrad/Kokoro-82M"),
                device=device,
            )
            print(
                "[TTS EN] Kokoro English ready "
                f"(voice={self.cfg.get('voice', 'af_heart')}, device={device}, version={__version__})"
            )

    def synthesize(self, text: str) -> SynthesisResult:
        self.load()
        clean = _clean_text(text, int(self.cfg.get("max_chars", 240)))
        if not clean:
            return SynthesisResult(np.zeros(0, dtype=np.float32), self.sample_rate, "kokoro_en", 0.0)

        start = time.perf_counter()
        chunks: list[np.ndarray] = []
        generator = self._pipeline(
            clean,
            voice=self.cfg.get("voice", "af_heart"),
            speed=float(self.cfg.get("speed", 1.0)),
            split_pattern=self.cfg.get("split_pattern", r"\n+"),
        )
        for result in generator:
            if result.audio is not None:
                chunks.append(_as_float32(result.audio))
        audio = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.zeros(0, dtype=np.float32)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SynthesisResult(audio, self.sample_rate, "kokoro_en", elapsed_ms)


class KokoroEnglishOnnxRuntime:
    """INT8 ONNX runtime for the stock English Kokoro voice."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._engine = None
        self._lock = threading.Lock()
        self.sample_rate = SAMPLE_RATE

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def load(self) -> None:
        if self._engine is not None:
            return
        with self._lock:
            if self._engine is not None:
                return

            model_path = self._resolve_path(self.cfg.get("onnx_model_path"))
            voices_path = self._resolve_path(self.cfg.get("onnx_voices_path"))
            if (
                not model_path.is_file()
                or model_path.stat().st_size <= 10 * 1024 * 1024
                or not voices_path.is_file()
                or voices_path.stat().st_size <= 10 * 1024 * 1024
            ):
                raise FileNotFoundError(
                    "Kokoro English ONNX assets are missing. Run "
                    "scripts/setup_optimized_tts.ps1 from the project root."
                )

            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            options = ort.SessionOptions()
            options.intra_op_num_threads = int(self.cfg.get("onnx_num_threads", 2))
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._engine = Kokoro.from_session(session, str(voices_path))
            print(
                "[TTS EN] Kokoro INT8 ONNX ready "
                f"(voice={self.cfg.get('voice', 'af_heart')}, model={model_path.name})"
            )

    def synthesize(self, text: str) -> SynthesisResult:
        self.load()
        clean = _clean_text(text, int(self.cfg.get("max_chars", 240)))
        if not clean:
            return SynthesisResult(np.zeros(0, dtype=np.float32), self.sample_rate, "kokoro_en_onnx_int8", 0.0)

        start = time.perf_counter()
        audio, sample_rate = self._engine.create(
            clean,
            voice=self.cfg.get("voice", "af_heart"),
            speed=float(self.cfg.get("speed", 1.0)),
            lang=self.cfg.get("onnx_lang", "en-us"),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SynthesisResult(
            _as_float32(audio),
            int(sample_rate),
            "kokoro_en_onnx_int8",
            elapsed_ms,
        )

    @staticmethod
    def _resolve_path(value: str | Path | None) -> Path:
        path = Path(value or "")
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
