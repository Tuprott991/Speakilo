"""Text-to-speech router for OneVoice.

The realtime translation path does not wait for TTS. This router loads Kokoro
lazily or through background warmup and serves queued committed translations
while the UI Voice toggle is enabled.
"""

from __future__ import annotations

import json
import gc
import os
import platform
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from .kokoro_runtime import (
    KokoroEnglishOnnxRuntime,
    KokoroEnglishRuntime,
    KokoroVietnameseRuntime,
    SynthesisResult,
)


class TTSEngine:
    """Direction-aware TTS router.

    direction="vi2en" means translated output is English.
    direction="en2vi" means translated output is Vietnamese.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("tts", {})
        self.sample_rate = int(config.get("audio", {}).get("sample_rate", 16000))
        self.vi_engine = self.cfg.get("vi_engine", "kokoro")
        self.en_engine = self.cfg.get("en_engine", "kokoro")
        self._kokoro_vi: Optional[KokoroVietnameseRuntime] = None
        self._kokoro_en: Optional[KokoroEnglishRuntime] = None
        self._kokoro_en_onnx: Optional[KokoroEnglishOnnxRuntime] = None
        self._kokoro_en_onnx_failed = False
        self._active_en_backend = "unloaded"
        self._en_selection_source = "none"
        self._active_output_language: str | None = None
        self._pyttsx3 = None
        self._loaded = False
        self._lock = threading.RLock()
        self.last_metrics: dict[str, float | str] = {}

    def load(self) -> None:
        """Prepare configured TTS engines.

        Callers can run this in a background thread to hide first-click latency.
        If an optional engine fails, synthesis still falls back gracefully.
        """
        with self._lock:
            if self._loaded:
                return
            start = time.perf_counter()
            print("[TTS] Preparing latency-aware TTS engines...")

            self.prepare(self.cfg.get("preload_direction", "vi2en"))

            self._loaded = True
            print(f"[TTS] TTS ready in {time.perf_counter() - start:.1f}s")

    @property
    def loaded(self) -> bool:
        return self._loaded

    def synthesize(
        self,
        text: str,
        direction: str = "vi2en",
        emotion: str = "neutral",
        reference_wav: str | None = None,
        original_text: str | None = None,
    ) -> tuple[np.ndarray, int]:
        del reference_wav, original_text
        if direction == "en2vi":
            return self.synthesize_vi(text, emotion=emotion)
        return self.synthesize_en(text)

    def synthesize_vi(self, text: str, emotion: str = "neutral") -> tuple[np.ndarray, int]:
        del emotion
        if self.vi_engine == "kokoro" and self.cfg.get("kokoro_vi", {}).get("enabled", True):
            try:
                self._activate_output_language("vi", warmup=False)
                result = self._ensure_kokoro_vi().synthesize(text)
                self._log_result(result, text)
                return self._nonempty(result.audio), result.sample_rate
            except Exception as exc:
                print(f"[TTS VI] Kokoro failed: {exc}")

        print("[TTS VI] No Vietnamese TTS available; returning short silence.")
        return self._silence()

    def synthesize_en(self, text: str) -> tuple[np.ndarray, int]:
        if self.en_engine == "kokoro" and self.cfg.get("kokoro_en", {}).get("enabled", True):
            self._activate_output_language("en", warmup=False)
            if (
                not self._kokoro_en_onnx_failed
                and self._active_en_backend == "onnx_int8"
            ):
                try:
                    result = self._ensure_kokoro_en_onnx().synthesize(text)
                    self._log_result(result, text)
                    return self._nonempty(result.audio), result.sample_rate
                except Exception as exc:
                    self._kokoro_en_onnx_failed = True
                    if self.cfg.get("kokoro_en", {}).get("backend") == "onnx":
                        print(f"[TTS EN] Kokoro ONNX failed: {exc}")
                    else:
                        print(f"[TTS EN] Kokoro ONNX unavailable; using PyTorch: {exc}")
            try:
                result = self._ensure_kokoro_en().synthesize(text)
                self._log_result(result, text)
                return self._nonempty(result.audio), result.sample_rate
            except Exception as exc:
                print(f"[TTS EN] Kokoro failed: {exc}")

        return self._synthesize_en_pyttsx3(text)

    def prepare(self, direction: str) -> None:
        output_language = "vi" if direction == "en2vi" else "en"
        self._activate_output_language(output_language, warmup=self.cfg.get("warmup", True))

    def runtime_status(self) -> dict:
        try:
            import torch

            torch_threads = torch.get_num_threads()
            cuda_available = torch.cuda.is_available()
            cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
            cuda_memory_mb = (
                round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
                if cuda_available
                else 0
            )
        except Exception:
            torch_threads = None
            cuda_available = False
            cuda_device = None
            cuda_memory_mb = 0
        return {
            "english_backend": self._active_en_backend,
            "english_selection_source": self._en_selection_source,
            "english_voice": self.cfg.get("kokoro_en", {}).get("voice"),
            "english_device": (
                self._kokoro_en.device if self._kokoro_en is not None else "unloaded"
            ),
            "vietnamese_backend": "pytorch",
            "vietnamese_voice": self.cfg.get("kokoro_vi", {}).get("voice"),
            "vietnamese_device": (
                self._kokoro_vi.device if self._kokoro_vi is not None else "unloaded"
            ),
            "active_output_language": self._active_output_language,
            "torch_threads": torch_threads,
            "cuda_available": cuda_available,
            "cuda_device": cuda_device,
            "cuda_memory_mb": cuda_memory_mb,
            "last_metrics": dict(self.last_metrics),
        }

    def play(self, audio: np.ndarray, sample_rate: int | None = None) -> None:
        sr = sample_rate or self.sample_rate
        try:
            sd.play(audio, samplerate=sr)
            sd.wait()
        except Exception as exc:
            print(f"[TTS] Playback error: {exc}")

    def run(self, text_queue) -> None:
        print("[TTS Worker] Started")
        while True:
            item = text_queue.get()
            try:
                text = item["text"]
                direction = item.get("direction", "vi2en")
                audio, sr = self.synthesize(text, direction=direction)
                self.play(audio, sample_rate=sr)
            finally:
                text_queue.task_done()

    def _ensure_kokoro_vi(self) -> KokoroVietnameseRuntime:
        with self._lock:
            if self._kokoro_vi is None:
                self._kokoro_vi = KokoroVietnameseRuntime(self.cfg.get("kokoro_vi", {}))
            self._kokoro_vi.load()
            return self._kokoro_vi

    def _ensure_kokoro_en(self) -> KokoroEnglishRuntime:
        with self._lock:
            if self._kokoro_en is None:
                self._kokoro_en = KokoroEnglishRuntime(self.cfg.get("kokoro_en", {}))
            self._kokoro_en.load()
            return self._kokoro_en

    def _ensure_kokoro_en_onnx(self) -> KokoroEnglishOnnxRuntime:
        with self._lock:
            if self._kokoro_en_onnx is None:
                self._kokoro_en_onnx = KokoroEnglishOnnxRuntime(self.cfg.get("kokoro_en", {}))
            self._kokoro_en_onnx.load()
            return self._kokoro_en_onnx

    def _activate_output_language(self, language: str, warmup: bool) -> None:
        with self._lock:
            if self._active_output_language == language:
                return
            if int(self.cfg.get("max_loaded_languages", 1)) <= 1:
                if language == "en":
                    self._release_vietnamese()
                else:
                    self._release_english()

            if language == "vi":
                self._ensure_kokoro_vi()
            else:
                self._prepare_english_backend()
            self._active_output_language = language

            if warmup:
                self._warmup()

    def _prepare_english_backend(self) -> None:
        backend = self.cfg.get("kokoro_en", {}).get("backend", "auto")
        if backend == "auto_benchmark":
            self._select_fastest_en_backend()
        elif backend in {"auto", "onnx"}:
            try:
                self._ensure_kokoro_en_onnx()
                self._active_en_backend = "onnx_int8"
            except Exception as exc:
                self._kokoro_en_onnx_failed = True
                if backend == "onnx":
                    raise
                print(f"[TTS EN] Kokoro ONNX unavailable; preparing PyTorch: {exc}")
                self._ensure_kokoro_en()
                self._active_en_backend = "pytorch"
        else:
            self._ensure_kokoro_en()
            self._active_en_backend = "pytorch"

    def _release_vietnamese(self) -> None:
        if self._kokoro_vi is not None:
            self._kokoro_vi = None
            self._release_device_memory()

    def _release_english(self) -> None:
        if self._kokoro_en is not None or self._kokoro_en_onnx is not None:
            self._kokoro_en = None
            self._kokoro_en_onnx = None
            self._active_en_backend = "unloaded"
            self._release_device_memory()

    @staticmethod
    def _release_device_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _select_fastest_en_backend(self) -> None:
        cfg = self.cfg.get("kokoro_en", {})
        cached_backend = self._load_cached_en_backend(cfg)
        if cached_backend == "onnx_int8":
            try:
                self._ensure_kokoro_en_onnx()
                self._active_en_backend = "onnx_int8"
                self._en_selection_source = "cache"
                print("[TTS EN] Using cached INT8 ONNX runtime selection.")
                return
            except Exception as exc:
                print(f"[TTS EN] Cached ONNX selection is invalid; recalibrating: {exc}")
        elif cached_backend == "pytorch":
            self._ensure_kokoro_en()
            self._active_en_backend = "pytorch"
            self._en_selection_source = "cache"
            print("[TTS EN] Using cached PyTorch runtime selection.")
            return

        try:
            runtime = self._ensure_kokoro_en_onnx()
            benchmark = runtime.synthesize(
                cfg.get("benchmark_text", "This is a realtime translation.")
            )
            audio_s = len(benchmark.audio) / max(benchmark.sample_rate, 1)
            rtf = (benchmark.elapsed_ms / 1000) / max(audio_s, 0.001)
            threshold = float(cfg.get("onnx_max_rtf", 0.9))
            print(
                f"[TTS EN] INT8 ONNX benchmark RTF={rtf:.2f} "
                f"(required <= {threshold:.2f})"
            )
            if rtf <= threshold:
                self._active_en_backend = "onnx_int8"
                self._en_selection_source = "benchmark"
                self._write_cached_en_backend(cfg, "onnx_int8", rtf)
                return
            print("[TTS EN] INT8 ONNX is slower on this CPU; selecting warmed PyTorch.")
        except Exception as exc:
            self._kokoro_en_onnx_failed = True
            print(f"[TTS EN] INT8 ONNX benchmark failed; selecting PyTorch: {exc}")

        self._kokoro_en_onnx = None
        self._kokoro_en_onnx_failed = True
        self._ensure_kokoro_en()
        self._active_en_backend = "pytorch"
        self._en_selection_source = "benchmark"
        self._write_cached_en_backend(cfg, "pytorch", rtf if "rtf" in locals() else None)

    def _load_cached_en_backend(self, cfg: dict) -> str | None:
        cache_path = self._selection_cache_path(cfg)
        try:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if record.get("signature") != self._selection_signature(cfg):
            return None
        backend = record.get("backend")
        return backend if backend in {"onnx_int8", "pytorch"} else None

    def _write_cached_en_backend(self, cfg: dict, backend: str, rtf: float | None) -> None:
        cache_path = self._selection_cache_path(cfg)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "backend": backend,
            "rtf": round(rtf, 4) if rtf is not None else None,
            "signature": self._selection_signature(cfg),
            "created_at": time.time(),
        }
        cache_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _selection_signature(self, cfg: dict) -> dict:
        model_path = Path(cfg.get("onnx_model_path", ""))
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[2] / model_path
        try:
            stat = model_path.stat()
            model_signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            model_signature = {"size": 0, "mtime_ns": 0}
        return {
            "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "onnx_model": model_signature,
            "onnx_threads": int(cfg.get("onnx_num_threads", 2)),
            "onnx_max_rtf": float(cfg.get("onnx_max_rtf", 0.9)),
        }

    @staticmethod
    def _selection_cache_path(cfg: dict) -> Path:
        path = Path(cfg.get("selection_cache", "models/kokoro-onnx/runtime-selection.json"))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path

    def _warmup(self) -> None:
        if self._kokoro_vi is not None and self._kokoro_vi.loaded:
            try:
                self._kokoro_vi.synthesize(self.cfg.get("kokoro_vi", {}).get("warmup_text", "xin chào"))
            except Exception as exc:
                print(f"[TTS VI] Warmup skipped: {exc}")
        if self._kokoro_en_onnx is not None and self._kokoro_en_onnx.loaded:
            try:
                self._kokoro_en_onnx.synthesize(
                    self.cfg.get("kokoro_en", {}).get("warmup_text", "hello")
                )
            except Exception as exc:
                print(f"[TTS EN] ONNX warmup skipped: {exc}")
        elif self._kokoro_en is not None and self._kokoro_en.loaded:
            try:
                self._kokoro_en.synthesize(self.cfg.get("kokoro_en", {}).get("warmup_text", "hello"))
            except Exception as exc:
                print(f"[TTS EN] Warmup skipped: {exc}")

    def _synthesize_en_pyttsx3(self, text: str) -> tuple[np.ndarray, int]:
        try:
            import pyttsx3
            import soundfile as sf

            start = time.perf_counter()
            engine = pyttsx3.init()
            rate = int(self.cfg.get("pyttsx3_rate", 165))
            engine.setProperty("rate", rate)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()
                audio, sr = sf.read(tmp_path, dtype="float32")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"[TTS EN] pyttsx3 fallback | {elapsed_ms:.0f}ms | {len(text)} chars")
            return self._nonempty(audio), sr
        except Exception as exc:
            print(f"[TTS EN] pyttsx3 fallback failed: {exc}")
            return self._silence()

    def _silence(self) -> tuple[np.ndarray, int]:
        return np.zeros(int(self.sample_rate * 0.35), dtype=np.float32), self.sample_rate

    def _nonempty(self, audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if len(arr) == 0:
            return np.zeros(int(self.sample_rate * 0.35), dtype=np.float32)
        return arr

    def _log_result(self, result: SynthesisResult, text: str) -> None:
        audio_s = len(result.audio) / max(result.sample_rate, 1)
        rtf = (result.elapsed_ms / 1000) / max(audio_s, 0.001)
        self.last_metrics = {
            "engine": result.engine,
            "inference_ms": round(result.elapsed_ms, 1),
            "audio_s": round(audio_s, 3),
            "rtf": round(rtf, 3),
            "chars": len(text),
        }
        print(
            f"[TTS] {result.engine} | {result.elapsed_ms:.0f}ms | "
            f"audio={audio_s:.2f}s | RTF={rtf:.2f} | chars={len(text)}"
        )
