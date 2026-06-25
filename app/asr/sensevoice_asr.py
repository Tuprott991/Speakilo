"""Quantized SenseVoice ONNX runtime for English input."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np


_SenseVoiceSmall = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SenseVoiceASR:
    """Wrapper around the official SenseVoice ONNX export."""

    def __init__(self, config: dict):
        global _SenseVoiceSmall
        cfg = config.get("sensevoice", {})
        self.model_dir = cfg.get("model_path", "models/sensevoice-small-onnx")
        self.model_id = cfg.get("model_id", "iic/SenseVoiceSmall-onnx")
        self.quantize = bool(cfg.get("quantize", True))
        self.num_threads = int(cfg.get("num_threads", 4))
        self.model = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

        if _SenseVoiceSmall is None:
            try:
                from funasr_onnx import SenseVoiceSmall

                _SenseVoiceSmall = SenseVoiceSmall
            except ImportError:
                self._load_error = "funasr_onnx not installed. Run: pip install funasr_onnx modelscope"
                print(f"[ASR] SenseVoice unavailable: {self._load_error}")
                return

        print(
            f"[ASR] SenseVoice ONNX source: {self.model_dir} "
            f"(fallback={self.model_id}, quantized={self.quantize})"
        )

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> bool:
        if self.model is not None:
            return True
        if _SenseVoiceSmall is None:
            return False

        with self._lock:
            if self.model is not None:
                return True
            try:
                model_dir = self._resolve_model_dir()
                self._validate_model_dir(model_dir)
                self.model = _SenseVoiceSmall(
                    model_dir,
                    batch_size=1,
                    quantize=self.quantize,
                    intra_op_num_threads=self.num_threads,
                )
                self._load_error = None
                print(f"[ASR] SenseVoice ONNX ready from {model_dir}")
                return True
            except Exception as exc:
                self._load_error = str(exc)
                print(f"[ASR] SenseVoice load failed: {exc}")
                return False

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> dict:
        del sample_rate
        if len(audio) == 0:
            return self._empty()
        if self.model is None and not self.load():
            raise RuntimeError(self._load_error or "SenseVoice ONNX is unavailable")

        try:
            audio_f32 = audio.astype(np.float32)
            peak = float(np.abs(audio_f32).max()) if len(audio_f32) else 0.0
            if peak > 1.0:
                audio_f32 /= 32768.0

            result = self.model(audio_f32, language="en", textnorm="withitn")
            if not result:
                return self._empty()

            parsed = self._parse_output(str(result[0]))
            if parsed["emotion"] != "neutral" or parsed["event"] not in {"speech", "none"}:
                print(f"[SenseVoice] emotion={parsed['emotion']} event={parsed['event']}")
            return parsed
        except Exception as exc:
            print(f"[ASR] SenseVoice transcription failed: {exc}")
            raise RuntimeError(f"SenseVoice transcription failed: {exc}") from exc

    def _resolve_model_dir(self) -> str:
        local_dir = Path(self.model_dir)
        if not local_dir.is_absolute():
            local_dir = PROJECT_ROOT / local_dir
        required_model = "model_quant.onnx" if self.quantize else "model.onnx"
        model_path = local_dir / required_model
        if model_path.is_file() and model_path.stat().st_size > 10 * 1024 * 1024:
            return str(local_dir)

        from modelscope import snapshot_download

        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ASR] Downloading {self.model_id} to {local_dir}...")
        return str(snapshot_download(self.model_id, local_dir=str(local_dir)))

    def _validate_model_dir(self, model_dir: str) -> None:
        root = Path(model_dir)
        model_name = "model_quant.onnx" if self.quantize else "model.onnx"
        required = [
            model_name,
            "config.yaml",
            "am.mvn",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
        ]
        missing = [name for name in required if not (root / name).is_file()]
        model_path = root / model_name
        if model_path.is_file() and model_path.stat().st_size <= 10 * 1024 * 1024:
            missing.append(f"{model_name} (incomplete)")
        if missing:
            raise FileNotFoundError(
                f"SenseVoice ONNX export at {root} is incomplete; missing {', '.join(missing)}. "
                f"Download ModelScope model {self.model_id}, not the PyTorch-only iic/SenseVoiceSmall."
            )

    def _parse_output(self, raw_text: str) -> dict:
        emotion_match = re.search(
            r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>",
            raw_text,
            re.IGNORECASE,
        )
        event_match = re.search(
            r"<\|(BGM|Speech|Applause|Laughter|Cry|Sneeze|Breath|Cough)\|>",
            raw_text,
            re.IGNORECASE,
        )
        clean_text = re.sub(r"<\|.*?\|>", "", raw_text).strip()
        return {
            "text": clean_text,
            "emotion": emotion_match.group(1).lower() if emotion_match else "neutral",
            "event": event_match.group(1).lower() if event_match else "speech",
            "raw": raw_text,
        }

    @staticmethod
    def _empty() -> dict:
        return {"text": "", "emotion": "neutral", "event": "speech"}
