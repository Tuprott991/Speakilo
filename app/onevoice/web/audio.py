"""Audio decoding helpers for HTTP uploads."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import UploadFile


async def decode_upload(file: UploadFile, sample_rate: int) -> np.ndarray:
    """Decode an uploaded audio file into mono float32 PCM at sample_rate."""
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        input_path = tmp.name

    try:
        try:
            import librosa

            audio, _ = librosa.load(input_path, sr=sample_rate, mono=True)
            return audio.astype(np.float32)
        except Exception:
            from pydub import AudioSegment

            segment = AudioSegment.from_file(input_path)
            segment = segment.set_channels(1).set_frame_rate(sample_rate)
            samples = np.array(segment.get_array_of_samples()).astype(np.float32)
            samples /= max(float(1 << (8 * segment.sample_width - 1)), 1.0)
            return samples
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass
