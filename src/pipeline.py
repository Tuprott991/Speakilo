"""
OneVoice Edge main pipeline orchestrator.

The pipeline is intentionally staged:
  audio raw queue -> denoise worker -> clean audio queue -> ASR worker
  -> source text queue -> MT worker -> translated result queue

TTS is disabled in the default live pipeline. The web UI can trigger TTS
explicitly after a translated result is available.
"""

import argparse
import os
import queue
import sys
import threading
import time

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from asr.asr_manager import ASRManager
from audio.capture import AudioCapture
from audio.denoise import Denoiser
from translation.mt_engine import Translator
from utils.srt_generator import SRTGenerator
from utils.text_normalizer import normalize


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class OneVoicePipeline:
    """End-to-end real-time VI<->EN speech translation pipeline."""

    def __init__(self, config_path: str = "config/config.yaml", direction: str = "vi2en"):
        self.cfg = load_config(config_path)
        self.direction = direction
        q_size = self.cfg["pipeline"]["queue_maxsize"]

        self.q_audio_raw = queue.Queue(maxsize=q_size)
        self.q_audio_clean = queue.Queue(maxsize=q_size)
        self.q_text_src = queue.Queue(maxsize=q_size)
        self.q_results = queue.Queue(maxsize=q_size)

        self.capture = AudioCapture(self.q_audio_raw, self.cfg)
        self.denoiser = Denoiser(num_threads=2)
        self.asr = ASRManager(self.cfg)
        self.translator = Translator(self.cfg)
        self.srt = SRTGenerator(bilingual=True)

        self._latency_log: list[float] = []
        self._models_loaded = False
        self._workers_started = False

    def _denoise_worker(self):
        print("[Denoise Worker] Started")
        while True:
            try:
                raw_item = self.q_audio_raw.get(timeout=1)
                if isinstance(raw_item, dict):
                    raw = raw_item["audio"]
                    meta = {k: v for k, v in raw_item.items() if k != "audio"}
                else:
                    raw = raw_item
                    meta = {}

                t0 = time.perf_counter()
                clean = self.denoiser.denoise(raw)
                meta["denoise_ms"] = (time.perf_counter() - t0) * 1000
                meta["audio"] = clean
                if not self.q_audio_clean.full():
                    self.q_audio_clean.put(meta)
                self.q_audio_raw.task_done()
            except queue.Empty:
                continue

    def _asr_worker(self):
        print(f"[ASR Worker] Started (default direction={self.direction})")
        while True:
            try:
                audio_item = self.q_audio_clean.get(timeout=1)
                if isinstance(audio_item, dict):
                    audio = audio_item["audio"]
                    denoise_ms = audio_item.get("denoise_ms", 0)
                    direction = audio_item.get("direction", self.direction)
                else:
                    audio, denoise_ms = audio_item
                    direction = self.direction

                t0 = time.perf_counter()
                result = self.asr.transcribe(audio, direction=direction)
                asr_ms = (time.perf_counter() - t0) * 1000

                if result.get("text"):
                    result["text"] = normalize(result["text"], lang=result["lang"])
                    result["denoise_ms"] = denoise_ms
                    result["asr_ms"] = asr_ms
                    result["audio_duration_s"] = len(audio) / self.cfg["audio"]["sample_rate"]
                    if isinstance(audio_item, dict):
                        result["request_id"] = audio_item.get("request_id")
                        result["created_at"] = audio_item.get("created_at")
                    self.q_text_src.put(result)
                elif isinstance(audio_item, dict) and audio_item.get("request_id"):
                    self._emit_error(
                        audio_item.get("request_id"),
                        "No speech recognized",
                        {
                            "denoise": round(denoise_ms, 1),
                            "asr": round(asr_ms, 1),
                            "mt": 0,
                            "total": round(denoise_ms + asr_ms, 1),
                        },
                    )

                self.q_audio_clean.task_done()
            except queue.Empty:
                continue

    def _mt_worker(self):
        print("[MT Worker] Started (VI<->EN)")
        while True:
            try:
                item = self.q_text_src.get(timeout=1)
                t0 = time.perf_counter()
                translated = self.translator.translate(item["text"], direction=item["direction"])
                item["mt_ms"] = (time.perf_counter() - t0) * 1000

                if translated:
                    item["translated"] = translated
                    total_ms = (
                        item.get("denoise_ms", 0)
                        + item.get("asr_ms", 0)
                        + item.get("mt_ms", 0)
                    )
                    self._log_latency(item, total_ms)
                    self.srt.add_entry(
                        item["text"],
                        item["translated"],
                        item.get("audio_duration_s", 1.0),
                    )
                    self._emit_result(item, total_ms)

                self.q_text_src.task_done()
            except queue.Empty:
                continue

    def _emit_result(self, item: dict, total_ms: float):
        result = {
            "request_id": item.get("request_id"),
            "source_text": item["text"],
            "translated_text": item["translated"],
            "direction": item.get("direction", self.direction),
            "emotion": item.get("emotion", "neutral"),
            "event": item.get("event", "speech"),
            "audio_duration_s": round(item.get("audio_duration_s", 0), 3),
            "latency_ms": {
                "denoise": round(item.get("denoise_ms", 0), 1),
                "asr": round(item.get("asr_ms", 0), 1),
                "mt": round(item.get("mt_ms", 0), 1),
                "total": round(total_ms, 1),
            },
        }
        if not self.q_results.full():
            self.q_results.put(result)

    def _emit_error(self, request_id: str, message: str, latency_ms: dict = None):
        if not self.q_results.full():
            self.q_results.put({
                "request_id": request_id,
                "error": message,
                "source_text": "",
                "translated_text": "",
                "direction": self.direction,
                "emotion": "neutral",
                "event": "none",
                "audio_duration_s": 0,
                "latency_ms": latency_ms or {},
            })

    def _log_latency(self, item: dict, total_ms: float):
        arrow = "VI->EN" if item.get("direction") == "vi2en" else "EN->VI"
        status = "OK" if total_ms < 1000 else "SLOW"
        print(
            f"\n{status} [{arrow}] Total: {total_ms:.0f}ms | "
            f"Denoise: {item.get('denoise_ms', 0):.0f}ms | "
            f"ASR: {item.get('asr_ms', 0):.0f}ms | "
            f"MT: {item.get('mt_ms', 0):.0f}ms | TTS: disabled"
        )
        print(f"   \"{item['text']}\" -> \"{item['translated']}\"\n")
        self._latency_log.append(total_ms)

    def load_models(self):
        if self._models_loaded:
            return
        print("\n[Pipeline] Loading models (VI<->EN pipeline)...")
        t0 = time.perf_counter()
        self.denoiser.load()
        self.asr.load()
        self.translator.load()
        self._models_loaded = True
        print(f"\n[Pipeline] All models loaded in {time.perf_counter() - t0:.1f}s\n")

    def start_workers(self, use_capture: bool = True):
        """Start the queue workers. Web mode sets use_capture=False."""
        self.load_models()
        if self._workers_started:
            return

        workers = [
            threading.Thread(target=self._denoise_worker, daemon=True, name="Denoise"),
            threading.Thread(target=self._asr_worker, daemon=True, name="ASR"),
            threading.Thread(target=self._mt_worker, daemon=True, name="MT"),
        ]
        if use_capture:
            workers.insert(0, threading.Thread(target=self.capture.start, daemon=True, name="AudioCapture"))

        for worker in workers:
            worker.start()
        self._workers_started = True

    def submit_audio(self, audio, direction: str = None, request_id: str = None):
        """Push a speech segment into the streaming queue path."""
        self.q_audio_raw.put({
            "audio": audio,
            "direction": direction or self.direction,
            "request_id": request_id,
            "created_at": time.perf_counter(),
        })

    def get_result(self, request_id: str = None, timeout: float = 15.0):
        """Wait for a translated result emitted by the MT worker."""
        deadline = time.perf_counter() + timeout
        parked = []
        try:
            while time.perf_counter() < deadline:
                remaining = max(0.05, deadline - time.perf_counter())
                item = self.q_results.get(timeout=min(0.25, remaining))
                if request_id is None or item.get("request_id") == request_id:
                    return item
                parked.append(item)
        except queue.Empty:
            pass
        finally:
            for item in parked:
                if not self.q_results.full():
                    self.q_results.put(item)
        return None

    def start(self):
        direction_label = "VI -> EN" if self.direction == "vi2en" else "EN -> VI"
        print(f"OneVoice Edge is LIVE - {direction_label}")
        print("Speak into the microphone. Press Ctrl+C to stop.\n")
        self.start_workers(use_capture=True)

        try:
            while True:
                time.sleep(5)
                if self._latency_log:
                    avg = sum(self._latency_log) / len(self._latency_log)
                    print(
                        f"[Pipeline] Queue sizes - "
                        f"raw:{self.q_audio_raw.qsize()} "
                        f"clean:{self.q_audio_clean.qsize()} "
                        f"src_text:{self.q_text_src.qsize()} "
                        f"results:{self.q_results.qsize()} | "
                        f"Avg latency: {avg:.0f}ms"
                    )
        except KeyboardInterrupt:
            self.capture.stop()
            if self.srt.entry_count > 0:
                srt_path = f"output_{int(time.time())}.srt"
                self.srt.save(srt_path)
                print(f"\nSRT subtitle saved: {srt_path}")
            print("\nOneVoice Edge stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneVoice Edge real-time VI<->EN speech translation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--direction",
        choices=["vi2en", "en2vi"],
        default="vi2en",
        help="Translation direction: vi2en or en2vi",
    )
    args = parser.parse_args()

    pipeline = OneVoicePipeline(config_path=args.config, direction=args.direction)
    pipeline.start()
