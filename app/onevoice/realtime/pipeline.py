"""Realtime ASR -> MT pipeline orchestration."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path
from typing import Any

from onevoice.adapters.asr import ASRManager
from onevoice.adapters.audio import AudioCapture, Denoiser
from onevoice.adapters.subtitles import SRTGenerator
from onevoice.adapters.text import normalize
from onevoice.adapters.translation import Translator
from onevoice.config import load_config


class OneVoicePipeline:
    """Staged realtime pipeline for VI<->EN speech translation."""

    def __init__(self, config_path: str | Path = "config/config.yaml", direction: str = "vi2en"):
        self.cfg = load_config(config_path)
        self.direction = direction
        q_size = self.cfg["pipeline"]["queue_maxsize"]

        self.q_audio_raw: queue.Queue = queue.Queue(maxsize=q_size)
        self.q_audio_clean: queue.Queue = queue.Queue(maxsize=q_size)
        self.q_text_src: queue.Queue = queue.Queue(maxsize=q_size)
        self.q_results: queue.Queue = queue.Queue(maxsize=q_size)

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
                    meta = {key: value for key, value in raw_item.items() if key != "audio"}
                else:
                    raw = raw_item
                    meta = {}

                start = time.perf_counter()
                clean = self.denoiser.denoise(raw)
                meta["denoise_ms"] = (time.perf_counter() - start) * 1000
                meta["audio"] = clean
                self._put_drop_oldest(self.q_audio_clean, meta)
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

                start = time.perf_counter()
                result = self.asr.transcribe(audio, direction=direction)
                asr_ms = (time.perf_counter() - start) * 1000

                if result.get("text"):
                    result["text"] = normalize(result["text"], lang=result["lang"])
                    result["denoise_ms"] = denoise_ms
                    result["asr_ms"] = asr_ms
                    result["audio_duration_s"] = len(audio) / self.cfg["audio"]["sample_rate"]
                    if isinstance(audio_item, dict):
                        result["request_id"] = audio_item.get("request_id")
                        result["created_at"] = audio_item.get("created_at")
                    self._put_drop_oldest(self.q_text_src, result)
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
                start = time.perf_counter()
                translated = self.translator.translate(item["text"], direction=item["direction"])
                item["mt_ms"] = (time.perf_counter() - start) * 1000

                if translated:
                    item["translated"] = translated
                    total_ms = item.get("denoise_ms", 0) + item.get("asr_ms", 0) + item.get("mt_ms", 0)
                    self._log_latency(item, total_ms)
                    self.srt.add_entry(item["text"], item["translated"], item.get("audio_duration_s", 1.0))
                    self._emit_result(item, total_ms)

                self.q_text_src.task_done()
            except queue.Empty:
                continue

    def _emit_result(self, item: dict[str, Any], total_ms: float):
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
        self._put_drop_oldest(self.q_results, result)

    def _emit_error(self, request_id: str, message: str, latency_ms: dict[str, float] | None = None):
        self._put_drop_oldest(
            self.q_results,
            {
                "request_id": request_id,
                "error": message,
                "source_text": "",
                "translated_text": "",
                "direction": self.direction,
                "emotion": "neutral",
                "event": "none",
                "audio_duration_s": 0,
                "latency_ms": latency_ms or {},
            },
        )

    def _log_latency(self, item: dict[str, Any], total_ms: float):
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
        start = time.perf_counter()
        self.denoiser.load()
        self.asr.load()
        self.translator.load()
        self._models_loaded = True
        print(f"\n[Pipeline] All models loaded in {time.perf_counter() - start:.1f}s\n")

    def start_workers(self, use_capture: bool = True):
        """Start queue workers. Web mode sets use_capture=False."""
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

    def submit_audio(self, audio, direction: str | None = None, request_id: str | None = None):
        """Push one speech segment into the realtime queue path."""
        self._put_drop_oldest(
            self.q_audio_raw,
            {
                "audio": audio,
                "direction": direction or self.direction,
                "request_id": request_id,
                "created_at": time.perf_counter(),
            },
        )

    def get_result(self, request_id: str | None = None, timeout: float = 15.0):
        """Wait for a translated result emitted by the MT worker."""
        deadline = time.perf_counter() + timeout
        parked = []
        try:
            while time.perf_counter() < deadline:
                remaining = max(0.05, deadline - time.perf_counter())
                try:
                    item = self.q_results.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    continue
                if request_id is None or item.get("request_id") == request_id:
                    return item
                parked.append(item)
        finally:
            for item in parked:
                self._put_drop_oldest(self.q_results, item)
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
                        f"[Pipeline] Queue sizes - raw:{self.q_audio_raw.qsize()} "
                        f"clean:{self.q_audio_clean.qsize()} src_text:{self.q_text_src.qsize()} "
                        f"results:{self.q_results.qsize()} | Avg latency: {avg:.0f}ms"
                    )
        except KeyboardInterrupt:
            self.capture.stop()
            if self.srt.entry_count > 0:
                srt_path = f"output_{int(time.time())}.srt"
                self.srt.save(srt_path)
                print(f"\nSRT subtitle saved: {srt_path}")
            print("\nOneVoice Edge stopped.")

    @staticmethod
    def _put_drop_oldest(target_queue: queue.Queue, item):
        if target_queue.full():
            try:
                target_queue.get_nowait()
                target_queue.task_done()
            except queue.Empty:
                pass
        target_queue.put(item)


def main():
    parser = argparse.ArgumentParser(description="OneVoice Edge real-time VI<->EN speech translation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--direction", choices=["vi2en", "en2vi"], default="vi2en")
    args = parser.parse_args()
    OneVoicePipeline(config_path=args.config, direction=args.direction).start()


if __name__ == "__main__":
    main()
