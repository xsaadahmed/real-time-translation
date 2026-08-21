"""Isolate how ASR and MT throughput scale with thread count on this host.

The end-to-end benchmark conflates scheduling with compute. This measures the
two runtimes directly so a thread setting can be chosen on evidence rather than
on the assumption that more cores is faster.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rtt.audio import load_audio  # noqa: E402
from rtt.config import ASR_SAMPLE_RATE  # noqa: E402

WAV = "samples/arabic/bench_stream.wav"
ARABIC = "أعتقد أن الحفاظ على اللغة العربية مسؤولية مشتركة بين الجميع."
REPEATS = 3


def bench_asr(audio: np.ndarray, threads: int, temps: tuple[float, ...]) -> float:
    from rtt.asr.faster_whisper_asr import FasterWhisperASR
    from rtt.config import ASRConfig

    cfg = ASRConfig()
    cfg.model_size = "base"
    cfg.beam_size = 1
    cfg.condition_on_previous_text = False
    cfg.initial_prompt = ""
    cfg.cpu_threads = threads
    cfg.num_workers = 1
    cfg.temperatures = temps

    engine = FasterWhisperASR(cfg)
    engine.load()
    engine.transcribe(audio, ASR_SAMPLE_RATE)  # warm

    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        engine.transcribe(audio, ASR_SAMPLE_RATE)
        best = min(best, time.perf_counter() - started)
    return best


def bench_mt(threads: int) -> float:
    import torch

    from rtt.config import MTConfig
    from rtt.mt.huggingface import HuggingFaceTranslator

    cfg = MTConfig()
    cfg.backend = "marian"
    cfg.model_name = ""
    cfg.num_beams = 2
    cfg.max_new_tokens = 128
    cfg.torch_threads = threads

    translator = HuggingFaceTranslator(cfg)
    translator.load()
    if threads > 0:
        torch.set_num_threads(threads)
    translator.translate_batch([ARABIC])  # warm

    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        translator.translate_batch([ARABIC])
        best = min(best, time.perf_counter() - started)
    return best


def main() -> int:
    import torch

    print(f"logical cores: {os.cpu_count()}  torch default threads: {torch.get_num_threads()}")

    audio = load_audio(Path(WAV), ASR_SAMPLE_RATE)
    chunk = audio[: int(4.0 * ASR_SAMPLE_RATE)]  # a typical live chunk
    print(f"ASR chunk: {chunk.size / ASR_SAMPLE_RATE:.1f}s\n")

    print("ASR cpu_threads (temperature=[0.0]):")
    for threads in (4, 8, 12, 16):
        secs = bench_asr(chunk, threads, (0.0,))
        print(f"  {threads:2d} threads -> {secs:.3f}s  (RTF {secs / 4.0:.3f})")

    print("\nASR temperature fallback at 8 threads:")
    for label, temps in (("[0.0]", (0.0,)), ("[0.0,0.2,0.4]", (0.0, 0.2, 0.4))):
        secs = bench_asr(chunk, 8, temps)
        print(f"  {label:14s} -> {secs:.3f}s")

    print("\nMT torch_threads (Marian, 1 sentence, beams=2):")
    for threads in (0, 4, 8, 12):
        secs = bench_mt(threads)
        label = "default" if threads == 0 else str(threads)
        print(f"  {label:>7s} -> {secs:.3f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
