"""Measure how far behind real time the live translation runs.

Feeds a WAV into :class:`LiveSessionStore` at wall-clock speed, exactly as the
microphone path does, and records the *emission lag*: at the moment new text
appears, how far behind the live audio position that text is.

Because audio is fed in real time from ``t0``, sample ``s`` arrives at
``t0 + s / sample_rate``. The text on screen covers audio up to
``state.processed_samples``, so::

    lag = now - (t0 + processed_samples / ASR_SAMPLE_RATE)

That is the number the user actually perceives as "the delay". Component
timings (ASR/MT compute, seconds of audio decoded) are collected alongside so a
regression can be attributed to a stage.

Usage::

    python scripts/bench_live_latency.py samples/arabic/bench_stream.wav
    python scripts/bench_live_latency.py --label after --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtt.audio import load_audio  # noqa: E402
from rtt.config import ASR_SAMPLE_RATE  # noqa: E402
from rtt.live import LiveSessionStore  # noqa: E402

FEED_CHUNK_SEC = 0.5


class _Instrumented:
    """Wraps a pipeline to accumulate per-stage compute without changing it."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.asr_calls = 0
        self.asr_seconds = 0.0
        self.asr_audio_seconds = 0.0
        self.mt_calls = 0
        self.mt_seconds = 0.0
        self.passes: list[dict] = []

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def transcribe(self, audio, sample_rate):
        started = time.perf_counter()
        result = self._inner.transcribe(audio, sample_rate)
        elapsed = time.perf_counter() - started
        chunk_sec = float(np.asarray(audio).size) / float(sample_rate)
        self.asr_seconds += elapsed
        self.asr_audio_seconds += chunk_sec
        self.asr_calls += 1
        self.passes.append(
            {
                "chunk_sec": round(chunk_sec, 2),
                "asr_sec": round(elapsed, 2),
                "chars": len(result[0] or ""),
            }
        )
        return result

    def translate(self, text):
        started = time.perf_counter()
        result = self._inner.translate(text)
        self.mt_seconds += time.perf_counter() - started
        self.mt_calls += 1
        return result


def _word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Word error rate of the live Arabic against the reference transcript.

    A speed win that quietly wrecks accuracy is not a win, so the benchmark has
    to price both. Plain Levenshtein over normalised words — no jiwer
    dependency for one metric.
    """
    from rtt.text import normalize_arabic

    ref = normalize_arabic(reference).split()
    hyp = normalize_arabic(hypothesis).split()
    if not ref:
        return None

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1] / len(ref)


def _track_coverage(store: LiveSessionStore, stats: dict) -> None:
    """Record how much captured audio the live path actually transcribed.

    ``_run_live_increment`` snapshots ``chunk`` and only later advances the
    cursor, so any audio that arrives while ASR is running can be marked
    processed without ever being decoded. This measures that gap.
    """
    from rtt.live import session as session_mod

    original = store._run_live_increment
    original_next = store._next_segment

    def wrapped(state, chunk_end=None, closed=False):
        before = state.processed_samples
        context = int(session_mod.LIVE_CONTEXT_SEC * ASR_SAMPLE_RATE)
        start = max(0, before - context)
        decoded_to = state.audio.size if chunk_end is None else chunk_end
        result = original(state, chunk_end=chunk_end, closed=closed)
        after = state.processed_samples
        if after > before:
            stats["skipped_samples"] += max(0, after - decoded_to)
            stats["decoded_span"] += max(0, min(after, decoded_to) - start)
        if closed:
            stats["closed_segments"] += 1
        else:
            stats["open_segments"] += 1
        return result

    def wrapped_next(state):
        decision = original_next(state)
        # Silence skipping advances the cursor without an ASR pass. That is
        # intentional, but if the detector misfires it is lost speech, so track
        # it separately rather than folding it into either bucket.
        if decision is not None and decision.skip_asr:
            stats["silence_skipped"] += max(0, decision.end - state.processed_samples)
        return decision

    store._run_live_increment = wrapped
    store._next_segment = wrapped_next


def _build_store() -> tuple[LiveSessionStore, _Instrumented]:
    """Build a store using the same live/final split the UI uses."""
    from rtt.ui.gradio_app import _final_config, _live_config
    from rtt.pipeline import build_pipeline

    live = _Instrumented(build_pipeline(_live_config(), include_tts=False))
    live.warmup()

    final_holder: dict[str, object] = {}

    def _get_final():
        if "p" not in final_holder:
            pipeline = build_pipeline(_final_config(), include_tts=False)
            pipeline.warmup()
            final_holder["p"] = pipeline
        return final_holder["p"]

    return LiveSessionStore(live, _get_final), live


def run(
    wav: Path, *, label: str, warmup_only: bool = False, do_finalize: bool = False
) -> dict:
    audio = load_audio(wav, ASR_SAMPLE_RATE)
    duration = float(audio.size) / ASR_SAMPLE_RATE
    print(f"[{label}] loaded {wav.name}: {duration:.2f}s")

    store, live = _build_store()
    if warmup_only:
        return {}

    coverage = {
        "skipped_samples": 0,
        "decoded_span": 0,
        "silence_skipped": 0,
        "closed_segments": 0,
        "open_segments": 0,
    }
    _track_coverage(store, coverage)

    state = store.create()
    store.start_processor(state.session_id)

    step = int(FEED_CHUNK_SEC * ASR_SAMPLE_RATE)
    lags: list[float] = []
    samples: list[dict] = []
    last_text = ""
    first_text_lag: float | None = None

    t0 = time.perf_counter()
    for start in range(0, audio.size, step):
        chunk = audio[start : start + step]
        # Feed at wall-clock speed, exactly as the browser streams the mic.
        target = t0 + (start + chunk.size) / ASR_SAMPLE_RATE
        while time.perf_counter() < target:
            now = time.perf_counter()
            if state.english_text != last_text:
                last_text = state.english_text
                committed = t0 + state.processed_samples / ASR_SAMPLE_RATE
                lag = now - committed
                lags.append(lag)
                if first_text_lag is None:
                    first_text_lag = now - t0
                samples.append({"wall": round(now - t0, 2), "lag": round(lag, 2)})
            time.sleep(0.02)
        store.append_chunk(state.session_id, ASR_SAMPLE_RATE, chunk)

    # Drain: let the backlog land before scoring, so word error rate measures
    # transcription quality rather than how far behind the config finished.
    # Lag is reported separately and already captures that.
    drain_started = time.perf_counter()
    while time.perf_counter() - drain_started < 40.0:
        if state.english_text != last_text:
            now = time.perf_counter()
            last_text = state.english_text
            committed = t0 + state.processed_samples / ASR_SAMPLE_RATE
            lags.append(now - committed)
            samples.append({"wall": round(now - t0, 2), "lag": round(now - committed, 2)})
        if not state.processing and state.new_audio_sec() < 0.1:
            break
        time.sleep(0.05)

    live_english = state.english_text
    live_arabic = state.arabic_text
    live_asr_calls, live_asr_s = live.asr_calls, live.asr_seconds
    live_audio_s, live_mt_s = live.asr_audio_seconds, live.mt_seconds
    live_mt_calls = live.mt_calls

    reference_path = wav.with_suffix(".txt")
    reference = (
        reference_path.read_text(encoding="utf-8") if reference_path.exists() else ""
    )
    wer = _word_error_rate(reference, live_arabic) if reference else None

    final_sec = None
    if do_finalize:
        final_started = time.perf_counter()
        store.finalize(state.session_id)
        final_sec = round(time.perf_counter() - final_started, 2)
    else:
        # The final pass pulls medium Whisper + NLLB; tiers 1-3 target the live
        # path, so it stays opt-in behind --finalize.
        store.remove(state.session_id)

    result = {
        "label": label,
        "audio_sec": round(duration, 2),
        "lag_median": round(statistics.median(lags), 2) if lags else None,
        "lag_p90": round(sorted(lags)[int(len(lags) * 0.9)], 2) if lags else None,
        "lag_max": round(max(lags), 2) if lags else None,
        "first_text_sec": round(first_text_lag, 2) if first_text_lag else None,
        "updates": len(lags),
        "asr_passes": live_asr_calls,
        "asr_compute_sec": round(live_asr_s, 2),
        "asr_audio_decoded_sec": round(live_audio_s, 2),
        "asr_redundancy": (
            round(live_audio_s / duration, 2) if duration else None
        ),
        "wer": (round(wer, 3) if wer is not None else None),
        "live_words": len(live_arabic.split()),
        "reference_words": len(reference.split()) if reference else 0,
        "audio_skipped_sec": round(coverage["skipped_samples"] / ASR_SAMPLE_RATE, 2),
        "audio_skipped_pct": (
            round(100.0 * coverage["skipped_samples"] / ASR_SAMPLE_RATE / duration, 1)
            if duration
            else None
        ),
        "silence_skipped_sec": round(coverage["silence_skipped"] / ASR_SAMPLE_RATE, 2),
        "closed_segments": coverage["closed_segments"],
        "open_segments": coverage["open_segments"],
        "mt_calls": live_mt_calls,
        "mt_compute_sec": round(live_mt_s, 2),
        "finalize_sec": final_sec,
        "live_arabic": live_arabic[:200],
        "live_english": live_english[:200],
        "passes": live.passes,
        "trace": samples,
    }
    return result


def _report(r: dict) -> None:
    print(f"\n===== [{r['label']}] {r['audio_sec']}s of Arabic =====")
    print(f"  emission lag   median {r['lag_median']}s | p90 {r['lag_p90']}s | max {r['lag_max']}s")
    print(f"  first text at  {r['first_text_sec']}s   ({r['updates']} updates)")
    print(
        f"  ASR  {r['asr_passes']} passes, {r['asr_compute_sec']}s compute, "
        f"decoded {r['asr_audio_decoded_sec']}s of audio ({r['asr_redundancy']}x realtime audio)"
    )
    print(
        f"  SKIPPED audio never transcribed: {r['audio_skipped_sec']}s "
        f"({r['audio_skipped_pct']}% of the stream)"
    )
    print(
        f"  silence skipped without ASR: {r['silence_skipped_sec']}s | "
        f"segments: {r['closed_segments']} closed, {r['open_segments']} open"
    )
    print(f"  MT   {r['mt_calls']} calls, {r['mt_compute_sec']}s compute")
    if r.get("wer") is not None:
        print(
            f"  Arabic WER {r['wer']:.3f}  "
            f"({r['live_words']} words transcribed vs {r['reference_words']} reference)"
        )
    print("  per pass: " + ", ".join(f"{p['chunk_sec']}s/{p['asr_sec']}s" for p in r.get("passes", [])))
    if r.get("finalize_sec") is not None:
        print(f"  finalize (full buffer, medium): {r['finalize_sec']}s")
    print(f"  EN: {r['live_english'][:160]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="?", default="samples/arabic/bench_stream.wav")
    parser.add_argument("--label", default="run")
    parser.add_argument("--json", dest="json_out", default="")
    parser.add_argument("--warmup-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    wav = Path(args.wav)
    if not wav.exists():
        print(f"missing audio: {wav}", file=sys.stderr)
        return 1

    result = run(
        wav,
        label=args.label,
        warmup_only=args.warmup_only,
        do_finalize=args.finalize,
    )
    if args.warmup_only:
        print("warmup complete")
        return 0
    _report(result)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
