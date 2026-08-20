"""Live microphone session with incremental ASR and dual-quality pipelines.

Live updates transcribe only NEW audio (not the entire growing buffer) using a
fast ASR model. The final pass re-transcribes everything with a larger model
for accuracy.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..audio import resample, save_wav, to_mono_float32
from ..config import ASR_SAMPLE_RATE, SecondOpinionConfig
from ..pipeline import TranslationPipeline
from ..second_opinion import SecondOpinionEngine, compare, log_record
from ..text import merge_incremental_text, reconcile_provisional

logger = logging.getLogger(__name__)

MIN_AUDIO_SEC = 0.8
MIN_NEW_AUDIO_SEC = 2.0
MIN_PROCESS_INTERVAL_SEC = 1.0
LIVE_CONTEXT_SEC = 3.0


@dataclass
class LiveStreamState:
    session_id: str = ""
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sample_rate: int = ASR_SAMPLE_RATE
    # Verified spans are immutable once written - only ever appended to.
    # Provisional spans cover the still-being-re-transcribed tail and are
    # wholesale replaced every live pass. See text.reconcile_provisional.
    arabic_verified: str = ""
    arabic_provisional: str = ""
    english_verified: str = ""
    english_provisional: str = ""
    status_message: str = "Click the microphone and speak Arabic."
    processed_samples: int = 0
    processing: bool = False
    is_active: bool = False
    last_process_wall: float = 0.0
    chunks_received: int = 0

    @property
    def arabic_text(self) -> str:
        return " ".join(p for p in (self.arabic_verified, self.arabic_provisional) if p)

    @property
    def english_text(self) -> str:
        return " ".join(p for p in (self.english_verified, self.english_provisional) if p)

    def duration_sec(self) -> float:
        if self.audio.size == 0 or self.sample_rate <= 0:
            return 0.0
        return float(self.audio.size) / float(self.sample_rate)

    def new_audio_sec(self) -> float:
        new_samples = self.audio.size - self.processed_samples
        return float(new_samples) / float(self.sample_rate)


def merge_audio(
    existing: np.ndarray,
    new_samples: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    new_mono = resample(to_mono_float32(new_samples), sample_rate, ASR_SAMPLE_RATE)
    if new_mono.size == 0:
        return existing
    if existing.size == 0:
        return new_mono
    if new_mono.size > existing.size:
        return new_mono
    if new_mono.size == existing.size:
        probe = min(800, existing.size)
        if probe > 0 and np.allclose(existing[:probe], new_mono[:probe], atol=0.02):
            return existing
        return np.concatenate([existing, new_mono])
    return np.concatenate([existing, new_mono])


class LiveSessionStore:
    def __init__(
        self,
        live_pipeline: TranslationPipeline,
        final_pipeline_getter: Callable[[], TranslationPipeline],
        second_opinion_config: SecondOpinionConfig | None = None,
    ) -> None:
        self.live_pipeline = live_pipeline
        self._get_final_pipeline = final_pipeline_getter
        self._sessions: dict[str, LiveStreamState] = {}
        self._lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._processors_running: set[str] = set()
        # Off by default - self-contained and independent of the cascade,
        # so it never blocks the main path (see second_opinion/README notes).
        self._second_opinion_config = second_opinion_config or SecondOpinionConfig()
        self._second_opinion_engine: SecondOpinionEngine | None = None
        self._second_opinion_lock = threading.Lock()

    def create(self) -> LiveStreamState:
        state = LiveStreamState(
            session_id=str(uuid.uuid4()),
            status_message="Listening… speak Arabic.",
            is_active=True,
        )
        with self._lock:
            self._sessions[state.session_id] = state
        return state

    def get(self, session_id: str | None) -> LiveStreamState | None:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
        self._processors_running.discard(session_id)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for state in self._sessions.values() if state.is_active)

    def append_chunk(
        self,
        session_id: str,
        sample_rate: int,
        samples: np.ndarray,
    ) -> LiveStreamState | None:
        state = self.get(session_id)
        if state is None:
            return None

        before = state.duration_sec()
        state.audio = merge_audio(state.audio, samples, sample_rate)
        state.sample_rate = ASR_SAMPLE_RATE
        state.chunks_received += 1
        after = state.duration_sec()

        logger.info(
            "Chunk #%d: %.2fs -> %.2fs total",
            state.chunks_received,
            before,
            after,
        )
        return state

    def start_processor(self, session_id: str) -> None:
        if session_id in self._processors_running:
            return
        self._processors_running.add(session_id)
        threading.Thread(
            target=self._processor_loop,
            args=(session_id,),
            name=f"live-asr-{session_id[:8]}",
            daemon=True,
        ).start()

    def _processor_loop(self, session_id: str) -> None:
        logger.info("Live processor started for %s", session_id[:8])
        while True:
            state = self.get(session_id)
            if state is None or not state.is_active:
                break

            now = time.time()
            should_run = (
                state.new_audio_sec() >= MIN_NEW_AUDIO_SEC
                and not state.processing
                and (now - state.last_process_wall) >= MIN_PROCESS_INTERVAL_SEC
            )

            if should_run:
                state.processing = True
                try:
                    self._run_live_increment(state)
                    state.last_process_wall = time.time()
                except Exception:
                    logger.exception("Live transcription failed")
                    state.status_message = "Transcription error — retrying…"
                finally:
                    state.processing = False

            time.sleep(0.2)

        self._processors_running.discard(session_id)
        logger.info("Live processor stopped for %s", session_id[:8])

    def _run_live_increment(self, state: LiveStreamState) -> LiveStreamState:
        """Transcribe only the new audio since the last pass (fast path)."""
        context = int(LIVE_CONTEXT_SEC * ASR_SAMPLE_RATE)
        start = max(0, state.processed_samples - context)
        chunk = state.audio[start:]
        chunk_sec = float(chunk.size) / float(ASR_SAMPLE_RATE)

        if chunk_sec < MIN_AUDIO_SEC:
            state.status_message = f"Listening… ({state.duration_sec():.1f}s captured)"
            return state

        state.status_message = f"Transcribing… ({state.duration_sec():.1f}s captured)"

        with self._process_lock:
            arabic_new, _, asr_times = self.live_pipeline.transcribe(chunk, state.sample_rate)
            newly_verified_ar, provisional_ar = reconcile_provisional(
                state.arabic_provisional, arabic_new
            )
            if newly_verified_ar:
                state.arabic_verified = merge_incremental_text(
                    state.arabic_verified, newly_verified_ar
                )
            state.arabic_provisional = provisional_ar

            # Advance cursor — we consumed audio up to the current end.
            state.processed_samples = state.audio.size

            mt_times: dict[str, float] = {}
            if newly_verified_ar:
                # Verified Arabic never changes again, so its translation is
                # final too — translate once and append permanently.
                verified_en, verified_mt_times = self.live_pipeline.translate(newly_verified_ar)
                if verified_en:
                    state.english_verified = merge_incremental_text(
                        state.english_verified, verified_en
                    )
                mt_times.update(verified_mt_times)
            if state.arabic_provisional:
                # Provisional Arabic can still change, so its translation is
                # always retranslated from scratch and fully replaced.
                provisional_en, provisional_mt_times = self.live_pipeline.translate(
                    state.arabic_provisional
                )
                state.english_provisional = provisional_en
                mt_times.update(provisional_mt_times)
            else:
                state.english_provisional = ""

        asr_s = asr_times.get("asr", 0.0)
        mt_s = mt_times.get("mt", 0.0)
        state.status_message = (
            f"Live — {state.duration_sec():.1f}s recorded · "
            f"chunk {chunk_sec:.1f}s · ASR {asr_s:.1f}s · MT {mt_s:.1f}s"
        )
        logger.info(
            "Live increment %.1fs chunk -> Arabic: %s",
            chunk_sec,
            state.arabic_text[:120],
        )
        return state

    def _run_final(self, state: LiveStreamState) -> LiveStreamState:
        """Full-buffer pass with the high-quality model."""
        duration = state.duration_sec()
        state.status_message = f"Finalizing {duration:.1f}s with high-quality model…"

        with self._process_lock:
            final_pipeline = self._get_final_pipeline()
            arabic_text, _, asr_times = final_pipeline.transcribe(
                state.audio, state.sample_rate
            )
            english_text, mt_times = final_pipeline.translate(arabic_text)

        # Full-buffer high-quality pass supersedes any provisional guesses.
        state.arabic_verified = arabic_text
        state.arabic_provisional = ""
        state.english_verified = english_text
        state.english_provisional = ""
        state.processed_samples = state.audio.size

        asr_s = asr_times.get("asr", 0.0)
        mt_s = mt_times.get("mt", 0.0)
        if arabic_text:
            state.status_message = (
                f"Done — {duration:.1f}s · ASR {asr_s:.1f}s · MT {mt_s:.1f}s"
            )
            logger.info("Final transcript: %s", arabic_text[:200])
        else:
            state.status_message = f"No speech detected in {duration:.1f}s"

        if self._second_opinion_config.enabled and arabic_text:
            self._run_second_opinion_async(state)

        return state

    def _run_second_opinion_async(self, state: LiveStreamState) -> threading.Thread:
        """Fire-and-forget: compare the cascade's final English against the
        independent Seamless channel, and log the comparison. Runs in a
        background thread so a slow/failing second opinion never blocks or
        breaks the main transcription path (README step 6: self-contained).
        """
        audio = state.audio
        sample_rate = state.sample_rate
        cascade_english = state.english_text
        session_id = state.session_id
        log_path = self._second_opinion_config.log_path

        def _worker() -> None:
            try:
                if self._second_opinion_engine is None:
                    with self._second_opinion_lock:
                        if self._second_opinion_engine is None:
                            from ..second_opinion import build_second_opinion

                            self._second_opinion_engine = build_second_opinion(
                                self._second_opinion_config
                            )
                engine = self._second_opinion_engine
                assert engine is not None
                second_opinion_text = engine.translate_speech(audio, sample_rate)
                record = compare(cascade_english, second_opinion_text, session_id=session_id)
                log_record(record, log_path)
                logger.info(
                    "Second opinion for %s: similarity=%.2f",
                    session_id[:8],
                    record.similarity,
                )
            except Exception:
                logger.exception("Second-opinion channel failed for %s", session_id[:8])

        thread = threading.Thread(
            target=_worker, name=f"second-opinion-{session_id[:8]}", daemon=True
        )
        thread.start()
        return thread

    def finalize(self, session_id: str) -> LiveStreamState | None:
        state = self.get(session_id)
        if state is None:
            return None

        state.is_active = False

        for _ in range(50):
            if not state.processing:
                break
            time.sleep(0.1)

        state.processing = True
        try:
            self._run_final(state)
        finally:
            state.processing = False

        self.remove(session_id)
        return state

    def save_debug(self, session_id: str, output_dir: str) -> None:
        state = self.get(session_id)
        if state is None or state.audio.size == 0:
            return
        from pathlib import Path

        path = Path(output_dir) / f"debug_live_{session_id[:8]}.wav"
        save_wav(path, state.audio, state.sample_rate)
        logger.info("Debug audio saved: %s (%.2fs)", path, state.duration_sec())


__all__ = [
    "LiveSessionStore",
    "LiveStreamState",
    "MIN_AUDIO_SEC",
    "MIN_NEW_AUDIO_SEC",
    "merge_audio",
]
