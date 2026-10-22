"""Bounded full-context live decoding for a CPU exhibition laptop.

Every update is a replaceable draft of the recording so far, just like the
Stop pass. There is one worker, no queue of stale chunks and no permanent
commit of words inferred from a short, mid-sentence window.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..audio import resample, to_mono_float32
from ..config import ASR_SAMPLE_RATE
from .session import LiveSessionStore, LiveStreamState, MAX_SESSION_SEC, SILENCE_RMS
from .vad import SegmentDecision


class ContextSessionStore(LiveSessionStore):
    update_audio_sec = 4.0

    def __init__(self, pipeline):
        super().__init__(pipeline, lambda: pipeline)
        self._audio_lock = threading.Lock()

    def active_count(self) -> int:
        # A disconnected native decode cannot be cancelled. Keep admission
        # closed until it has actually left the shared inference lock.
        with self._lock:
            return max(len(self._sessions), int(self._process_lock.locked()))

    def append_chunk(self, session_id, sample_rate, samples):
        incoming = resample(to_mono_float32(samples), sample_rate, ASR_SAMPLE_RATE)
        with self._audio_lock:
            state = self.get(session_id)
            if state is None or not state.is_active:
                return state
            remaining = max(0, int(MAX_SESSION_SEC * ASR_SAMPLE_RATE) - state.audio.size)
            state.audio = np.concatenate((state.audio, incoming[:remaining]))
            state.chunks_received += 1
            if state.audio.size >= int(MAX_SESSION_SEC * ASR_SAMPLE_RATE):
                state.status_message = "30-second recording limit reached — finalizing."
            return state

    def _next_segment(self, state):
        pending = state.audio.size - state.processed_samples
        # Quiet half-second after a short phrase: update without requiring
        # four seconds of speech. This is an energy heuristic, not a VAD claim.
        tail = state.audio[-8000:]
        preceding = state.audio[max(0, state.audio.size - 24000):-8000]
        phrase_pause = (pending >= 32000 and tail.size == 8000 and preceding.size > 0
                        and float(np.sqrt(np.mean(tail * tail))) < SILENCE_RMS
                        and float(np.sqrt(np.mean(preceding * preceding))) >= SILENCE_RMS)
        # Drain a partial window after a decode rather than waiting forever
        # for another 4s of speech. Otherwise a pause looks inaccurate right
        # up until Stop, even though the full context is already available.
        catch_up = (state.processed_samples > 0 and pending > 0
                    and time.time() - state.last_process_wall >= 1.0)
        if pending / ASR_SAMPLE_RATE >= self.update_audio_sec or catch_up or phrase_pause:
            return SegmentDecision(end=state.audio.size, closed=False, speech=True)
        return None

    def _decode(self, state, end=None):
        # Called under _process_lock. Snapshot all available audio, including
        # speech captured during the previous inference; never enqueue old cuts.
        with self._audio_lock:
            audio = state.audio.copy()
        if end is not None:
            audio = audio[:end]
        started = time.perf_counter()
        fresh = audio[state.processed_samples:]
        quiet_tail = fresh.size > 0 and float(np.sqrt(np.mean(fresh * fresh))) < SILENCE_RMS
        if not audio.size or float(np.sqrt(np.mean(audio * audio))) < SILENCE_RMS:
            arabic, english = "", ""
        elif quiet_tail and state.silence_rechecked:
            # Allow one decode with trailing silence to resolve the last word;
            # further silence adds no information and need not consume CPU.
            arabic, english = state.arabic_text, state.english_text
        else:
            arabic, _, _ = self.live_pipeline.transcribe(audio, ASR_SAMPLE_RATE)
            # Translate the complete Arabic sentence/context, rather than
            # concatenating unrelated translations of little fragments.
            if arabic == state.provisional_source:
                english = state.english_provisional or state.english_verified
            else:
                english, _ = self.live_pipeline.translate(arabic)
        return audio.size, arabic, english, time.perf_counter() - started, quiet_tail

    def _run_live_increment(self, state, chunk_end=None, closed=False):
        with self._process_lock:
            if not state.is_active:
                return state
            state.status_message = "Updating Arabic and English with full sentence context…"
            size, arabic, english, elapsed, quiet_tail = self._decode(state)
            if not state.is_active:
                return state
            state.arabic_verified = ""
            state.english_verified = ""
            state.arabic_provisional = arabic
            state.english_provisional = english
            state.provisional_source = arabic
            state.silence_rechecked = quiet_tail
            state.processed_samples = size
            state.decoded_duration_sec = size / ASR_SAMPLE_RATE
            state.decode_seconds = elapsed
            state.status_message = (
                f"Live draft — {state.decoded_duration_sec:.1f}s processed · "
                f"last update {elapsed:.1f}s · drafts can change"
                if arabic else "Listening — no speech detected yet."
            )
        return state

    def finalize(self, session_id):
        state = self.get(session_id)
        if state is None:
            return None
        state.is_active = False
        # Holding this lock through publication makes finalization and live
        # updates ordered. A delayed live worker cannot overwrite the final.
        with self._process_lock:
            state.processing = True
            try:
                if state.processed_samples == state.audio.size and state.audio.size:
                    arabic, english = state.arabic_provisional, state.english_provisional
                    size, elapsed = state.audio.size, 0.0
                else:
                    size, arabic, english, elapsed, _ = self._decode(state)
                state.arabic_verified = arabic
                state.english_verified = english
                state.arabic_provisional = state.english_provisional = ""
                state.provisional_source = arabic
                state.processed_samples = size
                state.decoded_duration_sec = size / ASR_SAMPLE_RATE
                state.decode_seconds = elapsed
                state.status_message = (
                    f"Final — {state.decoded_duration_sec:.1f}s recording"
                    if arabic else "No speech detected."
                )
            finally:
                state.processing = False
                self.remove(session_id)
        return state
