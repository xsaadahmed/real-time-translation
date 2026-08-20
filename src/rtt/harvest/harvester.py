"""Offline retrospective-label harvesting (step 7).

Simulates the live draft lane over one pre-recorded utterance: re-transcribes
a sliding window on each tick (same shape as live/session.py's increment
loop, but offline and wall-clock-free), reconciles verified/provisional
Arabic, translates, and checks guards - exactly the signals the commit
policy will see at runtime. Once the whole utterance is available, a
high-quality final pass gives ground truth, and every candidate is labeled
by whether it actually survived to that final sentence.

Deliberately NOT wired into a script that runs it, because there is no real
Arabic audio in this repo to run it against yet - see harvest_utterance's
docstring. Validated here against stub ASR/Translator backends instead.
"""

from __future__ import annotations

import logging

from ..asr.base import ASREngine
from ..audio import AudioArray, duration_seconds
from ..mt.base import Translator
from ..second_opinion.agreement import compare as second_opinion_compare
from ..second_opinion.base import SecondOpinionEngine
from ..text import check_structural_guards, merge_incremental_text, reconcile_provisional
from .record import CommitCandidateRecord, label_survival

logger = logging.getLogger(__name__)

TICK_SEC = 0.9
CONTEXT_SEC = 3.0
MIN_AUDIO_SEC = 0.5


def harvest_utterance(
    audio: AudioArray,
    sample_rate: int,
    asr: ASREngine,
    translator: Translator,
    final_asr: ASREngine,
    final_translator: Translator,
    session_id: str = "",
    second_opinion: SecondOpinionEngine | None = None,
    tick_sec: float = TICK_SEC,
    context_sec: float = CONTEXT_SEC,
) -> list[CommitCandidateRecord]:
    """Replay one complete utterance as a sequence of live-style ticks and
    retrospectively label every candidate prefix considered along the way.

    ``asr``/``translator`` are the fast draft-lane backends (mirroring the
    live path); ``final_asr``/``final_translator`` are the accurate backends
    used once, at the end, purely to produce ground truth for labeling -
    exactly like ``LiveSessionStore._run_final``.

    To actually harvest data, call this once per Arabic recording in a real
    corpus (README step 7 says "hours of Arabic speech") and append the
    results with :func:`record.log_records`. No such corpus exists in this
    repo yet.
    """
    total_sec = duration_seconds(audio, sample_rate)
    tick_samples = max(1, int(tick_sec * sample_rate))
    context_samples = max(1, int(context_sec * sample_rate))
    min_audio_samples = int(MIN_AUDIO_SEC * sample_rate)

    records: list[CommitCandidateRecord] = []
    arabic_verified = ""
    arabic_provisional = ""
    english_verified = ""

    processed = 0
    tick = 0
    total_samples = audio.size
    while processed < total_samples:
        target = min(processed + tick_samples, total_samples)
        window_start = max(0, target - context_samples)
        chunk = audio[window_start:target]

        if chunk.size < min_audio_samples:
            processed = target
            continue

        transcript = asr.transcribe(chunk, sample_rate)
        newly_verified, arabic_provisional = reconcile_provisional(
            arabic_provisional, transcript.text
        )
        if newly_verified:
            arabic_verified = merge_incremental_text(arabic_verified, newly_verified)
            verified_en = translator.translate(newly_verified)
            if verified_en:
                english_verified = merge_incremental_text(english_verified, verified_en)

        provisional_en = translator.translate(arabic_provisional) if arabic_provisional else ""
        english_candidate = " ".join(p for p in (english_verified, provisional_en) if p)

        current_tail = " ".join(p for p in (arabic_verified, arabic_provisional) if p)
        guard_result = check_structural_guards(current_tail)

        records.append(
            CommitCandidateRecord(
                session_id=session_id,
                tick=tick,
                arabic_verified=arabic_verified,
                arabic_provisional=arabic_provisional,
                english_candidate=english_candidate,
                guard_hold=guard_result.hold,
                guard_name=guard_result.guard,
                guard_reason=guard_result.reason,
            )
        )

        processed = target
        tick += 1

    if not records:
        logger.info("Utterance %s produced no candidates (too short: %.2fs)", session_id, total_sec)
        return records

    final_transcript = final_asr.transcribe(audio, sample_rate)
    final_english = final_translator.translate(final_transcript.text)

    second_opinion_similarity = None
    if second_opinion is not None:
        second_opinion_text = second_opinion.translate_speech(audio, sample_rate)
        second_opinion_similarity = second_opinion_compare(
            final_english, second_opinion_text, session_id=session_id
        ).similarity

    for candidate in records:
        candidate.final_english = final_english
        candidate.survived = label_survival(candidate.english_candidate, final_english)
        candidate.second_opinion_similarity = second_opinion_similarity

    return records


__all__ = ["CONTEXT_SEC", "MIN_AUDIO_SEC", "TICK_SEC", "harvest_utterance"]
