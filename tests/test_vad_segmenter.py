"""Segmentation decisions for the live path.

Silero itself is stubbed: what needs testing is where the segmenter cuts given
a set of speech spans, not whether Silero finds them.
"""

from __future__ import annotations

import numpy as np
import pytest

from rtt.config import ASR_SAMPLE_RATE, VADConfig
from rtt.live.vad import SpeechSegmenter

SR = ASR_SAMPLE_RATE


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _stub(monkeypatch, spans_sec):
    """Make the detector report ``spans_sec`` (seconds, relative to pending)."""
    import faster_whisper.vad as vad_mod

    def fake(audio, options=None, sampling_rate=SR, **kwargs):
        return [
            {"start": int(start * SR), "end": int(end * SR)} for start, end in spans_sec
        ]

    monkeypatch.setattr(vad_mod, "get_speech_timestamps", fake)


@pytest.fixture
def config() -> VADConfig:
    return VADConfig(
        enabled=True,
        threshold=0.5,
        min_silence_ms=300,
        speech_pad_ms=200,
        min_speech_ms=150,
        max_segment_sec=2.5,
        silence_skip_sec=1.5,
        min_pending_sec=0.3,
    )


def test_waits_until_enough_audio_arrives(config, monkeypatch):
    _stub(monkeypatch, [(0.0, 0.1)])
    segmenter = SpeechSegmenter(config)
    assert segmenter.decide(_audio(0.2), 0) is None


def test_pause_after_speech_closes_the_segment(config, monkeypatch):
    # 2s of speech then 0.5s of silence — a completed utterance.
    _stub(monkeypatch, [(0.0, 2.0)])
    segmenter = SpeechSegmenter(config)

    decision = segmenter.decide(_audio(2.5), 0)

    assert decision is not None
    assert decision.closed is True
    assert decision.speech is True
    # Cut at end of speech plus the configured pad, not at the buffer end.
    assert decision.end == int(2.2 * SR)


def test_speech_still_in_progress_does_not_cut(config, monkeypatch):
    # Speech runs right up to the buffer end: only 0.1s of trailing silence,
    # below the 300ms that closes an utterance, and under max_segment_sec.
    _stub(monkeypatch, [(0.0, 1.9)])
    segmenter = SpeechSegmenter(config)
    assert segmenter.decide(_audio(2.0), 0) is None


def test_unbroken_speech_is_force_cut_but_stays_open(config, monkeypatch):
    _stub(monkeypatch, [(0.0, 2.9)])
    segmenter = SpeechSegmenter(config)

    decision = segmenter.decide(_audio(3.0), 0)

    assert decision is not None
    assert decision.closed is False, "a monologue tail may still be revised"
    assert decision.speech is True
    assert decision.end == 3 * SR


def test_pure_silence_is_skipped_without_asr(config, monkeypatch):
    _stub(monkeypatch, [])
    segmenter = SpeechSegmenter(config)

    decision = segmenter.decide(_audio(2.0), 0)

    assert decision is not None
    assert decision.skip_asr is True
    assert decision.end == 2 * SR


def test_short_silence_is_not_yet_skipped(config, monkeypatch):
    _stub(monkeypatch, [])
    segmenter = SpeechSegmenter(config)
    assert segmenter.decide(_audio(1.0), 0) is None


def test_decisions_are_relative_to_the_cursor(config, monkeypatch):
    # 10s already transcribed; the pending tail holds 2s of speech + a pause.
    _stub(monkeypatch, [(0.0, 2.0)])
    segmenter = SpeechSegmenter(config)

    decision = segmenter.decide(_audio(12.5), 10 * SR)

    assert decision is not None
    assert decision.closed is True
    assert decision.end == 10 * SR + int(2.2 * SR)


def test_detector_failure_falls_back_to_a_timed_cut(config, monkeypatch):
    import faster_whisper.vad as vad_mod

    def boom(*args, **kwargs):
        raise RuntimeError("onnx exploded")

    monkeypatch.setattr(vad_mod, "get_speech_timestamps", boom)
    segmenter = SpeechSegmenter(config)

    # Below the force-cut threshold the failure is simply swallowed.
    assert segmenter.decide(_audio(1.0), 0) is None

    # Past it, the live path keeps running on the old fixed-window behaviour.
    decision = segmenter.decide(_audio(3.0), 0)
    assert decision is not None
    assert decision.closed is False
    assert decision.speech is True
