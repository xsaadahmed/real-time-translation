"""Tests for the speculative TTS shadow buffer and jitter buffer (step 10)."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.tts.base import SpeechAudio, TTSEngine
from rtt.tts.speculative import JitterBuffer, JitterBufferConfig, SpeculativeTTS

SAMPLE_RATE = 16_000


class CountingTTS(TTSEngine):
    """Fake TTS: one 'sample' of audio per word, so duration and synth
    call-count are both cheaply verifiable without a real voice model.
    """

    name = "counting"

    def __init__(self) -> None:
        self.synth_calls: list[str] = []

    def load(self) -> None:
        pass

    def synthesize(self, text: str) -> SpeechAudio:
        self.synth_calls.append(text)
        n_words = len(text.split())
        audio = np.ones(n_words * SAMPLE_RATE, dtype=np.float32)
        return SpeechAudio(audio=audio, sample_rate=SAMPLE_RATE)


# ---------------------------------------------------------------------------
# SpeculativeTTS
# ---------------------------------------------------------------------------


def test_speculate_synthesizes_into_shadow_buffer():
    engine = CountingTTS()
    spec = SpeculativeTTS(engine)

    clip = spec.speculate("hello world")
    assert clip is not None
    assert clip.text == "hello world"
    assert spec.shadow is clip
    assert engine.synth_calls == ["hello world"]


def test_speculate_is_a_cache_hit_when_text_unchanged():
    engine = CountingTTS()
    spec = SpeculativeTTS(engine)

    spec.speculate("hello world")
    spec.speculate("hello world")
    assert engine.synth_calls == ["hello world"]  # only synthesized once


def test_speculate_replaces_shadow_on_changed_text():
    engine = CountingTTS()
    spec = SpeculativeTTS(engine)

    spec.speculate("hello")
    spec.speculate("hello world")
    assert spec.shadow.text == "hello world"
    assert engine.synth_calls == ["hello", "hello world"]


def test_speculate_empty_text_clears_shadow():
    engine = CountingTTS()
    spec = SpeculativeTTS(engine)
    spec.speculate("hello")
    spec.speculate("")
    assert spec.shadow is None


def test_commit_is_speculative_hit_when_text_matches_shadow():
    engine = CountingTTS()
    spec = SpeculativeTTS(engine)
    spec.speculate("hello world")

    audio, was_hit = spec.commit("hello world")
    assert was_hit is True
    assert audio.duration == 2.0  # 2 words -> 2 seconds of fake audio
    assert engine.synth_calls == ["hello world"]  # no extra synth on commit
    assert spec.shadow is None


def test_commit_is_speculative_miss_when_text_does_not_match_shadow():
    engine = CountingTTS()
    spec = SpeculativeTTS(engine)
    spec.speculate("hello there")

    audio, was_hit = spec.commit("hello world")
    assert was_hit is False
    assert engine.synth_calls == ["hello there", "hello world"]
    assert spec.shadow is None


def test_commit_with_empty_shadow_synthesizes_fresh():
    engine = CountingTTS()
    spec = SpeculativeTTS(engine)

    audio, was_hit = spec.commit("hello world")
    assert was_hit is False
    assert engine.synth_calls == ["hello world"]


def test_reject_discards_shadow_silently():
    engine = CountingTTS()
    spec = SpeculativeTTS(engine)
    spec.speculate("hello world")
    spec.reject()
    assert spec.shadow is None
    # A subsequent commit must synthesize fresh - nothing to reuse.
    spec.commit("hello world")
    assert engine.synth_calls == ["hello world", "hello world"]


# ---------------------------------------------------------------------------
# JitterBuffer
# ---------------------------------------------------------------------------


def _audio(seconds: float) -> SpeechAudio:
    return SpeechAudio(audio=np.ones(int(seconds * SAMPLE_RATE), dtype=np.float32), sample_rate=SAMPLE_RATE)


def test_jitter_buffer_not_ready_below_target():
    buf = JitterBuffer(JitterBufferConfig(target_buffer_sec=0.5))
    buf.push(_audio(0.3))
    assert buf.ready() is False


def test_jitter_buffer_ready_once_target_reached():
    buf = JitterBuffer(JitterBufferConfig(target_buffer_sec=0.5))
    buf.push(_audio(0.3))
    buf.push(_audio(0.3))
    assert buf.ready() is True
    assert buf.buffered_sec == 0.6


def test_jitter_buffer_pop_is_fifo_and_updates_buffered_sec():
    buf = JitterBuffer()
    buf.push(_audio(1.0))
    buf.push(_audio(2.0))

    first = buf.pop()
    assert first.duration == 1.0
    assert buf.buffered_sec == 2.0

    second = buf.pop()
    assert second.duration == 2.0
    assert buf.buffered_sec == 0.0


def test_jitter_buffer_pop_empty_returns_none():
    buf = JitterBuffer()
    assert buf.pop() is None


def test_jitter_buffer_ignores_empty_audio():
    buf = JitterBuffer()
    buf.push(SpeechAudio.empty(SAMPLE_RATE))
    assert buf.buffered_sec == 0.0
    assert buf.pop() is None


def test_jitter_buffer_clear():
    buf = JitterBuffer()
    buf.push(_audio(1.0))
    buf.clear()
    assert buf.buffered_sec == 0.0
    assert buf.pop() is None
