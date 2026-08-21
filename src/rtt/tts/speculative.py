"""Speculative TTS shadow buffer and playback jitter buffer (README step 10).

README "Speculative pipeline": synthesized audio is fully speculative while
it sits in a **muted shadow buffer** - synthesize eagerly off the
provisional/grey text so playback is instant on commit, and discard silently
on rejection. Once unmuted (i.e. queued into the jitter buffer after a real
commit), audio is final: "no rollback, no correction" - only the shadow
buffer is ever speculative, never anything past it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .base import SpeechAudio, TTSEngine


@dataclass
class ShadowClip:
    """One speculatively-synthesized, still-muted candidate."""

    text: str
    audio: SpeechAudio


class SpeculativeTTS:
    """Eagerly synthesizes the current best-guess English text into a muted
    shadow slot, and resolves it on commit or rejection.

    Holds at most one shadow clip - the plan only ever has one "current best
    speculative branch" rendered on screen at a time (README step 7:
    "grey, damped to avoid visual flicker"), so there is only ever one
    plausible thing worth pre-synthesizing.
    """

    def __init__(self, engine: TTSEngine) -> None:
        self.engine = engine
        self._shadow: ShadowClip | None = None

    @property
    def shadow(self) -> ShadowClip | None:
        return self._shadow

    def speculate(self, provisional_text: str) -> ShadowClip | None:
        """Synthesize ``provisional_text`` into the muted shadow buffer.

        A no-op (cache hit) when it matches what's already shadowed - avoids
        re-synthesizing on every tick just because the provisional text
        hasn't changed. Passing an empty string clears the shadow.
        """
        if not provisional_text:
            self._shadow = None
            return None
        if self._shadow is not None and self._shadow.text == provisional_text:
            return self._shadow
        audio = self.engine.synthesize(provisional_text)
        self._shadow = ShadowClip(text=provisional_text, audio=audio)
        return self._shadow

    def commit(self, committed_text: str) -> tuple[SpeechAudio, bool]:
        """Resolve a real commit. Returns ``(audio, was_speculative_hit)``.

        On a hit, playback is instant - the audio already exists, just
        unmuted. On a miss (the commit didn't match what was shadowed, e.g.
        the guard/risk model committed something the speculative branch
        hadn't rendered yet), synthesize now, paying the latency the shadow
        buffer exists to avoid.
        """
        if self._shadow is not None and self._shadow.text == committed_text:
            audio = self._shadow.audio
            self._shadow = None
            return audio, True

        audio = self.engine.synthesize(committed_text)
        self._shadow = None
        return audio, False

    def reject(self) -> None:
        """Discard the shadow buffer silently - the speculative branch it
        was rendered from didn't survive.
        """
        self._shadow = None


@dataclass
class JitterBufferConfig:
    #: README steady-state target: ~0.5s of buffered audio before playback
    #: starts, trading a small fixed delay for smoothing out compute jitter.
    target_buffer_sec: float = 0.5


@dataclass
class JitterBuffer:
    """FIFO queue of committed (unmuted, final) audio awaiting playback.

    Only ever holds audio that has cleared :meth:`SpeculativeTTS.commit` -
    "must be verified before the ear" - so nothing queued here is ever
    rolled back; it is played in order, once ``target_buffer_sec`` worth has
    accumulated.
    """

    config: JitterBufferConfig = field(default_factory=JitterBufferConfig)
    _queue: deque[SpeechAudio] = field(default_factory=deque, init=False, repr=False)
    _buffered_sec: float = field(default=0.0, init=False)

    @property
    def buffered_sec(self) -> float:
        return self._buffered_sec

    def push(self, audio: SpeechAudio) -> None:
        if audio.is_empty():
            return
        self._queue.append(audio)
        self._buffered_sec += audio.duration

    def ready(self) -> bool:
        """True once enough audio is queued to start/continue playback
        without immediately starving - avoids an audible gap from starting
        playback the instant the very first commit lands.
        """
        return self._buffered_sec >= self.config.target_buffer_sec

    def pop(self) -> SpeechAudio | None:
        """Dequeue the next clip to play, or ``None`` if starved."""
        if not self._queue:
            return None
        audio = self._queue.popleft()
        self._buffered_sec = max(0.0, self._buffered_sec - audio.duration)
        return audio

    def clear(self) -> None:
        self._queue.clear()
        self._buffered_sec = 0.0


__all__ = ["JitterBuffer", "JitterBufferConfig", "ShadowClip", "SpeculativeTTS"]
