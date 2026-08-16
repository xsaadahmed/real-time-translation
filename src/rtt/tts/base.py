"""Interfaces shared by all speech synthesis backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..audio import AudioArray


@dataclass
class SpeechAudio:
    """Synthesised audio as mono float32 plus its sample rate."""

    audio: AudioArray
    sample_rate: int

    @property
    def duration(self) -> float:
        return float(self.audio.size) / float(self.sample_rate) if self.sample_rate else 0.0

    def is_empty(self) -> bool:
        return self.audio.size == 0

    @classmethod
    def empty(cls, sample_rate: int = 22_050) -> "SpeechAudio":
        return cls(audio=np.zeros(0, dtype=np.float32), sample_rate=sample_rate)


class TTSEngine(ABC):
    """English text-to-speech.

    A future streaming backend will add an incremental ``synthesize_stream``
    method; batch synthesis stays the baseline path.
    """

    name: str = "tts"

    @abstractmethod
    def load(self) -> None:
        """Load voice/model resources. Safe to call repeatedly."""

    @abstractmethod
    def synthesize(self, text: str) -> SpeechAudio:
        """Render ``text`` to audio."""


__all__ = ["SpeechAudio", "TTSEngine"]
