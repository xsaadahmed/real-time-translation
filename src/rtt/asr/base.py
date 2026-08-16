"""Interfaces shared by all speech recognition backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..audio import AudioArray


@dataclass
class TranscriptSegment:
    """One timestamped chunk of recognised speech.

    Timestamps are kept even though the current pipeline is offline: the
    streaming commitment policy planned for later needs them to measure lag.
    """

    start: float
    end: float
    text: str

    def as_dict(self) -> dict:
        return {"start": round(self.start, 2), "end": round(self.end, 2), "text": self.text}


@dataclass
class Transcript:
    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    audio_duration: float = 0.0

    def is_empty(self) -> bool:
        return not self.text.strip()


class ASREngine(ABC):
    """Batch speech-to-text. A streaming variant will subclass this later."""

    name: str = "asr"

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Safe to call repeatedly."""

    @abstractmethod
    def transcribe(self, audio: AudioArray, sample_rate: int) -> Transcript:
        """Transcribe mono float32 audio."""


__all__ = ["ASREngine", "Transcript", "TranscriptSegment"]
