"""Interface for the Seamless second-opinion channel.

README "Verdict on cascade vs end-to-end": a direct speech-to-text-
translation model runs *alongside* the cascade as an independent channel,
precisely because it does not share the cascade's ASR errors. Two
architecturally independent systems agreeing is a much stronger safety
signal than one system agreeing with itself - this is an error-decorrelation
sensor, not a competing primary translator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..audio import AudioArray


class SecondOpinionEngine(ABC):
    """Arabic speech -> English text, in one step, independent of the
    cascade's ASR. No intermediate Arabic text is exposed - only that
    would let a shared ASR mistake infect both channels.
    """

    name: str = "second_opinion"

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Safe to call repeatedly."""

    @abstractmethod
    def translate_speech(self, audio: AudioArray, sample_rate: int) -> str:
        """Translate Arabic speech directly to English text."""


__all__ = ["SecondOpinionEngine"]
