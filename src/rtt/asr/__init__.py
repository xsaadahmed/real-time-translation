"""Speech recognition backends."""

from __future__ import annotations

from ..config import ASRConfig
from .base import ASREngine, Transcript, TranscriptSegment
from .faster_whisper_asr import FasterWhisperASR


def build_asr(config: ASRConfig | None = None) -> ASREngine:
    """Construct the configured ASR engine."""
    return FasterWhisperASR(config or ASRConfig())


__all__ = ["ASREngine", "FasterWhisperASR", "Transcript", "TranscriptSegment", "build_asr"]
