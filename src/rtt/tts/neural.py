"""Optional higher-quality local TTS backends: Kokoro and Piper.

Both are entirely offline but need extra installs, so neither is a hard
dependency. :func:`rtt.tts.build_tts` tries them ahead of the SAPI5 fallback
and quietly moves on if they are unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..audio import to_mono_float32
from ..config import TTSConfig
from .base import SpeechAudio, TTSEngine

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24_000
DEFAULT_KOKORO_VOICE = "af_heart"


class KokoroTTS(TTSEngine):
    """Kokoro-82M. Needs ``pip install kokoro`` and espeak-ng on PATH."""

    name = "kokoro"

    def __init__(self, config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig()
        self.voice = self.config.voice or DEFAULT_KOKORO_VOICE
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        from kokoro import KPipeline

        # lang_code "a" selects American English.
        self._pipeline = KPipeline(lang_code="a")
        logger.info("Loaded Kokoro with voice '%s'", self.voice)

    def synthesize(self, text: str) -> SpeechAudio:
        text = text.strip()
        if not text:
            return SpeechAudio.empty(KOKORO_SAMPLE_RATE)

        self.load()
        assert self._pipeline is not None

        chunks: list[np.ndarray] = []
        for result in self._pipeline(text, voice=self.voice):
            # Kokoro <=0.8 yields (graphemes, phonemes, audio); >=0.9 yields an
            # object with an .audio attribute.
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)):
                audio = result[-1]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

        if not chunks:
            return SpeechAudio.empty(KOKORO_SAMPLE_RATE)
        return SpeechAudio(audio=np.concatenate(chunks), sample_rate=KOKORO_SAMPLE_RATE)


class PiperTTS(TTSEngine):
    """Piper. Needs ``pip install piper-tts`` and a downloaded ``.onnx`` voice."""

    name = "piper"

    def __init__(self, config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig()
        self._voice = None
        self._sample_rate = 22_050

    def _voice_path(self) -> Path:
        if self.config.piper_voice_path:
            return Path(self.config.piper_voice_path)
        candidates = sorted(Path(self.config.cache_dir, "piper").glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(
                "No Piper voice found. Set RTT_PIPER_VOICE to a .onnx voice file "
                f"or place one in {Path(self.config.cache_dir, 'piper')}"
            )
        return candidates[0]

    def load(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice

        path = self._voice_path()
        self._voice = PiperVoice.load(str(path))
        self._sample_rate = int(self._voice.config.sample_rate)
        logger.info("Loaded Piper voice %s at %d Hz", path.name, self._sample_rate)

    def synthesize(self, text: str) -> SpeechAudio:
        text = text.strip()
        if not text:
            return SpeechAudio.empty(self._sample_rate)

        self.load()
        assert self._voice is not None

        chunks: list[np.ndarray] = []
        if hasattr(self._voice, "synthesize_stream_raw"):
            for raw in self._voice.synthesize_stream_raw(text):
                chunks.append(np.frombuffer(raw, dtype=np.int16))
        else:
            for chunk in self._voice.synthesize(text):
                array = getattr(chunk, "audio_int16_array", None)
                if array is None:
                    array = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                chunks.append(np.asarray(array, dtype=np.int16))

        if not chunks:
            return SpeechAudio.empty(self._sample_rate)
        return SpeechAudio(
            audio=to_mono_float32(np.concatenate(chunks)), sample_rate=self._sample_rate
        )


__all__ = ["KokoroTTS", "PiperTTS"]
