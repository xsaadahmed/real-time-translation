"""English TTS through the Windows SAPI5 voices via pyttsx3.

The always-available fallback: no model download, no espeak-ng, no network.
Quality is modest, but it guarantees the pipeline produces audio on a stock
Windows machine.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from ..audio import to_mono_float32
from ..config import TTSConfig
from .base import SpeechAudio, TTSEngine

logger = logging.getLogger(__name__)


class SapiTTS(TTSEngine):
    name = "sapi"

    def __init__(self, config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig()
        self._voice_id: str | None = None
        self._loaded = False

    def _new_engine(self):
        """Create a fresh pyttsx3 engine.

        pyttsx3's SAPI5 driver can deadlock when ``runAndWait`` is called more
        than once on the same engine after ``save_to_file``, so each synthesis
        gets its own short-lived engine.
        """
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self.config.speaking_rate)
        if self._voice_id:
            engine.setProperty("voice", self._voice_id)
        return engine

    def _pick_voice(self, engine) -> str | None:
        voices = engine.getProperty("voices") or []
        requested = self.config.voice.strip().lower()

        if requested:
            for voice in voices:
                haystack = f"{voice.id} {getattr(voice, 'name', '')}".lower()
                if requested in haystack:
                    return voice.id
            logger.warning("TTS voice '%s' not found; falling back to an English voice", requested)

        for voice in voices:
            languages = [
                lang.decode("utf-8", "ignore") if isinstance(lang, bytes) else str(lang)
                for lang in (getattr(voice, "languages", None) or [])
            ]
            haystack = f"{voice.id} {getattr(voice, 'name', '')} {' '.join(languages)}".lower()
            if "en" in haystack or "english" in haystack:
                return voice.id

        return voices[0].id if voices else None

    def load(self) -> None:
        if self._loaded:
            return
        engine = self._new_engine()
        try:
            self._voice_id = self._pick_voice(engine)
            logger.info("SAPI5 voice: %s", self._voice_id)
        finally:
            engine.stop()
        self._loaded = True

    def synthesize(self, text: str) -> SpeechAudio:
        text = text.strip()
        if not text:
            return SpeechAudio.empty()

        self.load()

        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="rtt-tts-") as tmpdir:
            wav_path = Path(tmpdir) / "speech.wav"
            engine = self._new_engine()
            try:
                engine.save_to_file(text, str(wav_path))
                engine.runAndWait()
            finally:
                engine.stop()

            if not wav_path.exists() or wav_path.stat().st_size == 0:
                logger.error("SAPI5 produced no audio for %d characters of text", len(text))
                return SpeechAudio.empty()

            data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)

        return SpeechAudio(audio=to_mono_float32(np.asarray(data)), sample_rate=int(sample_rate))


__all__ = ["SapiTTS"]
