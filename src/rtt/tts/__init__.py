"""English speech synthesis backends."""

from __future__ import annotations

import logging

from ..config import TTSConfig
from .base import SpeechAudio, TTSEngine
from .neural import KokoroTTS, PiperTTS
from .sapi import SapiTTS
from .speculative import JitterBuffer, JitterBufferConfig, ShadowClip, SpeculativeTTS

logger = logging.getLogger(__name__)

BACKENDS: dict[str, type[TTSEngine]] = {
    "kokoro": KokoroTTS,
    "piper": PiperTTS,
    "sapi": SapiTTS,
}

#: Tried in order when the backend is "auto". SAPI5 is last because it is the
#: lowest quality, but it is also the only one guaranteed to be installable.
AUTO_ORDER = ("kokoro", "piper", "sapi")


def build_tts(config: TTSConfig | None = None) -> TTSEngine:
    """Construct a TTS engine, falling back through :data:`AUTO_ORDER`."""
    config = config or TTSConfig()

    if config.backend != "auto":
        if config.backend not in BACKENDS:
            raise ValueError(
                f"Unknown TTS backend '{config.backend}'. "
                f"Expected one of: {', '.join(sorted(BACKENDS))} or 'auto'"
            )
        engine = BACKENDS[config.backend](config)
        engine.load()
        return engine

    errors: list[str] = []
    for name in AUTO_ORDER:
        try:
            engine = BACKENDS[name](config)
            engine.load()
            logger.info("Using TTS backend '%s'", name)
            return engine
        except Exception as exc:  # noqa: BLE001 - probing optional backends
            errors.append(f"{name}: {exc}")
            logger.debug("TTS backend '%s' unavailable: %s", name, exc)

    raise RuntimeError("No TTS backend could be loaded.\n  " + "\n  ".join(errors))


__all__ = [
    "AUTO_ORDER",
    "BACKENDS",
    "JitterBuffer",
    "JitterBufferConfig",
    "KokoroTTS",
    "PiperTTS",
    "SapiTTS",
    "ShadowClip",
    "SpeculativeTTS",
    "SpeechAudio",
    "TTSEngine",
    "build_tts",
]
