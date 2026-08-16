"""Arabic to English translation backends."""

from __future__ import annotations

from ..config import MTConfig
from .base import Translator
from .huggingface import DEFAULT_MODELS, HuggingFaceTranslator


def build_translator(config: MTConfig | None = None) -> Translator:
    """Construct the configured translator."""
    return HuggingFaceTranslator(config or MTConfig())


__all__ = ["DEFAULT_MODELS", "HuggingFaceTranslator", "Translator", "build_translator"]
