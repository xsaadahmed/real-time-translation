"""Arabic to English translation backends."""

from __future__ import annotations

from ..config import MTConfig
from .agreement import AgreementResult, agreement_depth, default_quorum
from .base import Translator
from .huggingface import DEFAULT_MODELS, HuggingFaceTranslator


def build_translator(config: MTConfig | None = None) -> Translator:
    """Construct the configured translator."""
    return HuggingFaceTranslator(config or MTConfig())


__all__ = [
    "AgreementResult",
    "DEFAULT_MODELS",
    "HuggingFaceTranslator",
    "Translator",
    "agreement_depth",
    "build_translator",
    "default_quorum",
]
