"""Interfaces shared by all translation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..text import chunk_for_translation, join_translations


class Translator(ABC):
    """Arabic to English translation.

    Subclasses implement :meth:`translate_batch` only; chunking a long
    transcript into translatable pieces is handled here so every backend gets
    the same behaviour.
    """

    name: str = "translator"
    source_lang: str = "ar"
    target_lang: str = "en"

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Safe to call repeatedly."""

    @abstractmethod
    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a list of short, self-contained chunks."""

    def translate(self, text: str) -> str:
        """Translate arbitrary-length text."""
        chunks = chunk_for_translation(text)
        if not chunks:
            return ""
        return join_translations(self.translate_batch(chunks))


__all__ = ["Translator"]
