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

    def translate_branches(self, observed: str, futures: list[str]) -> list[str]:
        """Translate the observed-only branch plus one branch per sampled
        Arabic future, as a single batched call (README "Branched translator").

        ``futures`` are short sampled continuations of ``observed`` (from the
        0.5B Arabic futures model, not yet implemented). Returns ``len(futures)
        + 1`` English translations: index 0 is the observed-only branch,
        indices 1..K correspond to ``futures`` in order.

        This base implementation is the CPU / no-shared-KV-prefix baseline -
        it just batches K+1 independent short strings through translate_batch,
        with no speculative-decoding speedup. It exists to let the branch-
        agreement scoring (see agreement.py) be validated against the current
        Marian/NLLB backend now, on hardware with no GPU, well before the
        target Qwen3 + vLLM + EAGLE-3 + shared-KV-prefix setup is available.
        A backend with a real shared prefix can override this method without
        changing what callers see.
        """
        observed = observed.strip()
        branch_texts = [observed] + [
            f"{observed} {future}".strip() if future.strip() else observed
            for future in futures
        ]
        return self.translate_batch(branch_texts)


__all__ = ["Translator"]
