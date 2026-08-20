"""Arabic-aware text utilities.

Sentence splitting matters more than it looks: MarianMT truncates at 512
tokens and degrades on long inputs, so a paragraph-length transcript has to be
broken up before translation. This module is also the natural home for the
Arabic structural analysis the commitment policy will need later.
"""

from __future__ import annotations

import re

# Latin and Arabic sentence terminators. The Arabic question mark (؟) and the
# Urdu/Arabic full stop (۔) are separate codepoints from their Latin lookalikes.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[\.!\?؟۔])[\s\u200f\u200e]+|\n+")

# Soft boundaries used only when a sentence is still too long to translate in
# one piece. The Arabic comma (،) and semicolon (؛) are distinct codepoints.
_SOFT_BOUNDARY = re.compile(r"(?<=[،؛,;:])\s+")

_TATWEEL = "\u0640"
_WHITESPACE = re.compile(r"[ \t\u00a0]+")


def normalize_arabic(text: str) -> str:
    """Light normalisation: drop tatweel padding and collapse whitespace.

    Deliberately conservative. Diacritics and hamza forms are left alone
    because they carry information the translator can use.
    """
    text = text.replace(_TATWEEL, "")
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split Arabic (or English) text into sentence-like units."""
    text = normalize_arabic(text)
    if not text:
        return []
    parts = (part.strip() for part in _SENTENCE_BOUNDARY.split(text))
    return [part for part in parts if part]


def _split_long(sentence: str, max_words: int) -> list[str]:
    """Break an over-long sentence at soft punctuation, then at word count."""
    pieces: list[str] = []
    for clause in _SOFT_BOUNDARY.split(sentence):
        clause = clause.strip()
        if not clause:
            continue
        words = clause.split()
        if len(words) <= max_words:
            pieces.append(clause)
            continue
        for start in range(0, len(words), max_words):
            pieces.append(" ".join(words[start : start + max_words]))
    return pieces


def chunk_for_translation(text: str, max_words: int = 40) -> list[str]:
    """Split text into translation-sized chunks, never exceeding ``max_words``."""
    text = normalize_arabic(text)
    if not text:
        return []

    # Long Arabic monologues often lack Latin periods — split on Arabic commas too.
    if len(text.split()) > max_words and "،" in text:
        clause_parts = [part.strip() for part in text.split("،") if part.strip()]
        if len(clause_parts) > 1:
            chunks: list[str] = []
            for clause in clause_parts:
                if len(clause.split()) <= max_words:
                    chunks.append(clause)
                else:
                    chunks.extend(_split_long(clause, max_words))
            return chunks

    chunks: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence.split()) <= max_words:
            chunks.append(sentence)
        else:
            chunks.extend(_split_long(sentence, max_words))
    return chunks


def join_translations(pieces: list[str]) -> str:
    """Join translated chunks back into a single readable paragraph."""
    cleaned = [piece.strip() for piece in pieces if piece and piece.strip()]
    return " ".join(cleaned)


def merge_incremental_text(existing: str, new: str) -> str:
    """Append new ASR text, merging overlapping words at the boundary.

    Append-only: this has no way to correct earlier words if a later ASR
    pass revises them. Only safe to use on text that is already verified
    (immutable) and therefore only ever grows. For a mutable, re-transcribed
    tail, use :func:`reconcile_provisional` instead.
    """
    existing = normalize_arabic(existing)
    new = normalize_arabic(new)
    if not existing:
        return new
    if not new:
        return existing

    existing_words = existing.split()
    new_words = new.split()
    max_overlap = min(len(existing_words), len(new_words))
    for overlap in range(max_overlap, 0, -1):
        if existing_words[-overlap:] == new_words[:overlap]:
            return " ".join(existing_words + new_words[overlap:])
    return f"{existing} {new}"


def reconcile_provisional(old_provisional: str, new_hypothesis: str) -> tuple[str, str]:
    """Reconcile a fresh ASR hypothesis against the previous provisional tail.

    Live ASR re-transcribes a sliding window of recent audio on every pass,
    so ``new_hypothesis`` covers roughly the same audio as ``old_provisional``
    plus whatever was newly recorded, and may revise any word in that window
    (streaming Whisper does this constantly). Two independent passes producing
    the same word at the same position is exactly the "drafter and verifier
    agree across two consecutive frames" signal the commitment policy needs,
    so that leading run of agreement is promoted to verified; everything from
    the first disagreement onward is still liable to change and stays
    provisional, replacing the old guess outright (never appended to it).

    Matching is whole-word: a word that differs only by a suffix (``كتاب``
    vs ``كتابه``) does not match and is correctly kept provisional rather
    than verified as a partial word.

    Returns ``(newly_verified, remaining_provisional)``.
    """
    old_provisional = normalize_arabic(old_provisional)
    new_hypothesis = normalize_arabic(new_hypothesis)
    if not old_provisional:
        return "", new_hypothesis
    if not new_hypothesis:
        # This pass heard nothing new (e.g. a silent gap) - nothing new to
        # confirm, and no reason to discard the previous best guess.
        return "", old_provisional

    old_words = old_provisional.split()
    new_words = new_hypothesis.split()
    agree = 0
    for old_word, new_word in zip(old_words, new_words):
        if old_word != new_word:
            break
        agree += 1

    newly_verified = " ".join(new_words[:agree])
    remaining_provisional = " ".join(new_words[agree:])
    return newly_verified, remaining_provisional


__all__ = [
    "chunk_for_translation",
    "join_translations",
    "merge_incremental_text",
    "normalize_arabic",
    "reconcile_provisional",
    "split_sentences",
]
