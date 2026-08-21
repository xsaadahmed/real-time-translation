"""Arabic-aware text utilities.

Sentence splitting matters more than it looks: MarianMT truncates at 512
tokens and degrades on long inputs, so a paragraph-length transcript has to be
broken up before translation. This module is also the natural home for the
Arabic structural analysis the commitment policy will need later.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

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


#: Fraction of words that must match for an overlap to be believed. ASR revises
#: wording between passes (منزل التفولة -> منذ الطفولة), so requiring an exact
#: match finds no overlap at all on real output; half the words agreeing is
#: strong evidence for the same span of audio.
_ALIGN_MIN_AGREEMENT = 0.5


def _align_shift(old_words: list[str], new_words: list[str]) -> int | None:
    """Find the shift that lines the two hypotheses up, or ``None``.

    Returns ``s`` such that ``old_words[i]`` describes the same audio as
    ``new_words[i + s]``. It runs in both directions because both happen:

    * ``s < 0`` — the live path, where a pass replays some context and so
      starts *later* than the previous provisional began.
    * ``s > 0`` — a cumulative transcriber (the harvester's), which re-reports
      the whole utterance from the beginning every tick, and so starts
      *earlier* than the previous tail.

    Candidates are scored by how many words actually match, preferring the
    alignment with the most evidence rather than the first one that clears the
    threshold — a lone coincidental match should not beat a long agreeing run.
    """
    best_key: tuple[int, float] | None = None
    best_shift: int | None = None

    for shift in range(-(len(old_words) - 1), len(new_words)):
        low = max(0, -shift)
        high = min(len(old_words), len(new_words) - shift)
        span = high - low
        if span <= 0:
            continue
        matches = sum(
            1 for i in range(low, high) if old_words[i] == new_words[i + shift]
        )
        if not matches:
            continue
        ratio = matches / span
        if ratio < _ALIGN_MIN_AGREEMENT:
            continue
        key = (matches, ratio)
        if best_key is None or key > best_key:
            best_key, best_shift = key, shift

    return best_shift


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

    The two strings do not start at the same point in the audio, and assuming
    they do is what made this lose most of the transcript. A pass decodes
    ``LIVE_CONTEXT_SEC`` of already-seen audio before the new material, so a
    fresh hypothesis typically begins somewhere *inside* the previous
    provisional rather than at its first word::

        old: بالهي جزء أساسي من الهوية. في بيروت تسمع العربية والفرسية
        new:                              تسمع العربية والفرسية في الشارع

    Compared from index 0 those disagree immediately, nothing is ever promoted,
    and each pass discards its predecessor. Measured, 13 of 14 passes committed
    zero words. So the overlap is located first, and only then compared.

    Words of ``old_provisional`` that fall *before* the overlap are already
    outside the re-transcription window — no later pass will look at that audio
    again. They are committed rather than dropped: single-pass text is worth
    more than silence, and holding out for a confirmation that can never arrive
    is what lost them.

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

    shift = _align_shift(old_words, new_words)
    if shift is None:
        # The two passes have nothing in common — a long silence, or the
        # speaker moved on entirely. Keep the fresh hypothesis and commit
        # nothing, as before.
        return "", new_hypothesis

    if shift <= 0:
        # New pass starts inside the old tail. What precedes it has aged out of
        # the window and will never be re-transcribed, so commit it.
        offset = -shift
        aged_out = old_words[:offset]
        old_tail = old_words[offset:]
        new_start = 0
    else:
        # New pass reaches back before the old tail. That earlier span is
        # already committed, so skip it rather than emitting it twice.
        aged_out = []
        old_tail = old_words
        new_start = shift

    agree = 0
    for old_word, new_word in zip(old_tail, new_words[new_start:]):
        if old_word != new_word:
            break
        agree += 1

    # Aged-out words first, then the run this pass confirms.
    newly_verified = " ".join(aged_out + new_words[new_start : new_start + agree])
    remaining_provisional = " ".join(new_words[new_start + agree :])
    return newly_verified, remaining_provisional



# ---------------------------------------------------------------------------
# Structural guards
#
# These catch constructions where committing a translation of the observed
# tail risks being wrong once more Arabic arrives - not because the ASR/MT is
# uncertain, but because Arabic word order or morphology hasn't disclosed the
# information English needs yet. See README "Arabic structural guards".
#
# Each guard looks only at the tail's last word (the word nearest the commit
# boundary). Cheap, lexicon-only guards (TAM particles, numerals, dangling
# proclitics) need no dependency; the POS-dependent guards (VSO subject,
# iḍāfa) need camel-tools' dictionary Analyzer, loaded lazily and only once.
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    """Result of a structural-guard check on a candidate commit tail."""

    hold: bool
    reason: str = ""
    guard: str = ""


# Pre-verbal TAM (tense-aspect-mood) particles: what follows changes the verb
# tense/negation in English, so committing before it arrives is unrecoverable
# (e.g. "كان يذهب" = "was going", not "goes"; "لم يذهب" = "did not go").
_TAM_PARTICLES = {"كان", "كانت", "لم", "قد", "سوف", "لن"}

# Arabic 21-99 are spoken ones-then-tens (واحد وعشرون = "twenty-one"), the
# reverse of English word order. Emitting "one" before "وعشرون" arrives
# produces an unrecoverable "one ... twenty" ordering.
_NUMERAL_ONES = {
    "واحد", "واحدة", "اثنان", "اثنين", "ثلاثة", "أربعة", "خمسة",
    "ستة", "سبعة", "ثمانية", "تسعة",
}

# Single-letter proclitics (و=and, ف=so/then, ب=by/with, ك=like, ل=to/for)
# that normally attach to the following word. A standalone one-letter token
# means the word it belongs to hasn't been heard/segmented yet.
_DANGLING_PROCLITICS = {"و", "ف", "ب", "ك", "ل"}

_DEFINITE_PREFIX = "ال"

_CAMEL_DB_NAME = "calima-msa-r13"

_CAMEL_INSTALL_HINT = (
    "Install optional morphology guards with: "
    "pip install -r requirements-guards.txt && "
    "python scripts/download_models.py --camel-data-only"
)


def _guard_tam_particle(last_word: str) -> GuardResult:
    if last_word in _TAM_PARTICLES:
        return GuardResult(True, f"'{last_word}' sets up a verb not yet heard", "tam_particle")
    return GuardResult(False)


def _guard_partial_numeral(last_word: str) -> GuardResult:
    if last_word in _NUMERAL_ONES:
        return GuardResult(
            True, f"'{last_word}' may be the ones-digit of a compound number", "partial_numeral"
        )
    return GuardResult(False)


def _guard_dangling_proclitic(last_word: str) -> GuardResult:
    if last_word in _DANGLING_PROCLITICS:
        return GuardResult(
            True, f"'{last_word}' is a proclitic awaiting its host word", "dangling_proclitic"
        )
    return GuardResult(False)


@lru_cache(maxsize=1)
def _camel_analyzer():
    """Lazily load camel-tools' dictionary Analyzer, or None if unavailable.

    Morphology (VSO / iḍāfa) guards are optional: missing package or DB must
    not crash the live path. Lexicon-only guards still run. The neural
    disambiguator is intentionally not used — too slow for the live hot loop.
    """
    try:
        from camel_tools.morphology.analyzer import Analyzer
        from camel_tools.morphology.database import MorphologyDB
    except ImportError:
        logger.warning(
            "camel-tools is not installed; VSO/iḍāfa guards disabled. %s",
            _CAMEL_INSTALL_HINT,
        )
        return None

    try:
        db = MorphologyDB.builtin_db(db_name=_CAMEL_DB_NAME, flags="a")
    except FileNotFoundError:
        logger.warning(
            "camel-tools morphology database '%s' is missing; "
            "VSO/iḍāfa guards disabled. %s",
            _CAMEL_DB_NAME,
            _CAMEL_INSTALL_HINT,
        )
        return None

    return Analyzer(db)


def camel_morphology_available() -> bool:
    """True when camel-tools and the dictionary morphology DB can load."""
    return _camel_analyzer() is not None


def _guard_vso_no_subject(last_word: str) -> GuardResult:
    """Arabic is VSO: a bare verb with no subject yet may still need one,
    and committing an English subject (or subjectless form) risks being
    wrong once the real subject arrives.
    """
    analyzer = _camel_analyzer()
    if analyzer is None:
        return GuardResult(False)
    analyses = analyzer.analyze(last_word)
    if any(a.get("pos") == "verb" for a in analyses):
        return GuardResult(
            True, f"'{last_word}' analyzes as a verb with no subject yet", "vso_no_subject"
        )
    return GuardResult(False)


def _guard_idafa_head(last_word: str) -> GuardResult:
    """A bare (non-definite) noun that can be read in construct state (stt
    'c') may be the head of an iḍāfa chain still awaiting its genitive
    noun - "بيت" alone vs "بيت الرجل" ("a house" vs "the man's house").
    Definite nouns (ال-prefixed) can't head an iḍāfa, so they're exempt.
    """
    if last_word.startswith(_DEFINITE_PREFIX):
        return GuardResult(False)
    analyzer = _camel_analyzer()
    if analyzer is None:
        return GuardResult(False)
    analyses = analyzer.analyze(last_word)
    if any(a.get("pos") == "noun" and a.get("stt") == "c" for a in analyses):
        return GuardResult(
            True, f"'{last_word}' may head an iḍāfa chain not yet complete", "idafa_head"
        )
    return GuardResult(False)


# Cheapest first: lexicon-only guards run before the morphology-backed ones.
_GUARDS = (
    _guard_tam_particle,
    _guard_partial_numeral,
    _guard_dangling_proclitic,
    _guard_vso_no_subject,
    _guard_idafa_head,
)


def check_structural_guards(text: str) -> GuardResult:
    """Check whether the tail of ``text`` sits in a hazardous Arabic
    construction that should hold back commitment. Only the last word is
    examined - it is the word nearest the commit boundary.
    """
    text = normalize_arabic(text)
    if not text:
        return GuardResult(False)

    last_word = text.split()[-1]
    for guard in _GUARDS:
        result = guard(last_word)
        if result.hold:
            return result
    return GuardResult(False)


__all__ = [
    "GuardResult",
    "camel_morphology_available",
    "check_structural_guards",
    "chunk_for_translation",
    "join_translations",
    "merge_incremental_text",
    "normalize_arabic",
    "reconcile_provisional",
    "split_sentences",
]
