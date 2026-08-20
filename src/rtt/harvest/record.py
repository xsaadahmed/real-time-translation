"""Data model for retrospective commit-candidate labels (step 7).

README "Learned, calibrated risk head": run the pipeline offline over hours
of Arabic speech, and for every prefix *considered* for commit, retrospectively
label whether it survived once the full utterance was decoded. Step 8's risk
model trains on exactly these records.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

#: Below this word-level similarity to the same-length leading window of the
#: final sentence, a candidate is labeled as not having survived. Chosen
#: empirically: exact-prefix matching (see exact_prefix_match) turned out to
#: label essentially everything False, because two independent ASR+MT passes
#: over similar-but-not-identical audio windows rarely produce byte-identical
#: English phrasing even when both are semantically correct - see the step 7
#: harvest run notes. 0.6 is a starting point, not a validated cutoff.
SURVIVAL_THRESHOLD = 0.6

_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class CommitCandidateRecord:
    """One candidate English prefix considered for commit at one tick of one
    utterance, with the signals available at that moment and (once the
    utterance finishes decoding) whether it survived to the final sentence.
    """

    session_id: str
    tick: int
    arabic_verified: str
    arabic_provisional: str
    english_candidate: str

    # Guard state (text.check_structural_guards on the provisional tail).
    guard_hold: bool
    guard_name: str = ""
    guard_reason: str = ""

    # Branch-agreement signal (mt.agreement_depth). branch_count is 1 until
    # the Arabic futures drafter (README's 0.5B sampler) exists - the
    # observed-only branch is trivially "agreed" with itself, so treat
    # agreement_depth as unavailable (None) rather than a misleading number.
    agreement_depth: int | None = None
    agreement_prefix: str | None = None
    branch_count: int = 1

    # Second-opinion signal (second_opinion.compare), optional - only
    # populated when that channel is enabled.
    second_opinion_similarity: float | None = None

    timestamp: float = field(default_factory=time.time)

    # Filled in retrospectively once the utterance's ground truth is known.
    # survived is the fuzzy (survival_score >= SURVIVAL_THRESHOLD) label -
    # the one step 8 should train on. exact_prefix_match is the original,
    # much stricter word-for-word definition, kept as an auxiliary feature
    # since it's a stronger (if rarer) positive signal when it does fire.
    survived: bool | None = None
    survival_score: float | None = None
    exact_prefix_match: bool | None = None
    final_english: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def survival_score(candidate: str, final_english: str) -> float:
    """Word-level similarity between a candidate and the same-length leading
    window of the final sentence (not the whole sentence - that would reward
    short, vague candidates that happen to overlap something anywhere).

    Uses SequenceMatcher over word tokens rather than characters, so minor
    paraphrase differences (dropped articles, case, punctuation) don't
    tank the score the way an exact match would, while unrelated wording
    still scores low. 0.0 for an empty candidate - there's nothing to have
    survived.
    """
    candidate_words = _words(candidate)
    if not candidate_words:
        return 0.0
    window = _words(final_english)[: len(candidate_words)]
    if not window:
        return 0.0
    return SequenceMatcher(None, candidate_words, window).ratio()


def label_survival(candidate: str, final_english: str, threshold: float = SURVIVAL_THRESHOLD) -> bool:
    """Did this candidate prefix survive to the final decoded sentence?

    Fuzzy: survival_score(candidate, final_english) >= threshold. See
    SURVIVAL_THRESHOLD's docstring for why this isn't an exact match.
    """
    return survival_score(candidate, final_english) >= threshold


def exact_prefix_match(candidate: str, final_english: str) -> bool:
    """Strict version: True only if candidate is exactly the leading words
    of final_english, word for word. Kept as an auxiliary signal - see
    CommitCandidateRecord.exact_prefix_match.
    """
    candidate_words = candidate.split()
    final_words = final_english.split()
    if len(candidate_words) > len(final_words):
        return False
    return candidate_words == final_words[: len(candidate_words)]


def log_records(records: list[CommitCandidateRecord], log_path: str | Path) -> None:
    """Append records as JSON lines, creating parent dirs as needed."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


__all__ = [
    "SURVIVAL_THRESHOLD",
    "CommitCandidateRecord",
    "exact_prefix_match",
    "label_survival",
    "log_records",
    "survival_score",
]
