"""Data model for retrospective commit-candidate labels (step 7).

README "Learned, calibrated risk head": run the pipeline offline over hours
of Arabic speech, and for every prefix *considered* for commit, retrospectively
label whether it survived once the full utterance was decoded. Step 8's risk
model trains on exactly these records.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
    survived: bool | None = None
    final_english: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def label_survival(candidate: str, final_english: str) -> bool:
    """Did this candidate prefix survive to the final decoded sentence?

    A prefix "survives" if it is exactly the leading words of the ground
    truth - matching README's definition (a retrospective label on prefixes
    considered vs prefixes that actually survived), not a fuzzy similarity.
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


__all__ = ["CommitCandidateRecord", "label_survival", "log_records"]
