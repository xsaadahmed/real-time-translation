"""Compare and log the cascade's output against the second opinion's.

This is data collection, not a decision: step 7 ("harvest retrospective
labels") and the eventual risk model are what turn these comparisons into a
signal the commit policy can use. For now this just makes the disagreement
data real and inspectable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class SecondOpinionRecord:
    """One comparison between the cascade's English output and the
    independent second opinion's, for the same span of Arabic audio.
    """

    cascade_text: str
    second_opinion_text: str
    similarity: float
    agreed_prefix: str
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return asdict(self)


def _common_word_prefix(a: str, b: str) -> str:
    a_words, b_words = a.split(), b.split()
    agreed: list[str] = []
    for wa, wb in zip(a_words, b_words):
        if wa != wb:
            break
        agreed.append(wa)
    return " ".join(agreed)


def compare(
    cascade_text: str,
    second_opinion_text: str,
    session_id: str = "",
) -> SecondOpinionRecord:
    """Compare two independent English translations of the same Arabic audio.

    ``similarity`` is a whole-string ratio (difflib), since two independently
    phrased translations rarely agree word-for-word from the start even when
    both are correct - unlike agreement_depth's branch voting, where all
    branches share the same translator and a shared KV prefix, so an exact
    leading-word match is the meaningful signal there. ``agreed_prefix`` is
    kept too since an exact leading match, when it does happen, is a strong
    signal in its own right.
    """
    cascade_text = cascade_text.strip()
    second_opinion_text = second_opinion_text.strip()
    similarity = SequenceMatcher(None, cascade_text, second_opinion_text).ratio()
    agreed_prefix = _common_word_prefix(cascade_text, second_opinion_text)
    return SecondOpinionRecord(
        cascade_text=cascade_text,
        second_opinion_text=second_opinion_text,
        similarity=similarity,
        agreed_prefix=agreed_prefix,
        session_id=session_id,
    )


def log_record(record: SecondOpinionRecord, log_path: str | Path) -> None:
    """Append one comparison as a JSON line, creating parent dirs as needed."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


__all__ = ["SecondOpinionRecord", "compare", "log_record"]
