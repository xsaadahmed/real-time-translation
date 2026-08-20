"""Branch-agreement scoring (README "Anticipation mechanism", Axis 1).

Standalone from any particular Translator backend: takes the K+1 English
strings ``Translator.translate_branches`` returns and scores how much of
them agree. This is deliberately just the scoring primitive, not the full
commit policy (risk model, guards, lag governor) - those depend on signals
(Seamless agreement, temporal survival) that don't exist yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AgreementResult:
    """Longest English word-prefix shared by at least ``quorum`` branches."""

    prefix: str
    depth: int
    quorum: int
    branch_count: int


def default_quorum(branch_count: int) -> int:
    """7-of-9 in the target design (1 observed-only + K=8 futures branches),
    generalized to other branch counts as ceil(7/9 * branch_count).
    """
    if branch_count <= 0:
        return 0
    return max(1, math.ceil(7 * branch_count / 9))


def agreement_depth(branches: list[str], quorum: int | None = None) -> AgreementResult:
    """Score future-marginal invariance: the longest English prefix that
    appears, word for word from the start, in at least ``quorum`` of
    ``branches``. Depth (a word count) is a continuous confidence signal,
    not a binary flag - the more branches agree and the longer the agreed
    prefix, the safer that prefix is to consider for commitment.
    """
    branch_count = len(branches)
    if quorum is None:
        quorum = default_quorum(branch_count)

    if branch_count == 0:
        return AgreementResult("", 0, quorum, 0)

    word_lists = [b.split() for b in branches]
    max_len = max((len(words) for words in word_lists), default=0)

    agreed_words: list[str] = []
    for position in range(max_len):
        counts: dict[str, int] = {}
        for words in word_lists:
            if position < len(words):
                word = words[position]
                counts[word] = counts.get(word, 0) + 1
        if not counts:
            break
        best_word, best_count = max(counts.items(), key=lambda kv: kv[1])
        if best_count < quorum:
            break
        agreed_words.append(best_word)

    return AgreementResult(" ".join(agreed_words), len(agreed_words), quorum, branch_count)


__all__ = ["AgreementResult", "agreement_depth", "default_quorum"]
