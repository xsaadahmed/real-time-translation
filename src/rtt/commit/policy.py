"""Commit policy: README "Commitment algorithm" steps 4-6.

Given the current tick's candidate English prefixes (word-boundary
truncations of the best translation, shortest to longest), the structural
guard state, and the current lag, picks the longest prefix the risk model
predicts will survive at-or-above the lag-adjusted theta - or forces a
guard-safe fallback once lag has blown through the hard ceiling.

Deliberately independent of the live/harvest wiring: it only needs a
callable that turns an English prefix into the risk model's feature dict
(``feature_builder``), so it can be driven by LiveSessionStore, the offline
harvester, or a unit test with synthetic records, without depending on any
of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..risk import RiskModel
from ..text import GuardResult
from .lag_governor import LagGovernor


@dataclass
class CommitDecision:
    """One tick's outcome. ``committed_text`` is ``""`` for WAIT."""

    committed_text: str
    theta: float
    forced: bool
    reason: str


FeatureBuilder = Callable[[str], dict]


class CommitPolicy:
    """Stateful across ticks of one session: remembers the last guard-safe
    boundary so a hard-ceiling force-commit has somewhere safe to land even
    when the current tick's guard is holding.
    """

    def __init__(self, risk_model: RiskModel, lag_governor: LagGovernor | None = None) -> None:
        self.risk_model = risk_model
        self.lag_governor = lag_governor or LagGovernor()
        self._last_guard_safe_text = ""

    def reset(self) -> None:
        """Call between sessions/utterances - lag state and the remembered
        guard-safe boundary don't carry over.
        """
        self.lag_governor.reset()
        self._last_guard_safe_text = ""

    def decide(
        self,
        candidate_prefixes: list[str],
        guard_result: GuardResult,
        lag_sec: float,
        feature_builder: FeatureBuilder,
    ) -> CommitDecision:
        """``candidate_prefixes`` must be sorted shortest to longest (word
        prefixes of the current tick's best translation) - the longest
        entry is what "guard-safe boundary" falls back to.
        """
        theta = self.lag_governor.update(lag_sec)
        force = self.lag_governor.force_commit(lag_sec)

        if not guard_result.hold and candidate_prefixes:
            self._last_guard_safe_text = candidate_prefixes[-1]

        best = ""
        if not guard_result.hold:
            for prefix in candidate_prefixes:
                proba = self.risk_model.predict_survival_proba(feature_builder(prefix))
                if proba >= theta:
                    best = prefix  # ascending order: last pass is the longest pass

        if best:
            return CommitDecision(best, theta, forced=False, reason="")

        if force:
            # Nearest guard-safe boundary: this tick's if the guard just
            # cleared it, otherwise the last tick where it did.
            fallback = candidate_prefixes[-1] if not guard_result.hold and candidate_prefixes else self._last_guard_safe_text
            if fallback:
                return CommitDecision(fallback, theta, forced=True, reason="lag_hard_ceiling")
            return CommitDecision("", theta, forced=True, reason="lag_hard_ceiling_no_safe_boundary")

        reason = "guard_hold" if guard_result.hold else "below_threshold"
        return CommitDecision("", theta, forced=False, reason=reason)


__all__ = ["CommitDecision", "CommitPolicy", "FeatureBuilder"]
