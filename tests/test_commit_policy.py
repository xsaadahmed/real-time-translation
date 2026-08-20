"""Tests for the commit policy and lag governor (step 9)."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.commit import CommitPolicy, LagGovernor, LagGovernorConfig
from rtt.risk import RiskModel
from rtt.text import GuardResult

NO_HOLD = GuardResult(False)
HOLD = GuardResult(True, "reason", "tam_particle")


# ---------------------------------------------------------------------------
# LagGovernor
# ---------------------------------------------------------------------------


def test_lag_governor_starts_at_base_theta():
    governor = LagGovernor()
    assert governor.theta == 0.97


def test_lag_governor_decays_one_rung_per_call_above_high_lag():
    governor = LagGovernor()
    assert governor.update(3.1) == 0.90
    assert governor.update(3.1) == 0.80
    # Already at the floor - stays there.
    assert governor.update(3.1) == 0.80


def test_lag_governor_raises_one_rung_per_call_below_low_lag():
    governor = LagGovernor()
    governor.update(3.1)
    governor.update(3.1)
    assert governor.theta == 0.80
    assert governor.update(1.0) == 0.90
    assert governor.update(1.0) == 0.97
    # Already at the ceiling - stays there.
    assert governor.update(1.0) == 0.97


def test_lag_governor_holds_steady_in_the_middle_band():
    governor = LagGovernor()
    assert governor.update(2.0) == 0.97
    governor.update(3.1)
    assert governor.update(2.0) == 0.90


def test_lag_governor_force_commit_at_hard_ceiling():
    governor = LagGovernor()
    assert governor.force_commit(3.4) is False
    assert governor.force_commit(3.5) is True
    assert governor.force_commit(4.0) is True


def test_lag_governor_reset():
    governor = LagGovernor()
    governor.update(3.1)
    governor.reset()
    assert governor.theta == 0.97


def test_lag_governor_config_rejects_non_monotonic_steps():
    with pytest.raises(ValueError):
        LagGovernorConfig(decay_steps=(0.97, 0.95, 0.98))


def test_lag_governor_config_rejects_mismatched_base_theta():
    with pytest.raises(ValueError):
        LagGovernorConfig(base_theta=0.99, decay_steps=(0.97, 0.90, 0.80))


# ---------------------------------------------------------------------------
# CommitPolicy
# ---------------------------------------------------------------------------


def _record(text: str, guard_hold: bool = False) -> dict:
    return {
        "session_id": "s",
        "tick": 0,
        "arabic_verified": "",
        "arabic_provisional": "",
        "english_candidate": text,
        "guard_hold": guard_hold,
        "guard_name": "",
        "agreement_depth": None,
        "branch_count": 1,
        "second_opinion_similarity": None,
        "survived": len(text.split()) <= 2,  # short prefixes "survive", long ones don't
    }


def _fitted_risk_model() -> RiskModel:
    records = []
    for i in range(40):
        short = i % 2 == 0
        text = "a b" if short else "a b c d e"
        records.append(_record(text))
    return RiskModel().fit(records)


def _prefixes(text: str) -> list[str]:
    words = text.split()
    return [" ".join(words[: n + 1]) for n in range(len(words))]


def test_commit_policy_picks_longest_prefix_clearing_threshold():
    model = _fitted_risk_model()
    policy = CommitPolicy(model, LagGovernor())

    decision = policy.decide(
        candidate_prefixes=_prefixes("a b c d e"),
        guard_result=NO_HOLD,
        lag_sec=2.0,
        feature_builder=lambda text: _record(text),
    )
    assert decision.committed_text != ""
    assert decision.forced is False
    # Should not commit the full 5-word (low-survival) candidate.
    assert decision.committed_text.split() != "a b c d e".split()


def test_commit_policy_waits_on_guard_hold_without_forcing():
    model = _fitted_risk_model()
    policy = CommitPolicy(model, LagGovernor())

    decision = policy.decide(
        candidate_prefixes=_prefixes("a b"),
        guard_result=HOLD,
        lag_sec=1.0,
        feature_builder=lambda text: _record(text),
    )
    assert decision.committed_text == ""
    assert decision.forced is False
    assert decision.reason == "guard_hold"


def test_commit_policy_force_commits_at_hard_ceiling_using_last_safe_boundary():
    model = _fitted_risk_model()
    policy = CommitPolicy(model, LagGovernor())

    # First tick: guard clear, establishes a safe boundary.
    policy.decide(
        candidate_prefixes=_prefixes("a b"),
        guard_result=NO_HOLD,
        lag_sec=2.0,
        feature_builder=lambda text: _record(text),
    )
    # Second tick: guard now holds, but lag has blown the hard ceiling -
    # must still emit something, falling back to the earlier safe boundary.
    decision = policy.decide(
        candidate_prefixes=_prefixes("a b c"),
        guard_result=HOLD,
        lag_sec=4.0,
        feature_builder=lambda text: _record(text),
    )
    assert decision.forced is True
    assert decision.committed_text == "a b"
    assert decision.reason == "lag_hard_ceiling"


def test_commit_policy_force_commit_with_no_safe_boundary_yet_returns_wait():
    model = _fitted_risk_model()
    policy = CommitPolicy(model, LagGovernor())

    decision = policy.decide(
        candidate_prefixes=_prefixes("a b"),
        guard_result=HOLD,
        lag_sec=4.0,
        feature_builder=lambda text: _record(text),
    )
    assert decision.committed_text == ""
    assert decision.forced is True
    assert decision.reason == "lag_hard_ceiling_no_safe_boundary"


def test_commit_policy_reset_clears_lag_and_safe_boundary():
    model = _fitted_risk_model()
    policy = CommitPolicy(model, LagGovernor())
    policy.decide(
        candidate_prefixes=_prefixes("a b"),
        guard_result=NO_HOLD,
        lag_sec=3.1,
        feature_builder=lambda text: _record(text),
    )
    policy.reset()
    assert policy.lag_governor.theta == 0.97
    assert policy._last_guard_safe_text == ""
