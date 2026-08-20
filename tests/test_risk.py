"""Tests for the commit-risk model (step 8).

Uses synthetic records rather than a real harvest, so the model's plumbing
(feature extraction, fit, calibrated predict, save/load round trip) is
validated independently of how much real Arabic audio has been harvested.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.risk import FEATURE_NAMES, RiskModel, record_to_features
from rtt.risk.model import _to_matrix


def _record(session_id="s1", tick=0, guard_hold=False, guard_name="",
            candidate_words=3, survived=True, agreement_depth=None,
            second_opinion_similarity=None):
    return {
        "session_id": session_id,
        "tick": tick,
        "arabic_verified": "كلمة " * 2,
        "arabic_provisional": "كلمة",
        "english_candidate": " ".join(["word"] * candidate_words),
        "guard_hold": guard_hold,
        "guard_name": guard_name,
        "agreement_depth": agreement_depth,
        "branch_count": 1,
        "second_opinion_similarity": second_opinion_similarity,
        "survived": survived,
    }


def test_record_to_features_has_all_expected_keys():
    features = record_to_features(_record())
    assert set(features.keys()) == set(FEATURE_NAMES)


def test_record_to_features_sentinel_for_missing_signals():
    features = record_to_features(_record(agreement_depth=None, second_opinion_similarity=None))
    assert features["has_agreement_depth"] == 0.0
    assert features["has_second_opinion"] == 0.0
    assert features["agreement_depth"] == -1.0


def test_record_to_features_guard_one_hot():
    features = record_to_features(_record(guard_hold=True, guard_name="tam_particle"))
    assert features["guard_hold"] == 1.0
    assert features["guard_is_tam_particle"] == 1.0
    assert features["guard_is_idafa_head"] == 0.0


def _synthetic_dataset(n_sessions=6, per_session=20):
    """Guard-holding candidates rarely survive; short non-held ones usually
    do - a signal strong enough for a tiny GradientBoosting model to learn
    from a small synthetic set, unlike real harvested data.
    """
    records = []
    for s in range(n_sessions):
        for t in range(per_session):
            guard_hold = t % 4 == 0
            survived = (not guard_hold) and (t % 3 != 0)
            records.append(
                _record(
                    session_id=f"session-{s}",
                    tick=t,
                    guard_hold=guard_hold,
                    guard_name="tam_particle" if guard_hold else "",
                    candidate_words=(t % 5) + 1,
                    survived=survived,
                )
            )
    return records


def test_risk_model_fit_and_predict_returns_probability_in_range():
    records = _synthetic_dataset()
    model = RiskModel().fit(records)
    for record in records[:10]:
        proba = model.predict_survival_proba(record)
        assert 0.0 <= proba <= 1.0


def test_risk_model_separates_guard_hold_from_clean_candidates():
    records = _synthetic_dataset()
    model = RiskModel().fit(records)

    held = _record(guard_hold=True, guard_name="tam_particle", candidate_words=1)
    clean = _record(guard_hold=False, candidate_words=1)

    assert model.predict_survival_proba(clean) > model.predict_survival_proba(held)


def test_risk_model_save_and_load_round_trip(tmp_path):
    records = _synthetic_dataset()
    model = RiskModel().fit(records)
    path = tmp_path / "risk_model.joblib"
    model.save(path)

    loaded = RiskModel.load(path)
    sample = records[0]
    assert loaded.predict_survival_proba(sample) == model.predict_survival_proba(sample)


def test_predict_before_fit_raises():
    model = RiskModel()
    try:
        model.predict_survival_proba(_record())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unfitted model")


def test_to_matrix_matches_feature_order():
    records = [_record()]
    matrix = _to_matrix(records)
    expected = record_to_features(records[0])
    assert matrix.shape == (1, len(FEATURE_NAMES))
    for i, name in enumerate(FEATURE_NAMES):
        assert matrix[0, i] == expected[name]
