"""Turn a harvested CommitCandidateRecord into the risk model's feature vector.

README lists the intended signal set: branch agreement depth, target
divergence, guard state, ASR posterior, Seamless agreement, current lag,
tokens since last commit. Step 7's harvester only populates a subset so far
(no futures drafter yet, so agreement_depth is always None; no ASR posterior
plumbed through faster-whisper; no live "current lag" concept in the offline
replay) - those show up below as an explicit availability flag plus a
sentinel value, so the model can still learn from the signals that exist
without a real number silently standing in for "unknown". Extend this
function (not the model) once step 4's branch drafter and ASR posteriors
are wired into the harvester.
"""

from __future__ import annotations

# Guard names check_structural_guards can produce - see text.py's _GUARDS.
# Fixed order so one-hot columns stay stable across a training run.
_GUARD_NAMES = (
    "tam_particle",
    "partial_numeral",
    "dangling_proclitic",
    "vso_no_subject",
    "idafa_head",
)

_SENTINEL = -1.0

FEATURE_NAMES = (
    "guard_hold",
    *[f"guard_is_{name}" for name in _GUARD_NAMES],
    "agreement_depth",
    "has_agreement_depth",
    "branch_count",
    "second_opinion_similarity",
    "has_second_opinion",
    "tick",
    "verified_len_words",
    "provisional_len_words",
    "candidate_len_words",
)


def record_to_features(record: dict) -> dict[str, float]:
    """Map one harvested record (dict, as loaded from JSONL) to named features.

    Deliberately excludes ``survival_score`` / ``exact_prefix_match`` /
    ``final_english`` - those are derived from the label itself and would
    leak it straight into the model.
    """
    guard_name = record.get("guard_name") or ""
    agreement_depth = record.get("agreement_depth")
    second_opinion_similarity = record.get("second_opinion_similarity")

    features: dict[str, float] = {
        "guard_hold": float(bool(record.get("guard_hold"))),
        "agreement_depth": float(agreement_depth) if agreement_depth is not None else _SENTINEL,
        "has_agreement_depth": float(agreement_depth is not None),
        "branch_count": float(record.get("branch_count") or 1),
        "second_opinion_similarity": (
            float(second_opinion_similarity) if second_opinion_similarity is not None else _SENTINEL
        ),
        "has_second_opinion": float(second_opinion_similarity is not None),
        "tick": float(record.get("tick") or 0),
        "verified_len_words": float(len((record.get("arabic_verified") or "").split())),
        "provisional_len_words": float(len((record.get("arabic_provisional") or "").split())),
        "candidate_len_words": float(len((record.get("english_candidate") or "").split())),
    }
    for name in _GUARD_NAMES:
        features[f"guard_is_{name}"] = float(guard_name == name)

    return features


__all__ = ["FEATURE_NAMES", "record_to_features"]
