"""Tests for Translator.translate_branches and agreement_depth (step 5).

Uses a stub Translator so this stays a fast, hermetic unit test - no model
download or CPU inference. Validating the branching/agreement mechanism
itself against a real backend is the point of step 4(b): plug in
HuggingFaceTranslator here once its weights are cached locally.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.mt.agreement import agreement_depth, default_quorum
from rtt.mt.base import Translator


class StubTranslator(Translator):
    """Deterministic word-for-word 'translation': Arabic text back verbatim
    with a fixed English gloss per Arabic word, so branch outputs are
    predictable and easy to assert on.
    """

    GLOSS = {
        "ذهب": "went",
        "الولد": "the-boy",
        "الى": "to",
        "المدرسة": "school",
        "امس": "yesterday",
        "اليوم": "today",
    }

    def load(self) -> None:
        pass

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [
            " ".join(self.GLOSS.get(word, word) for word in text.split())
            for text in texts
        ]


def test_translate_branches_returns_k_plus_one_outputs():
    t = StubTranslator()
    branches = t.translate_branches("ذهب الولد", ["الى المدرسة", "امس", "اليوم"])
    assert len(branches) == 4
    assert branches[0] == "went the-boy"
    assert branches[1] == "went the-boy to school"
    assert branches[2] == "went the-boy yesterday"
    assert branches[3] == "went the-boy today"


def test_translate_branches_empty_future_falls_back_to_observed():
    t = StubTranslator()
    branches = t.translate_branches("ذهب الولد", [""])
    assert branches == ["went the-boy", "went the-boy"]


def test_default_quorum_matches_seven_of_nine():
    assert default_quorum(9) == 7


def test_agreement_depth_full_agreement():
    branches = ["went the boy to school"] * 9
    result = agreement_depth(branches)
    assert result.prefix == "went the boy to school"
    assert result.depth == 5
    assert result.quorum == 7


def test_agreement_depth_stops_at_first_divergence():
    branches = (
        ["went the boy to school"] * 7
        + ["went the girl to the park"]
        + ["ran the dog home"]
    )
    result = agreement_depth(branches)
    # 7 of 9 agree on "went the boy to school" (position 0-1 "went the"
    # from all but one branch; position 2 "boy" from exactly 7 of 9).
    assert result.prefix == "went the boy to school"
    assert result.depth == 5


def test_agreement_depth_stops_when_quorum_word_not_reached():
    branches = (
        ["went the boy to school"] * 6
        + ["went the girl to the park"]
        + ["ran the dog home"]
        + ["flew the cat away"]
    )
    result = agreement_depth(branches)
    # Position 0 "went" reaches 7/9; position 1 "the" also reaches 7/9;
    # position 2 splits boy/girl/dog/cat - no word reaches the 7-vote quorum.
    assert result.prefix == "went the"
    assert result.depth == 2


def test_agreement_depth_below_quorum_is_empty():
    branches = ["went home"] * 3 + ["ran away"] * 6
    result = agreement_depth(branches, quorum=7)
    assert result.depth == 0
    assert result.prefix == ""


def test_agreement_depth_empty_branches():
    result = agreement_depth([])
    assert result.depth == 0
    assert result.branch_count == 0
