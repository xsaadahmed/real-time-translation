"""Tests for src/rtt/text.py structural guards.

Each case is a small Arabic fragment with a known correct WAIT/OK label,
building the seed set the plan calls for reusing later when harvesting
retrospective commit labels.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.text import GuardResult, check_structural_guards

# (arabic fragment, expect_hold, guard_name or None, description)
CASES = [
    # --- TAM particles: closed lexicon, no dependency ---
    ("ذهب الولد الى المدرسة", False, None, "complete simple past sentence"),
    ("كان", True, "tam_particle", "bare TAM particle, verb not yet heard"),
    ("الطالب لم", True, "tam_particle", "negation particle awaiting the verb it negates"),
    ("سوف", True, "tam_particle", "future particle awaiting its verb"),
    # --- Numerals: ones-then-tens, no dependency ---
    ("قرأت واحد", True, "partial_numeral", "bare ones-digit that may combine with a following tens word"),
    ("قرأت واحد وعشرون كتابا اليوم", False, None, "ones+tens numeral already fully combined, tail moved past it"),
    ("عندي خمسة", True, "partial_numeral", "bare ones-digit could still be the start of a compound number"),
    # --- Dangling proclitics: lexicon/pattern, no dependency ---
    ("ذهبت الى المدرسة و", True, "dangling_proclitic", "standalone 'and' awaiting its host word"),
    ("ذهبت و عاد الاولاد", False, None, "'و' already attached to a following word, tail has moved past it"),
    # --- VSO without subject: needs CAMeL POS ---
    # Dictionary-only analysis has no diacritics to disambiguate person, so
    # even pro-drop verbs (subject encoded in the verb, README's "free
    # latency" case) can't be told apart from bare 3rd-person verbs missing
    # their subject; every bare verb reading is conservatively held. That
    # precision only arrives with the neural disambiguator, out of scope for
    # the 240ms hot loop - see README "Install Arabic NLP tooling".
    ("قفز", True, "vso_no_subject", "bare verb reading possible, no subject yet (VSO)"),
    ("قفز الولد", False, None, "verb followed by its subject, no longer ambiguous"),
    # --- iḍāfa chain: needs CAMeL morphology ---
    ("رأيت منزل", True, "idafa_head", "bare indefinite noun may head an iḍāfa awaiting its genitive"),
    ("رأيت المنزل", False, None, "definite noun (ال-) cannot head an iḍāfa"),
]


@pytest.mark.parametrize("fragment,expect_hold,expect_guard,description", CASES)
def test_guard_label(fragment, expect_hold, expect_guard, description):
    result = check_structural_guards(fragment)
    assert result.hold == expect_hold, (
        f"{description}: expected hold={expect_hold} for {fragment!r}, "
        f"got hold={result.hold} reason={result.reason!r}"
    )
    if expect_hold:
        assert result.guard == expect_guard, (
            f"{description}: expected guard {expect_guard!r}, got {result.guard!r}"
        )


def test_empty_text_never_holds():
    result = check_structural_guards("")
    assert result == GuardResult(False)
