"""Tests for src/rtt/text.py structural guards.

Lexicon-only guards always run. Morphology (VSO / iḍāfa) cases need the
optional camel-tools step — see requirements-guards.txt.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.text import GuardResult, camel_morphology_available, check_structural_guards

# (arabic fragment, expect_hold, guard_name or None, description)
LEXICON_CASES = [
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
]

MORPHOLOGY_CASES = [
    # --- VSO without subject: needs CAMeL POS ---
    # Dictionary-only analysis has no diacritics to disambiguate person, so
    # even pro-drop verbs can't be told apart from bare 3rd-person verbs;
    # every bare verb reading is conservatively held.
    ("قفز", True, "vso_no_subject", "bare verb reading possible, no subject yet (VSO)"),
    ("قفز الولد", False, None, "verb followed by its subject, no longer ambiguous"),
    # --- iḍāfa chain: needs CAMeL morphology ---
    ("رأيت منزل", True, "idafa_head", "bare indefinite noun may head an iḍāfa awaiting its genitive"),
    ("رأيت المنزل", False, None, "definite noun (ال-) cannot head an iḍāfa"),
]


def _assert_guard(fragment, expect_hold, expect_guard, description):
    result = check_structural_guards(fragment)
    assert result.hold == expect_hold, (
        f"{description}: expected hold={expect_hold} for {fragment!r}, "
        f"got hold={result.hold} reason={result.reason!r}"
    )
    if expect_hold:
        assert result.guard == expect_guard, (
            f"{description}: expected guard {expect_guard!r}, got {result.guard!r}"
        )


@pytest.mark.parametrize("fragment,expect_hold,expect_guard,description", LEXICON_CASES)
def test_lexicon_guard_label(fragment, expect_hold, expect_guard, description):
    _assert_guard(fragment, expect_hold, expect_guard, description)


@pytest.mark.skipif(
    not camel_morphology_available(),
    reason="requires: pip install -r requirements-guards.txt && "
    "python scripts/download_models.py --camel-data-only",
)
@pytest.mark.parametrize("fragment,expect_hold,expect_guard,description", MORPHOLOGY_CASES)
def test_morphology_guard_label(fragment, expect_hold, expect_guard, description):
    _assert_guard(fragment, expect_hold, expect_guard, description)


def test_empty_text_never_holds():
    result = check_structural_guards("")
    assert result == GuardResult(False)
