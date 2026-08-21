"""Tests for src/rtt/text.py, especially the verified/provisional split."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.text import merge_incremental_text, reconcile_provisional


def test_reconcile_first_pass_all_provisional():
    verified, provisional = reconcile_provisional("", "ذهبت الى")
    assert verified == ""
    assert provisional == "ذهبت الى"


def test_reconcile_full_agreement_verifies_everything():
    verified, provisional = reconcile_provisional("ذهبت الى", "ذهبت الى المدرسة")
    assert verified == "ذهبت الى"
    assert provisional == "المدرسة"


def test_reconcile_disagreement_stops_verification_at_first_diff():
    # Second pass revises "كتاب" -> "كتابه" (a book -> his book): must not
    # verify the partially-agreeing word, and the tail is replaced outright.
    verified, provisional = reconcile_provisional("قرأت كتاب", "قرأت كتابه اليوم")
    assert verified == "قرأت"
    assert provisional == "كتابه اليوم"


def test_reconcile_silence_keeps_previous_guess():
    verified, provisional = reconcile_provisional("ذهبت الى المدرسة", "")
    assert verified == ""
    assert provisional == "ذهبت الى المدرسة"


def test_reconcile_no_prior_agreement_new_hypothesis_fully_provisional():
    verified, provisional = reconcile_provisional("مرحبا", "وداعا")
    assert verified == ""
    assert provisional == "وداعا"


def test_merge_incremental_text_still_append_only():
    assert merge_incremental_text("ذهبت الى", "الى المدرسة") == "ذهبت الى المدرسة"
    assert merge_incremental_text("", "مرحبا") == "مرحبا"
    assert merge_incremental_text("مرحبا", "") == "مرحبا"


def test_reconcile_realigns_when_new_pass_starts_mid_provisional():
    """The common live case: the fresh pass begins inside the old provisional.

    Each pass replays LIVE_CONTEXT_SEC of already-seen audio, so the new
    hypothesis starts partway through the previous tail. Compared from index 0
    these look completely different, which used to commit nothing at all.
    """
    old = "جزء أساسي من الهوية في بيروت تسمع العربية والفرنسية"
    new = "تسمع العربية والفرنسية في الشارع نفسه"

    verified, provisional = reconcile_provisional(old, new)

    # Everything before the overlap has aged out of the window and is kept.
    assert verified.startswith("جزء أساسي من الهوية في بيروت")
    # The overlapping run both passes agree on is committed too.
    assert "تسمع العربية والفرنسية" in verified
    assert provisional == "في الشارع نفسه"


def test_reconcile_tolerates_revised_wording_in_the_overlap():
    """ASR rewords between passes; an exact-match search finds no overlap."""
    old = "وأتحدث لغة العربية منزل التفولة اليوم"
    new = "لغة العربية منذ الطفولة اليوم وغدا"

    verified, provisional = reconcile_provisional(old, new)

    # Aligned despite منزل/منذ and التفولة/الطفولة differing.
    assert verified.startswith("وأتحدث")
    assert provisional.endswith("وغدا")
    assert "وغدا" not in verified


def test_reconcile_commits_nothing_when_passes_share_nothing():
    verified, provisional = reconcile_provisional(
        "ذهبت الى المدرسة صباحا", "الطقس جميل جدا اليوم"
    )
    assert verified == ""
    assert provisional == "الطقس جميل جدا اليوم"


def test_reconcile_never_loses_words_across_a_sequence_of_passes():
    """End to end: every word spoken should end up committed or still pending."""
    passes = [
        "مرحبا أنا من لبنان",
        "أنا من لبنان وأتحدث العربية",
        "وأتحدث العربية منذ الطفولة",
        "منذ الطفولة في بيروت",
    ]
    verified_total: list[str] = []
    provisional = ""
    for hypothesis in passes:
        newly_verified, provisional = reconcile_provisional(provisional, hypothesis)
        if newly_verified:
            verified_total.append(newly_verified)

    final = merge_incremental_text(" ".join(verified_total), provisional)
    for word in ("مرحبا", "لبنان", "وأتحدث", "العربية", "الطفولة", "بيروت"):
        assert word in final, f"lost {word!r}"
