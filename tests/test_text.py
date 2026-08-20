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
