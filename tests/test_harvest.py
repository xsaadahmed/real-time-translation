"""Tests for offline retrospective-label harvesting (step 7).

Uses stub ASR/Translator backends driven by a synthetic "audio" array whose
values encode elapsed sample position (not real audio), so transcription
content is a deterministic function of how much audio the stub has "seen" -
letting the tick-simulation and reconcile/guard/label logic be validated
without any real Arabic speech.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.asr.base import ASREngine, Transcript
from rtt.harvest.harvester import harvest_utterance
from rtt.harvest.record import exact_prefix_match, label_survival, log_records, survival_score
from rtt.mt.base import Translator

SAMPLE_RATE = 16_000


class ScheduledASR(ASREngine):
    """Returns a growing Arabic transcript keyed on the last sample's
    encoded position - simulates more of the utterance becoming audible
    as (fake) time passes, without needing real audio.
    """

    def __init__(self, schedule: list[tuple[float, str]]) -> None:
        self.schedule = sorted(schedule)

    def load(self) -> None:
        pass

    def transcribe(self, audio, sample_rate: int) -> Transcript:
        if audio.size == 0:
            return Transcript(text="")
        position = float(audio[-1])
        text = ""
        for threshold, candidate in self.schedule:
            if position >= threshold:
                text = candidate
        return Transcript(text=text)


class GlossTranslator(Translator):
    GLOSS = {
        "ذهب": "went",
        "الولد": "the-boy",
        "الى": "to",
        "المدرسة": "school",
    }

    def load(self) -> None:
        pass

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [
            " ".join(self.GLOSS.get(word, word) for word in text.split())
            for text in texts
        ]


def _fake_audio(seconds: float) -> np.ndarray:
    """Every sample's value is its own index - lets ScheduledASR read off
    'how much audio has elapsed' without decoding anything."""
    n = int(seconds * SAMPLE_RATE)
    return np.arange(n, dtype=np.float32)


SCHEDULE = [
    (0.0, "ذهب"),
    (1.2 * SAMPLE_RATE, "ذهب الولد"),
    (2.4 * SAMPLE_RATE, "ذهب الولد الى المدرسة"),
]


def test_exact_prefix_match_is_strict():
    final = "went the-boy to school"
    assert exact_prefix_match("went the-boy", final) is True
    assert exact_prefix_match("went the-boy to school", final) is True
    assert exact_prefix_match("went the-girl", final) is False
    assert exact_prefix_match("went the-boy to school today", final) is False


def test_label_survival_is_fuzzy_and_tolerates_minor_drift():
    final = "went the-boy to school"
    # A genuine exact prefix still survives under the fuzzy metric.
    assert label_survival("went the-boy", final) is True
    # Completely unrelated wording does not.
    assert label_survival("flew away quickly", final) is False
    assert survival_score("flew away quickly", final) < survival_score("went the-boy", final)


def test_survival_score_is_zero_for_empty_candidate():
    assert survival_score("", "went the-boy to school") == 0.0


def test_harvest_utterance_produces_labeled_records():
    audio = _fake_audio(3.5)
    asr = ScheduledASR(SCHEDULE)
    translator = GlossTranslator()

    records = harvest_utterance(
        audio,
        SAMPLE_RATE,
        asr=asr,
        translator=translator,
        final_asr=asr,
        final_translator=translator,
        session_id="utt-1",
    )

    assert len(records) > 0
    ticks = [r.tick for r in records]
    assert ticks == sorted(ticks)  # monotonic

    # Every record must be labeled once the utterance has been harvested.
    for record in records:
        assert record.survived is not None
        assert record.final_english == "went the-boy to school"

    # The last record should have caught up to the full steady-state text.
    assert records[-1].arabic_verified + " " + records[-1].arabic_provisional
    assert records[-1].survived is True


def test_harvest_utterance_too_short_returns_no_records():
    audio = _fake_audio(0.05)  # below MIN_AUDIO_SEC
    asr = ScheduledASR(SCHEDULE)
    translator = GlossTranslator()

    records = harvest_utterance(
        audio, SAMPLE_RATE, asr=asr, translator=translator,
        final_asr=asr, final_translator=translator, session_id="utt-short",
    )
    assert records == []


def test_harvest_utterance_flags_guard_hold_on_hazardous_tail():
    # A schedule that stalls on a bare TAM particle for a while before the
    # verb arrives - the guard should hold during that stretch.
    schedule = [(0.0, "كان"), (2.0 * SAMPLE_RATE, "كان الولد يقرأ")]
    audio = _fake_audio(2.5)
    asr = ScheduledASR(schedule)
    translator = GlossTranslator()

    records = harvest_utterance(
        audio, SAMPLE_RATE, asr=asr, translator=translator,
        final_asr=asr, final_translator=translator, session_id="utt-guard",
    )
    assert any(r.guard_hold and r.guard_name == "tam_particle" for r in records)


def test_log_records_round_trip(tmp_path):
    audio = _fake_audio(3.5)
    asr = ScheduledASR(SCHEDULE)
    translator = GlossTranslator()
    records = harvest_utterance(
        audio, SAMPLE_RATE, asr=asr, translator=translator,
        final_asr=asr, final_translator=translator, session_id="utt-log",
    )
    log_path = tmp_path / "harvest.jsonl"
    log_records(records, log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(records)
