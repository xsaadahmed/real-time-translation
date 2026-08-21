"""Tests wiring the risk-based commit policy (steps 8-10) into
LiveSessionStore. Uses stub ASR/Translator/TTS so this stays fast and
hermetic - no real models, mirroring tests/test_harvest.py's approach.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.asr.base import ASREngine, Transcript
from rtt.commit import CommitPolicy
from rtt.config import CommitPolicyConfig
from rtt.live.session import LiveSessionStore, MIN_NEW_AUDIO_SEC
from rtt.mt.base import Translator
from rtt.pipeline import TranslationPipeline
from rtt.risk import RiskModel
from rtt.tts.base import SpeechAudio, TTSEngine

SAMPLE_RATE = 16_000


class FixedASR(ASREngine):
    """Always transcribes the same Arabic sentence, regardless of chunk -
    the live commit policy loop under test doesn't depend on real ASR
    behavior, just on there being provisional English to commit against.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def load(self) -> None:
        pass

    def transcribe(self, audio, sample_rate: int) -> Transcript:
        return Transcript(text=self.text)


class GlossTranslator(Translator):
    GLOSS = {
        "ذهب": "went", "الولد": "the-boy", "الى": "to", "المدرسة": "school",
    }

    def load(self) -> None:
        pass

    def translate_batch(self, texts: list[str]) -> list[str]:
        return [
            " ".join(self.GLOSS.get(word, word) for word in text.split())
            for text in texts
        ]


class RecordingTTS(TTSEngine):
    name = "recording"

    def __init__(self) -> None:
        self.synth_calls: list[str] = []

    def load(self) -> None:
        pass

    def synthesize(self, text: str) -> SpeechAudio:
        self.synth_calls.append(text)
        n_words = max(1, len(text.split()))
        return SpeechAudio(audio=np.ones(n_words * SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)


def _record(text: str, survived: bool) -> dict:
    return {
        "session_id": "s", "tick": 0, "arabic_verified": "", "arabic_provisional": "",
        "english_candidate": text, "guard_hold": False, "guard_name": "",
        "agreement_depth": None, "branch_count": 1, "second_opinion_similarity": None,
        "survived": survived,
    }


def _always_commits_risk_model() -> RiskModel:
    """A risk model trained so every non-trivial candidate clears any
    reasonable theta - isolates "does the wiring commit at all" from risk
    model quality, which step 8's real training script already covers.
    """
    records = [_record("went the-boy to school", True) for _ in range(20)]
    records += [_record("x", False) for _ in range(20)]
    return RiskModel().fit(records)


def _make_pipeline(text: str) -> TranslationPipeline:
    return TranslationPipeline(asr=FixedASR(text), translator=GlossTranslator())


def _fake_audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_commit_policy_disabled_by_default_leaves_state_unchanged():
    store = LiveSessionStore(
        _make_pipeline("ذهب الولد الى المدرسة"),
        final_pipeline_getter=lambda: pytest.fail("should not be called"),
    )
    state = store.create()
    store.append_chunk(state.session_id, SAMPLE_RATE, _fake_audio(MIN_NEW_AUDIO_SEC + 1))

    store._run_live_increment(state)

    assert state.english_risk_committed == ""
    assert state.commit_theta == 0.0


def test_commit_policy_enabled_promotes_text_out_of_provisional():
    policy = CommitPolicy(_always_commits_risk_model())
    store = LiveSessionStore(
        _make_pipeline("ذهب الولد الى المدرسة"),
        final_pipeline_getter=lambda: pytest.fail("should not be called"),
        commit_policy_config=CommitPolicyConfig(enabled=True),
        commit_policy=policy,
    )
    state = store.create()
    store.append_chunk(state.session_id, SAMPLE_RATE, _fake_audio(MIN_NEW_AUDIO_SEC + 1))

    store._run_live_increment(state)

    assert state.english_risk_committed != ""
    assert state.committed_through_sec > 0.0
    # Whatever got risk-committed should be a real prefix of the full
    # translation, not something fabricated.
    assert "went the-boy to school".startswith(state.english_risk_committed)


def test_risk_committed_text_is_included_in_english_text_property():
    policy = CommitPolicy(_always_commits_risk_model())
    store = LiveSessionStore(
        _make_pipeline("ذهب الولد الى المدرسة"),
        final_pipeline_getter=lambda: pytest.fail("should not be called"),
        commit_policy_config=CommitPolicyConfig(enabled=True),
        commit_policy=policy,
    )
    state = store.create()
    store.append_chunk(state.session_id, SAMPLE_RATE, _fake_audio(MIN_NEW_AUDIO_SEC + 1))
    store._run_live_increment(state)

    assert state.english_risk_committed in state.english_text


def test_missing_risk_model_file_degrades_gracefully():
    from rtt.config import CommitPolicyConfig as CPC

    store = LiveSessionStore(
        _make_pipeline("ذهب الولد"),
        final_pipeline_getter=lambda: pytest.fail("should not be called"),
        commit_policy_config=CPC(enabled=True, risk_model_path="does/not/exist.joblib"),
    )
    state = store.create()
    store.append_chunk(state.session_id, SAMPLE_RATE, _fake_audio(MIN_NEW_AUDIO_SEC + 1))

    # Must not raise - falls back to old behaviour.
    store._run_live_increment(state)
    assert state.english_risk_committed == ""


def test_speculative_tts_commits_and_pushes_to_jitter_buffer():
    policy = CommitPolicy(_always_commits_risk_model())
    tts = RecordingTTS()
    store = LiveSessionStore(
        _make_pipeline("ذهب الولد الى المدرسة"),
        final_pipeline_getter=lambda: pytest.fail("should not be called"),
        commit_policy_config=CommitPolicyConfig(enabled=True, speculative_tts=True),
        commit_policy=policy,
        tts_engine=tts,
    )
    state = store.create()
    store.append_chunk(state.session_id, SAMPLE_RATE, _fake_audio(MIN_NEW_AUDIO_SEC + 1))

    store._run_live_increment(state)

    assert len(tts.synth_calls) > 0
    assert store.jitter_buffer(state.session_id).buffered_sec > 0.0


def test_remove_clears_jitter_buffer():
    store = LiveSessionStore(
        _make_pipeline("ذهب"),
        final_pipeline_getter=lambda: pytest.fail("should not be called"),
    )
    state = store.create()
    store.jitter_buffer(state.session_id)
    assert state.session_id in store._jitter_buffers
    store.remove(state.session_id)
    assert state.session_id not in store._jitter_buffers
