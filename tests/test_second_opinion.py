"""Tests for the Seamless second-opinion channel (step 6).

Uses a stub SecondOpinionEngine so this stays fast and hermetic - no model
download. The stub is injected directly into LiveSessionStore, bypassing the
lazy build_second_opinion() import so no real Seamless weights are needed.
"""

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.config import SecondOpinionConfig
from rtt.live.session import LiveSessionStore, LiveStreamState
from rtt.second_opinion.agreement import compare, log_record
from rtt.second_opinion.base import SecondOpinionEngine


class StubSecondOpinion(SecondOpinionEngine):
    name = "stub"

    def __init__(self, fixed_text: str) -> None:
        self.fixed_text = fixed_text
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def translate_speech(self, audio, sample_rate) -> str:
        return self.fixed_text


def test_compare_identical_text_is_full_similarity():
    record = compare("The boy went to school", "The boy went to school")
    assert record.similarity == 1.0
    assert record.agreed_prefix == "The boy went to school"


def test_compare_divergent_text_has_partial_similarity_and_prefix():
    record = compare("The boy went to school", "The boy ran home quickly")
    assert 0.0 < record.similarity < 1.0
    assert record.agreed_prefix == "The boy"


def test_compare_empty_second_opinion_has_zero_similarity():
    record = compare("The boy went to school", "")
    assert record.similarity == 0.0
    assert record.agreed_prefix == ""


def test_log_record_round_trip():
    record = compare("hello world", "hello there", session_id="abc123")
    with tempfile.TemporaryDirectory() as tmp:
        log_path = pathlib.Path(tmp) / "second_opinion_log.jsonl"
        log_record(record, log_path)
        log_record(record, log_path)  # appends, doesn't overwrite

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["cascade_text"] == "hello world"
        assert parsed["second_opinion_text"] == "hello there"
        assert parsed["session_id"] == "abc123"
        assert parsed["agreed_prefix"] == "hello"


def test_second_opinion_disabled_by_default_never_runs():
    config = SecondOpinionConfig()
    assert config.enabled is False


def test_live_session_store_logs_second_opinion_when_enabled():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = pathlib.Path(tmp) / "log.jsonl"
        config = SecondOpinionConfig(enabled=True, log_path=str(log_path))

        store = LiveSessionStore(
            live_pipeline=None,  # type: ignore[arg-type]
            final_pipeline_getter=lambda: None,  # type: ignore[arg-type,return-value]
            second_opinion_config=config,
        )
        store._second_opinion_engine = StubSecondOpinion("The boy went to school")

        state = LiveStreamState(session_id="sess-1")
        state.english_verified = "The boy went to school"

        thread = store._run_second_opinion_async(state)
        thread.join(timeout=5)

        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["cascade_text"] == "The boy went to school"
        assert parsed["second_opinion_text"] == "The boy went to school"
        assert parsed["similarity"] == 1.0
        assert parsed["session_id"] == "sess-1"


def test_live_session_store_survives_second_opinion_failure():
    """A crashing second-opinion engine must not raise into the caller -
    it runs in a background thread and only ever logs, per the design."""

    class BrokenSecondOpinion(SecondOpinionEngine):
        def load(self) -> None:
            pass

        def translate_speech(self, audio, sample_rate) -> str:
            raise RuntimeError("boom")

    with tempfile.TemporaryDirectory() as tmp:
        log_path = pathlib.Path(tmp) / "log.jsonl"
        config = SecondOpinionConfig(enabled=True, log_path=str(log_path))
        store = LiveSessionStore(
            live_pipeline=None,  # type: ignore[arg-type]
            final_pipeline_getter=lambda: None,  # type: ignore[arg-type,return-value]
            second_opinion_config=config,
        )
        store._second_opinion_engine = BrokenSecondOpinion()

        state = LiveStreamState(session_id="sess-2")
        state.english_verified = "hello"

        thread = store._run_second_opinion_async(state)
        thread.join(timeout=5)  # must not raise / hang

        assert not log_path.exists()
