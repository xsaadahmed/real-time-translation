import threading
import asyncio

import numpy as np

from rtt.live.context_session import ContextSessionStore
from rtt.live.runtime import create_live_config


class Pipeline:
    def __init__(self):
        self.audio_lengths = []
        self.sources = []

    def transcribe(self, audio, rate):
        self.audio_lengths.append(audio.size)
        return ("كلام ناقص" if audio.size < 128000 else "الكلام الصحيح كاملا"), None, {}

    def translate(self, source):
        self.sources.append(source)
        return f"translation({source})", {}


def append(store, state, seconds):
    store.append_chunk(state.session_id, 16000, np.full(int(seconds * 16000), .1, np.float32))


def test_full_context_revises_earlier_mistakes_and_translates_whole_source():
    pipeline = Pipeline()
    store = ContextSessionStore(pipeline)
    state = store.create()
    append(store, state, 4)
    store._run_live_increment(state)
    append(store, state, 4)
    store._run_live_increment(state)
    assert pipeline.audio_lengths == [64000, 128000]
    assert not state.arabic_verified
    assert state.arabic_provisional == "الكلام الصحيح كاملا"
    assert state.english_provisional == "translation(الكلام الصحيح كاملا)"
    assert pipeline.sources == ["كلام ناقص", "الكلام الصحيح كاملا"]


def test_stop_reuses_identical_snapshot_and_decodes_any_new_tail():
    pipeline = Pipeline()
    store = ContextSessionStore(pipeline)
    state = store.create()
    append(store, state, 4)
    store._run_live_increment(state)
    final = store.finalize(state.session_id)
    assert pipeline.audio_lengths == [64000]
    assert final.arabic_verified == "كلام ناقص"
    assert not final.arabic_provisional
    state = store.create()
    append(store, state, 4)
    store._run_live_increment(state)
    append(store, state, 4)
    final = store.finalize(state.session_id)
    assert pipeline.audio_lengths[-1] == 128000
    assert final.arabic_verified == "الكلام الصحيح كاملا"


def test_partial_window_after_pause_gets_decoded_without_stop():
    store = ContextSessionStore(Pipeline())
    state = store.create()
    append(store, state, 4)
    store._run_live_increment(state)
    append(store, state, .5)
    state.last_process_wall = 0
    decision = store._next_segment(state)
    assert decision is not None
    store._run_live_increment(state)
    assert state.processed_samples == 72000


def test_short_phrase_pause_triggers_earlier_update():
    store = ContextSessionStore(Pipeline())
    state = store.create()
    append(store, state, 1.5)
    store.append_chunk(state.session_id, 16000, np.zeros(8000, np.float32))
    assert state.duration_sec() == 2
    assert store._next_segment(state) is not None


def test_repeated_silence_keeps_text_without_repeated_inference():
    pipeline = Pipeline()
    store = ContextSessionStore(pipeline)
    state = store.create()
    append(store, state, 4)
    store._run_live_increment(state)
    for _ in range(3):
        store.append_chunk(state.session_id, 16000, np.zeros(8000, np.float32))
        store._run_live_increment(state)
    assert len(pipeline.audio_lengths) == 2
    assert state.arabic_text == "كلام ناقص"
    append(store, state, .5)
    store._run_live_increment(state)
    assert len(pipeline.audio_lengths) == 3


def test_silence_and_empty_input_never_reach_model():
    pipeline = Pipeline()
    store = ContextSessionStore(pipeline)
    for size in (0, 64000):
        state = store.create()
        store.append_chunk(state.session_id, 16000, np.zeros(size, np.float32))
        store._run_live_increment(state)
        final = store.finalize(state.session_id)
        assert not final.english_text
    assert not pipeline.audio_lengths and not pipeline.sources


def test_audio_arriving_during_decode_is_preserved():
    pipeline = Pipeline()
    store = ContextSessionStore(pipeline)
    state = store.create()
    append(store, state, 4)
    original = pipeline.transcribe

    def inject(audio, rate):
        append(store, state, 1)
        return original(audio, rate)

    pipeline.transcribe = inject
    store._run_live_increment(state)
    assert state.processed_samples == 64000
    assert state.audio.size == 80000
    pipeline.transcribe = original
    store.finalize(state.session_id)
    assert pipeline.audio_lengths == [64000, 80000]


def test_final_cannot_be_overwritten_by_late_live_decode():
    pipeline = Pipeline()
    store = ContextSessionStore(pipeline)
    state = store.create()
    append(store, state, 8)
    entered, release = threading.Event(), threading.Event()
    original = pipeline.transcribe

    def delayed(audio, rate):
        entered.set()
        assert release.wait(3)
        return original(audio, rate)

    pipeline.transcribe = delayed
    live = threading.Thread(target=store._run_live_increment, args=(state,))
    live.start()
    assert entered.wait(2)
    final = threading.Thread(target=store.finalize, args=(state.session_id,))
    final.start()
    release.set()
    live.join(3)
    final.join(3)
    assert not live.is_alive() and not final.is_alive()
    assert state.arabic_verified == "الكلام الصحيح كاملا"
    assert not state.arabic_provisional
    assert store.active_count() == 0


def test_recording_cap_and_inflight_admission():
    store = ContextSessionStore(Pipeline())
    state = store.create()
    append(store, state, 31)
    assert state.audio.size == 480000
    store.remove(state.session_id)
    with store._process_lock:
        assert store.active_count() == 1
    assert store.active_count() == 0


def test_api_profile_is_reviewed_and_uses_one_cpu_worker(monkeypatch):
    monkeypatch.setenv("RTT_MT_BACKEND", "nllb")
    monkeypatch.setenv("RTT_MT_MODEL", "facebook/nllb-200-distilled-600M")
    cfg = create_live_config()
    assert cfg.mt.backend == "marian"
    assert cfg.mt.model_name == "Helsinki-NLP/opus-mt-ar-en"
    assert cfg.asr.num_workers == 1
    assert cfg.asr.beam_size == 3
    assert cfg.asr.initial_prompt == ""


def test_socket_reports_processed_progress_even_when_text_is_unchanged(monkeypatch):
    from rtt.api import production_server as api
    store = ContextSessionStore(Pipeline())
    state = store.create()
    state.status_message = "No speech"
    monkeypatch.setattr(api, "get_store", lambda: store)
    monkeypatch.setattr(api, "WS_POLL_SEC", .01)

    async def run():
        stop = asyncio.Event()
        messages = []

        class Socket:
            async def send_json(self, payload):
                messages.append(payload)
                if len(messages) == 1:
                    state.processed_samples = 16000
                    state.decoded_duration_sec = 1
                else:
                    stop.set()

        await asyncio.wait_for(api._poll_updates(Socket(), state.session_id, stop), 1)
        assert len(messages) == 2
        assert messages[-1]["decoded_duration_sec"] == 1

    asyncio.run(run())
