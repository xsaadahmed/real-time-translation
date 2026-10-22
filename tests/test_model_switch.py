from types import SimpleNamespace
import threading
import weakref

import pytest

from rtt.config import ASRConfig
from rtt.live import runtime
from rtt.live.context_session import ContextSessionStore
from rtt.api import readiness


class Weights:
    pass


class Engine:
    def __init__(self, config):
        self.config = config
        self._model = None

    def load(self):
        self._model = Weights()


@pytest.fixture
def setup_runtime(monkeypatch):
    engine = Engine(ASRConfig(model_size="small"))
    engine.load()
    pipeline = SimpleNamespace(asr=engine, translator=object(), config=SimpleNamespace(asr=engine.config))
    store = ContextSessionStore(pipeline)
    monkeypatch.setattr(runtime, "_store", store)
    monkeypatch.setattr(runtime, "_model_size", "small")
    monkeypatch.setattr(runtime, "_loading_model", None)
    monkeypatch.setattr(runtime, "_model_error", None)
    from rtt import asr
    monkeypatch.setattr(asr, "build_asr", Engine)
    return store


@pytest.mark.parametrize("size", ["tiny", "base", "medium", "large-v3-turbo", "large-v3"])
def test_switch_unloads_old_weights_before_loading_and_keeps_translator(setup_runtime, monkeypatch, size):
    store = setup_runtime
    old = weakref.ref(store.live_pipeline.asr._model)
    translator = store.live_pipeline.translator
    load = Engine.load

    def checked_load(self):
        assert old() is None
        load(self)

    monkeypatch.setattr(Engine, "load", checked_load)
    statuses = []
    runtime.change_model(size, statuses.append, lambda: statuses.append("ready"), statuses.append)
    assert runtime.describe_runtime_config()["live_asr"] == size
    assert runtime.describe_runtime_config()["final_asr"] == size
    assert store.live_pipeline.translator is translator
    assert store._get_final_pipeline() is store.live_pipeline
    assert statuses[-1] == "ready"


def test_busy_and_invalid_requests_leave_models_untouched(setup_runtime):
    weights = setup_runtime.live_pipeline.asr._model
    for invalid in ("large", "../../outside", "facebook/nllb"):
        with pytest.raises(ValueError):
            runtime.change_model(invalid, print, lambda: None, print)
    setup_runtime.create()
    with pytest.raises(RuntimeError, match="Stop"):
        runtime.change_model("medium", print, lambda: None, print)
    assert setup_runtime.live_pipeline.asr._model is weights
    assert runtime.describe_runtime_config()["live_asr"] == "small"


def test_failure_restores_cached_previous_model(setup_runtime, monkeypatch):
    load = Engine.load
    calls = []

    def flaky(self):
        calls.append(self.config.model_size)
        if self.config.model_size == "medium":
            raise OSError("simulated model failure")
        load(self)

    monkeypatch.setattr(Engine, "load", flaky)
    states = []
    with pytest.raises(RuntimeError, match="Restoring small"):
        runtime.change_model("medium", states.append, lambda: states.append("ready"), states.append)
    assert calls == ["medium", "small"]
    assert setup_runtime.live_pipeline.asr._model is not None
    assert runtime.describe_runtime_config()["live_asr"] == "small"
    assert runtime.describe_runtime_config()["loading_model"] is None
    assert states[-1] == "ready"


def test_loading_excludes_sessions_and_parallel_switches(setup_runtime, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    load = Engine.load

    def delayed(self):
        entered.set()
        assert release.wait(3)
        load(self)

    monkeypatch.setattr(Engine, "load", delayed)
    worker = threading.Thread(target=runtime.change_model, args=("medium", lambda _: None, lambda: None, lambda _: None))
    worker.start()
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="loading"):
            runtime.create_session()
        with pytest.raises(RuntimeError, match="already"):
            runtime.change_model("base", print, lambda: None, print)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()


def test_legacy_inconsistent_or_unlicensed_settings_are_rejected(setup_runtime):
    with pytest.raises(ValueError):
        readiness.apply_config(live_asr="small", final_asr="medium")
    with pytest.raises(ValueError):
        readiness.apply_config(live_mt="nllb")
