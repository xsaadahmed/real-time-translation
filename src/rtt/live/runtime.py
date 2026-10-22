"""Single model owner for the local exhibition API (no Gradio import)."""
from __future__ import annotations

import threading
import gc
from dataclasses import replace

from ..config import PipelineConfig
from ..pipeline import build_pipeline
from .context_session import ContextSessionStore

_store = None
_init_lock = threading.Lock()
MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
_model_size = "small"
_loading_model = None
_model_error = None


def create_live_config(model_size=None):
    cfg = PipelineConfig()
    cfg.asr.model_size = model_size or _model_size
    cfg.asr.device = "cpu"
    cfg.asr.compute_type = "int8"
    cfg.asr.language = "ar"
    cfg.asr.beam_size = 3
    cfg.asr.temperatures = (0.0,)
    cfg.asr.condition_on_previous_text = True
    cfg.asr.initial_prompt = ""
    cfg.asr.vad_filter = False
    cfg.asr.cpu_threads = 4
    cfg.asr.num_workers = 1
    cfg.mt.backend = "marian"
    cfg.mt.model_name = "Helsinki-NLP/opus-mt-ar-en"
    cfg.mt.device = "cpu"
    cfg.mt.num_beams = 2
    cfg.mt.torch_threads = 4
    cfg.mt.max_new_tokens = 256
    cfg.second_opinion.enabled = False
    cfg.vad.enabled = False
    return cfg


def get_store():
    global _store
    if _store is None:
        with _init_lock:
            if _store is None:
                pipeline = build_pipeline(create_live_config(), include_tts=False)
                pipeline.warmup()
                _store = ContextSessionStore(pipeline)
    return _store


def describe_runtime_config():
    return {"live_asr": _model_size, "live_mt": "marian", "final_asr": _model_size,
            "final_mt": "marian", "live_strategy": "full_context",
            "update_audio_sec": 4.0, "asr_beam": 3,
            "loading_model": _loading_model, "model_error": _model_error,
            "finalize_timeout_sec": {"medium": 120, "large-v3-turbo": 180, "large-v3": 240}.get(_model_size, 45)}


def create_session():
    """Atomically admit a session against model switching, without blocking UI."""
    if not _init_lock.acquire(blocking=False):
        raise RuntimeError("A speech model is loading. Please wait.")
    try:
        if _loading_model or _store is None:
            raise RuntimeError("Speech model is not ready.")
        if _store.active_count():
            raise RuntimeError("A live session is already active or processing.")
        return _store, _store.create()
    finally:
        _init_lock.release()


def change_model(model_size, mark_loading, mark_ready, mark_failed):
    """Unload old ASR before loading the next. Keep Marian cached, reuse audio policy.

    On failure reload the old cached ASR; never retain two large models as a
    rollback mechanism. This lock also prevents session/config races.
    """
    global _model_size, _loading_model, _model_error
    if model_size not in MODEL_SIZES:
        raise ValueError("Unsupported speech model.")
    if not _init_lock.acquire(blocking=False):
        raise RuntimeError("Another model operation is already running.")
    try:
        if _store is None:
            raise RuntimeError("Wait for the initial speech model to finish loading.")
        if _store.active_count():
            raise RuntimeError("Stop the current session before changing models.")
        if model_size == _model_size and _store.live_pipeline.asr._model is not None:
            return
        from ..asr import build_asr
        pipeline = _store.live_pipeline
        old_config = replace(pipeline.asr.config)
        _loading_model, _model_error = model_size, None
        mark_loading(f"Loading {model_size} speech model; first use may download weights…")
        # The translator does not change. Dropping the native Whisper object
        # first is essential on a 16GB shared-memory laptop.
        pipeline.asr._model = None
        gc.collect()
        try:
            pipeline.asr = build_asr(create_live_config(model_size).asr)
            pipeline.asr.load()
            pipeline.config.asr = pipeline.asr.config
            _model_size = model_size
            mark_ready()
        except Exception as exc:
            pipeline.asr._model = None
            gc.collect()
            _model_error = f"Could not load {model_size}. Restoring {old_config.model_size}."
            try:
                pipeline.asr = build_asr(old_config)
                pipeline.asr.load()
                pipeline.config.asr = old_config
                mark_ready()
            except Exception:
                mark_failed("Speech model unavailable. Restart or use the simulated preview.")
            raise RuntimeError(_model_error) from exc
        finally:
            _loading_model = None
    finally:
        _init_lock.release()
