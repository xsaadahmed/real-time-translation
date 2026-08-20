"""API readiness + live/final model configuration."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_ready = False
_ready_error: str | None = None
_lock = threading.Lock()

LIVE_ASR_OPTIONS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
FINAL_ASR_OPTIONS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
MT_OPTIONS = ("marian", "nllb")
# Product is Arabic → English; exposed for UI clarity / future expansion.
SOURCE_LANG = "ar"
TARGET_LANG = "en"


def is_ready() -> bool:
    return _ready


def ready_error() -> str | None:
    return _ready_error


def mark_ready() -> None:
    global _ready, _ready_error
    with _lock:
        _ready = True
        _ready_error = None


def mark_not_ready(error: str | None = None) -> None:
    global _ready, _ready_error
    with _lock:
        _ready = False
        _ready_error = error


def current_config() -> dict[str, Any]:
    from ..config import detect_device
    from ..ui.gradio_app import describe_runtime_config

    return {
        "ready": _ready,
        "source_lang": SOURCE_LANG,
        "target_lang": TARGET_LANG,
        "device": detect_device(),
        "options": {
            "live_asr": list(LIVE_ASR_OPTIONS),
            "final_asr": list(FINAL_ASR_OPTIONS),
            "live_mt": list(MT_OPTIONS),
        },
        **describe_runtime_config(),
    }


def apply_config(
    *,
    live_asr: str | None = None,
    final_asr: str | None = None,
    live_mt: str | None = None,
) -> dict[str, Any]:
    """Update model env prefs and rebuild the live store when idle."""
    from ..ui.gradio_app import active_session_count, reset_store

    if live_asr is not None:
        if live_asr not in LIVE_ASR_OPTIONS:
            raise ValueError(f"Invalid live_asr: {live_asr}")
        os.environ["RTT_LIVE_ASR_MODEL"] = live_asr
    if final_asr is not None:
        if final_asr not in FINAL_ASR_OPTIONS:
            raise ValueError(f"Invalid final_asr: {final_asr}")
        os.environ["RTT_ASR_MODEL"] = final_asr
    if live_mt is not None:
        if live_mt not in MT_OPTIONS:
            raise ValueError(f"Invalid live_mt: {live_mt}")
        os.environ["RTT_LIVE_MT_BACKEND"] = live_mt

    if active_session_count() > 0:
        raise RuntimeError("Cannot change models while a session is active.")

    mark_not_ready("Reloading models…")
    try:
        reset_store()
        # Force reload via get_store side effect
        from ..ui.gradio_app import get_store

        get_store()
        mark_ready()
    except Exception as exc:
        mark_not_ready(str(exc))
        raise

    return current_config()


def warmup_models() -> None:
    """Load live pipeline (and mark ready). Called from API lifespan."""
    mark_not_ready("Loading models…")
    try:
        from ..ui.gradio_app import get_store

        get_store()
        mark_ready()
        logger.info("Models ready", extra={"event": "models_ready"})
    except Exception as exc:
        mark_not_ready(str(exc))
        logger.exception("Model warmup failed", extra={"event": "models_warmup_failed"})
        raise


__all__ = [
    "FINAL_ASR_OPTIONS",
    "LIVE_ASR_OPTIONS",
    "MT_OPTIONS",
    "SOURCE_LANG",
    "TARGET_LANG",
    "apply_config",
    "current_config",
    "is_ready",
    "mark_not_ready",
    "mark_ready",
    "ready_error",
    "warmup_models",
]
