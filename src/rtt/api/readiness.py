"""API readiness + live/final model configuration."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_ready = False
_ready_error: str | None = None
_lock = threading.Lock()

LIVE_ASR_OPTIONS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
FINAL_ASR_OPTIONS = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")
MT_OPTIONS = ("marian",)
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
    from ..live.runtime import describe_runtime_config

    return {
        "ready": _ready,
        "error": _ready_error,
        "source_lang": SOURCE_LANG,
        "target_lang": TARGET_LANG,
        "modes": ["live", "simulated"],
        "exhibition_mode": True,
        "device": "cpu",
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
    """Change only the speech model; live and final use the same selected size."""
    from ..live.runtime import change_model, describe_runtime_config
    selected = live_asr or final_asr or describe_runtime_config()["live_asr"]
    if live_asr and final_asr and live_asr != final_asr:
        raise ValueError("Live and final must use the same speech model.")
    if live_mt is not None and live_mt != "marian":
        raise ValueError("The translation model must remain Marian.")
    change_model(selected, mark_not_ready, mark_ready, mark_not_ready)
    return current_config()


def warmup_models() -> None:
    """Load live pipeline (and mark ready). Called from API lifespan."""
    mark_not_ready("Loading models…")
    try:
        from ..live.runtime import get_store

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
