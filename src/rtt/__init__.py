"""Local Arabic -> English speech translation.

Heavy pipeline imports are lazy so lightweight modules (e.g. ``python_compat``,
``text``) can load on a wrong interpreter long enough to print a clear
Python 3.12 requirement instead of a cryptic numpy wheel error.
"""

from __future__ import annotations

import os
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "TranslationPipeline",
    "__version__",
    "build_pipeline",
]


def _configure_environment() -> None:
    """Environment defaults that must be set before huggingface_hub is imported.

    Windows refuses to create symlinks without Developer Mode or admin rights,
    which makes the default Hugging Face cache layout fail outright. Copying
    blobs instead costs some disk space and always works.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


_configure_environment()


def __getattr__(name: str) -> Any:
    if name == "PipelineConfig":
        from .config import PipelineConfig

        return PipelineConfig
    if name in {"PipelineResult", "TranslationPipeline", "build_pipeline"}:
        from .pipeline import PipelineResult, TranslationPipeline, build_pipeline

        return {
            "PipelineResult": PipelineResult,
            "TranslationPipeline": TranslationPipeline,
            "build_pipeline": build_pipeline,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
