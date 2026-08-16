"""Local Arabic -> English speech translation.

Stage 1: a straightforward offline cascade of speech recognition, machine
translation and speech synthesis, running entirely on the local machine with
no external API calls. Streaming, anticipation and the commitment policy are
planned as later additions; the component interfaces here are built to accept
them without restructuring.
"""

from __future__ import annotations

import os


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

from .config import PipelineConfig  # noqa: E402
from .pipeline import PipelineResult, TranslationPipeline, build_pipeline  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "TranslationPipeline",
    "__version__",
    "build_pipeline",
]
