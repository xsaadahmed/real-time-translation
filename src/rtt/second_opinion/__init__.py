"""Independent Seamless ar->en second-opinion channel."""

from __future__ import annotations

from ..config import SecondOpinionConfig
from .agreement import SecondOpinionRecord, compare, log_record
from .base import SecondOpinionEngine


def build_second_opinion(config: SecondOpinionConfig | None = None) -> SecondOpinionEngine:
    """Construct the configured second-opinion engine."""
    from .seamless import SeamlessSecondOpinion

    return SeamlessSecondOpinion(config or SecondOpinionConfig())


__all__ = [
    "SecondOpinionEngine",
    "SecondOpinionRecord",
    "build_second_opinion",
    "compare",
    "log_record",
]
