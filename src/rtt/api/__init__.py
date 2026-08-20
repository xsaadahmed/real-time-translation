"""HTTP/WebSocket API for the production Next.js UI."""

from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from .production_server import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
