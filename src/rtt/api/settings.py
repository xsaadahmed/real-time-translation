"""Production API settings from environment variables."""

from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _csv_list(name: str, default: str) -> list[str]:
    raw = _env(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


# Bind address inside the container / process.
API_HOST: str = _env("RTT_API_HOST", "0.0.0.0")
API_PORT: int = int(_env("RTT_API_PORT", "8765"))

# Public site URL (https://translate.example.com). Used for docs and optional
# runtime-config generation. No trailing slash.
PUBLIC_URL: str = _env("RTT_PUBLIC_URL", "").rstrip("/")

# WebSocket URL exposed to browsers (wss://translate.example.com/ws). When empty,
# the UI falls back to same-origin /ws via the reverse proxy.
PUBLIC_WS_URL: str = _env("RTT_PUBLIC_WS_URL", "").rstrip("/")

# Comma-separated browser origins allowed to call the API directly (dev / split
# deploy). Same-origin reverse-proxy deploys do not need CORS for /ws.
_DEFAULT_CORS = (
    "http://127.0.0.1:3000,"
    "http://localhost:3000,"
    "http://127.0.0.1:3001,"
    "http://localhost:3001"
)
CORS_ORIGINS: list[str] = _csv_list("RTT_CORS_ORIGINS", _DEFAULT_CORS)

# Trust X-Forwarded-* from reverse proxy (Caddy, nginx, ALB).
TRUST_PROXY_HEADERS: bool = _env("RTT_TRUST_PROXY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

__all__ = [
    "API_HOST",
    "API_PORT",
    "CORS_ORIGINS",
    "PUBLIC_URL",
    "PUBLIC_WS_URL",
    "TRUST_PROXY_HEADERS",
]
