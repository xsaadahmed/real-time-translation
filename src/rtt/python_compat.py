"""Enforce the supported Python version early.

Docker and CI use CPython 3.12. Newer runtimes (especially 3.14) break
native wheels such as numpy, which is how local installs fail before
any RTT code runs.
"""

from __future__ import annotations

import sys

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 12


def supported_python(*, info: sys.version_info | None = None) -> bool:
    """Return True when *info* is a supported 3.12.x interpreter."""
    version = info or sys.version_info
    return version.major == REQUIRED_MAJOR and version.minor == REQUIRED_MINOR


def require_supported_python() -> None:
    """Exit with a clear install hint when the interpreter is not 3.12.x."""
    if supported_python():
        return

    found = (
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    raise SystemExit(
        f"This project requires Python {REQUIRED_MAJOR}.{REQUIRED_MINOR}.x "
        f"(found {found}).\n"
        "Python 3.13+ is not supported — native wheels such as numpy often fail.\n"
        "\n"
        "Create a 3.12 virtualenv, then retry:\n"
        "  Windows:  py -3.12 -m venv .venv && .\\.venv\\Scripts\\Activate.ps1\n"
        "  macOS/Linux: python3.12 -m venv .venv && source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
    )


__all__ = [
    "REQUIRED_MAJOR",
    "REQUIRED_MINOR",
    "require_supported_python",
    "supported_python",
]
