"""Launch the production UI with a hardcoded target-architecture demo API.

Same shell as ``run_production.py`` (Next.js UI + WebSocket API + port
fallback), but the API plays a scripted Arabic→English session that shows
the *expected* draft-and-verify UX:

  • grey provisional text appears first (~600 ms)
  • black committed text grows as words are verified (~1.5 s+)
  • grey stays ahead with speculative wording (including a revision)
  • Stop finalizes in under a second (no model load)

No Whisper / NLLB weights are loaded. Use this to show the boss what the
final product should feel like; use ``run_production.py`` for the real cascade.

    python run_demo.py
    python run_demo.py --api-port 8765 --ui-port 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
UI = ROOT / "production-ui"
HOST = "127.0.0.1"
MAX_PORT_ATTEMPTS = 20
API_BIND_TIMEOUT_SEC = 15.0

sys.path.insert(0, str(SRC))
from rtt.python_compat import require_supported_python  # noqa: E402

require_supported_python()

# Reuse production launcher helpers so behavior stays aligned.
from run_production import (  # noqa: E402
    _api_alive,
    _api_ready,
    _cleanup_stale_api,
    _cleanup_stale_next_dev,
    _install_ui_deps,
    _is_port_in_use,
    _resolve_pm,
    _start_ui,
)


def _start_demo_api(env: dict[str, str], start_port: int) -> tuple[subprocess.Popen, int]:
    for port in range(start_port, start_port + MAX_PORT_ATTEMPTS):
        if _is_port_in_use(port):
            print(f"API port {port} is busy, trying next…")
            continue

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "rtt.api.demo_server:app",
            "--host",
            HOST,
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env)

        bind_deadline = time.monotonic() + API_BIND_TIMEOUT_SEC
        while time.monotonic() < bind_deadline:
            if proc.poll() is not None:
                break
            if _api_alive(port) and _api_ready(port):
                if port != start_port:
                    print(f"Demo API ready on port {port} (requested {start_port} was busy).")
                else:
                    print(f"Demo API ready on http://{HOST}:{port}")
                return proc, port
            time.sleep(0.2)

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(f"Demo API did not become ready on port {port}, trying next…")

    raise RuntimeError(
        f"Failed to start demo API after checking ports {start_port}–"
        f"{start_port + MAX_PORT_ATTEMPTS - 1}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo UI: target draft-and-verify UX (hardcoded, no models)"
    )
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--ui-port", type=int, default=3000)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not UI.is_dir():
        print(f"Missing production UI at {UI}", file=sys.stderr)
        return 1

    pm, _ = _resolve_pm()
    if not args.skip_install:
        _install_ui_deps(pm)

    print()
    print("=" * 60)
    print("  DEMO MODE — target architecture UX (hardcoded)")
    print("  Black = committed   Grey = speculative")
    print("  No models loaded. Press Start, watch the script play.")
    print("  Real pipeline:  python run_production.py")
    print("=" * 60)
    print()

    print("Cleaning up any leftover API workers from earlier runs…")
    _cleanup_stale_api(args.api_port)
    _cleanup_stale_next_dev()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)

    try:
        api_proc, api_port = _start_demo_api(env, args.api_port)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    ws_url = f"ws://{HOST}:{api_port}/ws"
    ui_env = env.copy()
    ui_env["NEXT_PUBLIC_WS_URL"] = ws_url

    runtime_config = UI / "public" / "runtime-config.json"
    runtime_config.write_text(
        json.dumps({"wsUrl": ws_url, "mode": "demo"}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"API:  http://{HOST}:{api_port}  (demo)")
    print(f"WS:   {ws_url}")
    print(f"Using: {pm}")
    print("Press Ctrl+C to stop both servers.")

    try:
        ui_proc, ui_port = _start_ui(pm, args.ui_port, ui_env)
    except RuntimeError as exc:
        api_proc.terminate()
        print(exc, file=sys.stderr)
        return 1

    print(f"UI:   http://{HOST}:{ui_port}")
    print()
    print("Open the UI → click Start → watch grey then black grow.")
    print("Click Stop anytime for a fast 'final' commit of the full line.")

    def shutdown(_signum: int | None = None, _frame: object | None = None) -> None:
        for proc in (ui_proc, api_proc):
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            if api_proc.poll() is not None:
                print("Demo API process exited.", file=sys.stderr)
                shutdown()
                break
            if ui_proc.poll() is not None:
                print("UI process exited.", file=sys.stderr)
                shutdown()
                break
            time.sleep(0.5)
    finally:
        shutdown()
        for proc in (ui_proc, api_proc):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
