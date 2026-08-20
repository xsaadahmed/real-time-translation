"""Launch the production Next.js UI and translation API.

    python run_production.py
    python run_production.py --api-port 8765 --ui-port 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
UI = ROOT / "production-ui"
HOST = "127.0.0.1"
MAX_PORT_ATTEMPTS = 20
# /health is available as soon as uvicorn binds (warmup is background).
API_BIND_TIMEOUT_SEC = 30.0
API_READY_TIMEOUT_SEC = 600.0
API_READY_POLL_SEC = 2.0


def _port_accepting(port: int, host: str = HOST) -> bool:
    """True when something accepts TCP connections on the port (bound)."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _api_alive(port: int) -> bool:
    """True when the API process is up (/health) or at least bound on the port."""
    if _port_accepting(port):
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            # Bound but HTTP not ready yet — still count as alive for bind phase.
            return True
    return False


def _api_ready(port: int) -> bool:
    """True when /ready succeeds (models loaded)."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/ready", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _api_healthy(port: int) -> bool:
    return _api_ready(port) or _api_alive(port)

# Gate before any rtt / numpy imports so Python 3.14 fails with a clear hint.
sys.path.insert(0, str(SRC))
from rtt.python_compat import require_supported_python  # noqa: E402

require_supported_python()


def _resolve_pm() -> tuple[str, list[str]]:
    """Return an executable path and base args for npm/pnpm (Windows-safe)."""
    candidates = (
        ["pnpm.cmd", "pnpm", "npm.cmd", "npm"]
        if sys.platform == "win32"
        else ["pnpm", "npm"]
    )
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path, []

    raise RuntimeError(
        "Node.js is required for the production UI. Install Node.js and ensure "
        "npm or pnpm is on PATH, then retry."
    )


def _is_port_in_use(port: int, host: str = HOST) -> bool:
    """Return True if something is already accepting connections on this port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _pick_port(start: int, host: str = HOST) -> int:
    for port in range(start, start + MAX_PORT_ATTEMPTS):
        if not _is_port_in_use(port, host):
            if port != start:
                print(f"Port {start} is busy — using {port} instead.")
            return port
    raise RuntimeError(
        f"No free port found between {start} and {start + MAX_PORT_ATTEMPTS - 1}."
    )


def _kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _kill_process_on_port(port: int) -> None:
    if sys.platform == "win32":
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts:
                    try:
                        _kill_pid(int(parts[-1]))
                    except ValueError:
                        pass
    else:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        for pid in result.stdout.split():
            try:
                _kill_pid(int(pid))
            except ValueError:
                pass


def _cleanup_stale_api(start_port: int, span: int = 60) -> None:
    """Stop leftover uvicorn instances from prior crashed / timed-out runs."""
    for port in range(start_port, start_port + span):
        if not _is_port_in_use(port):
            continue
        # Kill anything left listening — prior false timeouts leave many orphans.
        print(f"Releasing API port {port} (previous instance)…")
        _kill_process_on_port(port)
        time.sleep(0.3)


def _cleanup_stale_next_dev() -> None:
    """Next.js keeps a dev lock per project — clear it if a prior run crashed."""
    lock_path = UI / ".next" / "dev" / "lock"
    if not lock_path.exists():
        return

    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(data.get("pid", 0))
        if pid:
            print(f"Stopping stale Next.js dev server (PID {pid})…")
            _kill_pid(pid)
    except (json.JSONDecodeError, ValueError, OSError):
        pass

    try:
        lock_path.unlink()
    except OSError:
        pass


def _ui_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/", timeout=1.5) as resp:
            return resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


UI_STARTUP_TIMEOUT_SEC = 20.0


def _run_pm(pm: str, args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run([pm, *args], cwd=cwd, env=env, check=True)


def _popen_pm(
    pm: str,
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    return subprocess.Popen([pm, *args], cwd=cwd, env=env)


def _install_ui_deps(pm: str) -> None:
    node_modules = UI / "node_modules"
    if node_modules.exists():
        return
    print("Installing production UI dependencies (first run only)…")
    _run_pm(pm, ["install"], cwd=UI)


def _warmup_models() -> None:
    sys.path.insert(0, str(SRC))
    from rtt.ui.gradio_app import get_store

    print("Loading translation models…")
    get_store()
    print("Models ready.")


def _start_api(env: dict[str, str], start_port: int) -> tuple[subprocess.Popen, int]:
    for port in range(start_port, start_port + MAX_PORT_ATTEMPTS):
        if _is_port_in_use(port):
            print(f"API port {port} is busy, trying next…")
            continue

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "rtt.api.production_server:app",
            "--host",
            HOST,
            "--port",
            str(port),
        ]
        # Parent may already have warmed models; child still needs its own
        # in-process load. Give it a long ready window instead of port-hopping.
        child_env = env.copy()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env)

        # Phase 1: process must accept connections (bind succeeded).
        bind_deadline = time.monotonic() + API_BIND_TIMEOUT_SEC
        bound = False
        while time.monotonic() < bind_deadline:
            if proc.poll() is not None:
                break
            if _api_alive(port):
                bound = True
                break
            time.sleep(0.25)

        if proc.poll() is not None:
            print(f"API exited before binding on port {port}, trying next…")
            continue

        if not bound:
            print(f"API did not become reachable on port {port}, trying next…")
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            continue

        # Phase 2: wait for models (/ready). Do NOT hop ports while loading.
        print(
            f"API listening on {HOST}:{port} — loading models "
            f"(this can take several minutes on first run)…"
        )
        ready_deadline = time.monotonic() + API_READY_TIMEOUT_SEC
        last_note = 0.0
        while time.monotonic() < ready_deadline:
            if proc.poll() is not None:
                print(f"API process exited while loading models on port {port}.")
                break
            if _api_ready(port):
                if port != start_port:
                    print(f"API ready on port {port} (requested {start_port} was busy).")
                else:
                    print(f"API ready on http://{HOST}:{port}")
                return proc, port
            now = time.monotonic()
            if now - last_note >= 15.0:
                remaining = int(ready_deadline - now)
                print(f"  still loading models… ({remaining}s timeout remaining)")
                last_note = now
            time.sleep(API_READY_POLL_SEC)

        if proc.poll() is None:
            print(
                f"Timed out waiting for models on port {port} "
                f"after {int(API_READY_TIMEOUT_SEC)}s — stopping this worker."
            )
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Only try another port if this one failed hard; usually one attempt is enough.
        raise RuntimeError(
            f"API started on port {port} but models did not become ready within "
            f"{int(API_READY_TIMEOUT_SEC)}s. Check CPU/disk and retry; "
            f"or run `python scripts/download_models.py` first."
        )

    raise RuntimeError(
        f"Failed to start API after checking ports {start_port}–"
        f"{start_port + MAX_PORT_ATTEMPTS - 1}."
    )


def _start_ui(
    pm: str,
    start_port: int,
    env: dict[str, str],
) -> tuple[subprocess.Popen, int]:
    _cleanup_stale_next_dev()

    for port in range(start_port, start_port + MAX_PORT_ATTEMPTS):
        if _is_port_in_use(port):
            print(f"UI port {port} is busy, trying next…")
            continue

        proc = _popen_pm(
            pm,
            ["run", "dev", "--", "-H", HOST, "-p", str(port)],
            cwd=UI,
            env=env,
        )
        deadline = time.monotonic() + UI_STARTUP_TIMEOUT_SEC

        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(f"UI failed on port {port}, trying next…")
                break
            if _ui_healthy(port):
                time.sleep(0.5)
                if proc.poll() is None:
                    if port != start_port:
                        print(f"UI started on port {port} (requested {start_port} was busy).")
                    return proc, port
                print(f"UI exited on port {port}, trying next…")
                break
            time.sleep(0.5)

        if proc.poll() is None:
            return proc, port

    raise RuntimeError(
        f"Failed to start UI after checking ports {start_port}–"
        f"{start_port + MAX_PORT_ATTEMPTS - 1}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Production Arabic→English interpreter UI")
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--ui-port", type=int, default=3000)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--warmup-parent",
        action="store_true",
        help="Also load models in this process before starting the API (usually slower; API loads them anyway)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Deprecated alias: parent warmup is off by default now",
    )
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

    if args.warmup_parent and not args.skip_warmup:
        _warmup_models()
    else:
        print("Models will load inside the API process (first run can take a few minutes).")

    print("Cleaning up any leftover API workers from earlier runs…")
    _cleanup_stale_api(args.api_port)
    _cleanup_stale_next_dev()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env.setdefault("RTT_SKIP_MODEL_WARMUP", "0")

    try:
        api_proc, api_port = _start_api(env, args.api_port)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    ui_port = args.ui_port

    ui_env = env.copy()
    ws_url = f"ws://{HOST}:{api_port}/ws"
    ui_env["NEXT_PUBLIC_WS_URL"] = ws_url

    runtime_config = UI / "public" / "runtime-config.json"
    runtime_config.write_text(
        json.dumps({"wsUrl": ws_url}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"API:  http://{HOST}:{api_port}")
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

    def shutdown(_signum: int | None = None, _frame: object | None = None) -> None:
        for proc in (ui_proc, api_proc):
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            if api_proc.poll() is not None:
                print("API process exited.", file=sys.stderr)
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
