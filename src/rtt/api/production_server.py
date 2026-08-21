"""FastAPI WebSocket bridge for the production Next.js interpreter UI."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..ui.gradio_app import get_store
from .logging_config import configure_logging
from .metrics import metrics
from .readiness import (
    apply_config,
    current_config,
    is_ready,
    ready_error,
    warmup_models,
)
from .settings import CORS_ORIGINS, PUBLIC_URL, PUBLIC_WS_URL

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rtt-api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging(
        json_logs=os.environ.get("RTT_JSON_LOGS", "1").strip().lower()
        in {"1", "true", "yes", "on"},
        level=os.environ.get("RTT_LOG_LEVEL", "INFO"),
    )
    # Always warm in the background so uvicorn can bind and answer /health
    # immediately. Blocking warmup here made the launcher think the port
    # never came up (models take far longer than the bind timeout).
    from .readiness import mark_not_ready

    mark_not_ready("Loading models…")
    loop = asyncio.get_running_loop()
    warm_future = loop.run_in_executor(_executor, warmup_models)
    logger.info(
        "API listening; model warmup running in background",
        extra={"event": "api_start"},
    )
    try:
        yield
    finally:
        if not warm_future.done():
            warm_future.cancel()
        logger.info("API shutting down", extra={"event": "api_shutdown"})
        _executor.shutdown(wait=False, cancel_futures=True)


#: How often the websocket checks session state for new text to push. This sits
#: on top of the live pipeline's own latency, so keep it well under it.
WS_POLL_SEC = float(os.environ.get("RTT_WS_POLL_SEC", "0.1"))

app = FastAPI(title="RTT Production API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigUpdate(BaseModel):
    live_asr: str | None = Field(default=None)
    final_asr: str | None = Field(default=None)
    live_mt: str | None = Field(default=None)


def _state_payload(
    state: Any,
    *,
    finalized: bool = False,
    phase: str | None = None,
) -> dict[str, Any]:
    payload = {
        "type": "final" if finalized else "update",
        "arabic": state.arabic_text,
        "english": state.english_text,
        "arabic_verified": state.arabic_verified,
        "arabic_provisional": state.arabic_provisional,
        "english_verified": state.english_verified,
        "english_provisional": state.english_provisional,
        "status": state.status_message,
        "duration_sec": state.duration_sec(),
        "finalized": finalized,
        "phase": phase
        or ("final" if finalized else ("listening" if state.is_active else "idle")),
    }
    return payload


async def _send_state(
    websocket: WebSocket,
    state: Any,
    *,
    finalized: bool = False,
    phase: str | None = None,
) -> None:
    await websocket.send_json(_state_payload(state, finalized=finalized, phase=phase))


async def _send_progress(websocket: WebSocket, status: str, *, phase: str = "finalize") -> None:
    await websocket.send_json(
        {
            "type": "progress",
            "status": status,
            "phase": phase,
            "finalizing": True,
            "finalized": False,
        }
    )


async def _poll_updates(
    websocket: WebSocket,
    session_id: str,
    stop_event: asyncio.Event,
) -> None:
    store = get_store()
    last_ar_verified = ""
    last_ar_provisional = ""
    last_en_verified = ""
    last_en_provisional = ""
    last_status = ""

    while not stop_event.is_set():
        state = store.get(session_id)
        if state is None or not state.is_active:
            break

        changed = (
            state.arabic_verified != last_ar_verified
            or state.arabic_provisional != last_ar_provisional
            or state.english_verified != last_en_verified
            or state.english_provisional != last_en_provisional
            or state.status_message != last_status
        )
        if changed:
            last_ar_verified = state.arabic_verified
            last_ar_provisional = state.arabic_provisional
            last_en_verified = state.english_verified
            last_en_provisional = state.english_provisional
            last_status = state.status_message
            await _send_state(websocket, state, phase="listening")

        await asyncio.sleep(WS_POLL_SEC)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness — process is up (Compose/Caddy may still wait on /ready)."""
    payload: dict[str, str] = {"status": "ok"}
    if PUBLIC_URL:
        payload["public_url"] = PUBLIC_URL
    if PUBLIC_WS_URL:
        payload["ws_url"] = PUBLIC_WS_URL
    return payload


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness — models loaded and accepting sessions."""
    if is_ready():
        return JSONResponse({"status": "ready", "ready": True})
    body = {
        "status": "not_ready",
        "ready": False,
        "error": ready_error() or "Models still loading",
    }
    return JSONResponse(body, status_code=503)


@app.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    return {"ready": is_ready(), **metrics.snapshot()}


@app.get("/config")
async def get_config() -> dict[str, Any]:
    return current_config()


@app.put("/config")
async def put_config(body: ConfigUpdate) -> dict[str, Any]:
    if not is_ready() and ready_error() and "Reloading" not in (ready_error() or ""):
        raise HTTPException(status_code=503, detail=ready_error() or "Not ready")
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: apply_config(
                live_asr=body.live_asr,
                final_asr=body.final_asr,
                live_mt=body.live_mt,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.websocket("/ws")
async def interpreter_socket(websocket: WebSocket) -> None:
    if not is_ready():
        await websocket.close(code=1013, reason="Models not ready")
        return

    await websocket.accept()
    store = get_store()
    state = store.create()
    session_id = state.session_id
    short_id = session_id[:8]
    metrics.session_start()
    logger.info(
        "Session started",
        extra={"event": "session_start", "session_id": short_id},
    )
    store.start_processor(session_id)

    stop_event = asyncio.Event()
    poll_task = asyncio.create_task(_poll_updates(websocket, session_id, stop_event))
    outcome = "disconnect"

    try:
        await _send_state(websocket, state, phase="listening")

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                samples = np.frombuffer(message["bytes"], dtype=np.int16)
                store.append_chunk(session_id, 16_000, samples)
                continue

            if "text" not in message:
                continue

            payload = json.loads(message["text"])
            msg_type = payload.get("type")

            if msg_type == "audio":
                raw = base64.b64decode(payload.get("data", ""))
                if not raw:
                    continue
                samples = np.frombuffer(raw, dtype=np.int16)
                sample_rate = int(payload.get("rate", 16_000))
                store.append_chunk(session_id, sample_rate, samples)
            elif msg_type == "stop":
                current = store.get(session_id)
                if current is None or current.audio.size == 0:
                    await websocket.send_json(
                        {
                            "type": "final",
                            "arabic": "",
                            "english": "",
                            "arabic_verified": "",
                            "arabic_provisional": "",
                            "english_verified": "",
                            "english_provisional": "",
                            "status": "No audio captured — speak, then stop.",
                            "duration_sec": 0.0,
                            "finalized": True,
                            "phase": "empty",
                        }
                    )
                    outcome = "complete"
                    break

                stop_event.set()
                poll_task.cancel()

                await _send_progress(
                    websocket,
                    "Finalizing: loading high-quality model…",
                    phase="finalize_load",
                )

                loop = asyncio.get_running_loop()
                t0 = time.perf_counter()
                fut = loop.run_in_executor(_executor, store.finalize, session_id)

                while not fut.done():
                    live = store.get(session_id)
                    status = (
                        live.status_message
                        if live is not None
                        else "Finalizing: re-transcribing…"
                    )
                    await _send_progress(websocket, status, phase="finalize")
                    await asyncio.wait({fut}, timeout=0.4)

                try:
                    final = fut.result()
                    latency = time.perf_counter() - t0
                    metrics.record_finalize(latency, ok=final is not None)
                    logger.info(
                        "Session finalized",
                        extra={
                            "event": "session_finalize",
                            "session_id": short_id,
                            "latency_sec": round(latency, 3),
                        },
                    )
                    if final is not None:
                        await _send_state(websocket, final, finalized=True, phase="final")
                    outcome = "complete"
                except Exception:
                    metrics.record_finalize(time.perf_counter() - t0, ok=False)
                    logger.exception(
                        "Finalize failed",
                        extra={"event": "session_finalize_error", "session_id": short_id},
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "status": "Finalize failed. Try again.",
                            "error": "Finalize failed. Try again.",
                            "phase": "error",
                        }
                    )
                    outcome = "fail"
                break
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected",
            extra={"event": "session_disconnect", "session_id": short_id},
        )
        outcome = "disconnect"
    except Exception:
        logger.exception(
            "WebSocket error",
            extra={"event": "session_error", "session_id": short_id},
        )
        outcome = "fail"
    finally:
        stop_event.set()
        poll_task.cancel()
        current = store.get(session_id)
        if current is not None:
            current.is_active = False
            store.remove(session_id)
        if outcome == "complete":
            metrics.session_complete()
        elif outcome == "fail":
            metrics.session_fail()
        else:
            metrics.session_disconnect()


__all__ = ["app"]
