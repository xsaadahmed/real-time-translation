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
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..live.runtime import get_store, create_session
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
    global _executor
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rtt-api")
    configure_logging(
        json_logs=os.environ.get("RTT_JSON_LOGS", "1").strip().lower()
        in {"1", "true", "yes", "on"},
        level=os.environ.get("RTT_LOG_LEVEL", "INFO"),
    )
    # Always warm in the background so uvicorn can bind and answer /health
    # immediately. Blocking warmup here made the launcher think the port
    # never came up (models take far longer than the bind timeout).
    from .readiness import mark_not_ready

    if os.environ.get("RTT_FALLBACK_ONLY", "").lower() in {"1", "true", "yes"}:
        mark_not_ready("Live mode disabled; simulated preview is available.")
        warm_future = None
    else:
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
        if warm_future is not None and not warm_future.done():
            warm_future.cancel()
        logger.info("API shutting down", extra={"event": "api_shutdown"})
        _executor.shutdown(wait=False, cancel_futures=True)


#: How often the websocket checks session state for new text to push. This sits
#: on top of the live pipeline's own latency, so keep it well under it.
WS_POLL_SEC = float(os.environ.get("RTT_WS_POLL_SEC", "0.1"))
MAX_FRAME_BYTES = int(os.environ.get("RTT_MAX_FRAME_BYTES", str(256 * 1024)))
ALLOWED_RATES = {8_000, 16_000, 22_050, 24_000, 44_100, 48_000, 96_000}

app = FastAPI(title="RTT Production API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigUpdate(BaseModel):
    model_config = {"extra": "forbid"}
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
        "decoded_duration_sec": getattr(state, "decoded_duration_sec", 0.0),
        "decode_seconds": getattr(state, "decode_seconds", 0.0),
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
    last_processed_samples = -1

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
            or state.processed_samples != last_processed_samples
        )
        if changed:
            last_ar_verified = state.arabic_verified
            last_ar_provisional = state.arabic_provisional
            last_en_verified = state.english_verified
            last_en_provisional = state.english_provisional
            last_status = state.status_message
            last_processed_samples = state.processed_samples
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
async def put_config(body: ConfigUpdate, request: Request) -> dict[str, Any]:
    if request.client and request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(status_code=403, detail="Model controls are local-only.")
    if request.headers.get("origin") and request.headers["origin"] not in CORS_ORIGINS:
        raise HTTPException(status_code=403, detail="Model control origin is not allowed.")
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
    mode = websocket.query_params.get("mode", "live")
    if mode == "simulated":
        # The demo implementation is deliberately isolated from model imports.
        from .demo_server import interpreter_socket as scripted_socket
        await scripted_socket(websocket)
        return
    if mode != "live":
        await websocket.close(code=1008, reason="Unknown demo mode")
        return
    if not is_ready():
        await websocket.close(code=1013, reason="Models not ready")
        return

    await websocket.accept()
    try:
        store, state = create_session()
    except RuntimeError as exc:
        await websocket.send_json({"type": "error", "status": str(exc), "phase": "error"})
        await websocket.close(code=1013, reason=str(exc))
        return
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
                raw_bytes = message["bytes"]
                if len(raw_bytes) > MAX_FRAME_BYTES or len(raw_bytes) % 2:
                    await websocket.send_json({"type": "error", "status": "Invalid audio frame.", "phase": "error"})
                    outcome = "fail"
                    break
                samples = np.frombuffer(raw_bytes, dtype=np.int16)
                current = store.append_chunk(session_id, 16_000, samples)
                if current is not None and current.duration_sec() >= 30.0:
                    message = {"text": '{"type":"stop"}'}
                else:
                    continue

            if "text" not in message:
                continue

            try:
                payload = json.loads(message["text"])
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "status": "Invalid message.", "phase": "error"})
                outcome = "fail"
                break
            if not isinstance(payload, dict):
                await websocket.send_json({"type": "error", "status": "Invalid message.", "phase": "error"})
                outcome = "fail"
                break
            msg_type = payload.get("type")

            if msg_type == "audio":
                try:
                    raw = base64.b64decode(payload.get("data", ""), validate=True)
                except (ValueError, TypeError):
                    await websocket.send_json({"type": "error", "status": "Invalid audio frame.", "phase": "error"})
                    outcome = "fail"
                    break
                if not raw:
                    continue
                if len(raw) > MAX_FRAME_BYTES or len(raw) % 2:
                    await websocket.send_json({"type": "error", "status": "Audio frame too large.", "phase": "error"})
                    outcome = "fail"
                    break
                samples = np.frombuffer(raw, dtype=np.int16)
                sample_rate = int(payload.get("rate", 16_000))
                if sample_rate not in ALLOWED_RATES:
                    await websocket.send_json({"type": "error", "status": "Unsupported microphone sample rate.", "phase": "error"})
                    outcome = "fail"
                    break
                if samples.size > sample_rate:
                    await websocket.send_json({"type": "error", "status": "Audio frame exceeds one second.", "phase": "error"})
                    outcome = "fail"
                    break
                current = store.append_chunk(session_id, sample_rate, samples)
                if current is not None and current.duration_sec() >= 30.0:
                    msg_type = "stop"
                else:
                    continue
            if msg_type == "stop":
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
                    "Finalizing the full recording…",
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
            else:
                await websocket.send_json({"type": "error", "status": "Unknown message.", "phase": "error"})
                outcome = "fail"
                break
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
