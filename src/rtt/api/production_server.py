"""FastAPI WebSocket bridge for the production Next.js interpreter UI."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..ui.gradio_app import get_store

logger = logging.getLogger(__name__)

app = FastAPI(title="RTT Production API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rtt-api")


def _state_payload(state: Any, *, finalized: bool = False) -> dict[str, Any]:
    return {
        "type": "final" if finalized else "update",
        "arabic": state.arabic_text,
        "english": state.english_text,
        "status": state.status_message,
        "duration_sec": state.duration_sec(),
        "finalized": finalized,
    }


async def _send_state(websocket: WebSocket, state: Any, *, finalized: bool = False) -> None:
    await websocket.send_json(_state_payload(state, finalized=finalized))


async def _poll_updates(
    websocket: WebSocket,
    session_id: str,
    stop_event: asyncio.Event,
) -> None:
    store = get_store()
    last_arabic = ""
    last_english = ""
    last_status = ""

    while not stop_event.is_set():
        state = store.get(session_id)
        if state is None or not state.is_active:
            break

        if (
            state.arabic_text != last_arabic
            or state.english_text != last_english
            or state.status_message != last_status
        ):
            last_arabic = state.arabic_text
            last_english = state.english_text
            last_status = state.status_message
            await _send_state(websocket, state)

        await asyncio.sleep(0.25)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def interpreter_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    store = get_store()
    state = store.create()
    session_id = state.session_id
    store.start_processor(session_id)

    stop_event = asyncio.Event()
    poll_task = asyncio.create_task(_poll_updates(websocket, session_id, stop_event))

    try:
        await _send_state(websocket, state)

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
                            "status": "No audio captured.",
                            "duration_sec": 0.0,
                            "finalized": True,
                        }
                    )
                    break

                stop_event.set()
                poll_task.cancel()

                loop = asyncio.get_running_loop()
                final = await loop.run_in_executor(_executor, store.finalize, session_id)
                if final is not None:
                    await _send_state(websocket, final, finalized=True)
                break
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id[:8])
    except Exception:
        logger.exception("WebSocket error for session %s", session_id[:8])
    finally:
        stop_event.set()
        poll_task.cancel()
        current = store.get(session_id)
        if current is not None:
            current.is_active = False
            store.remove(session_id)


__all__ = ["app"]
