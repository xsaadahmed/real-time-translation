"""Local web UI with real-time incremental transcription."""

from __future__ import annotations

import logging
import os
import threading

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from ..config import PipelineConfig, detect_device  # noqa: E402
from ..live import LiveSessionStore  # noqa: E402
from ..pipeline import TranslationPipeline, build_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

STREAM_INTERVAL_SEC = 0.5
STREAM_TIME_LIMIT_SEC = 600
UI_POLL_SEC = 0.3

_store: LiveSessionStore | None = None
_live_pipeline: TranslationPipeline | None = None
_final_pipeline: TranslationPipeline | None = None
_config = PipelineConfig()
_init_lock = threading.Lock()

_last_result: dict[str, str] = {
    "arabic": "",
    "english": "",
    "status": "Ready. Click the microphone to begin.",
}


def _ui_config(config: PipelineConfig | None = None) -> PipelineConfig:
    cfg = config or PipelineConfig()
    cfg.asr.vad_filter = False
    if os.environ.get("RTT_MT_BACKEND") is None and cfg.mt.backend == "marian":
        cfg.mt.backend = "nllb"
        cfg.mt.model_name = ""
    return cfg


def _live_config() -> PipelineConfig:
    """Fast ASR + Marian MT for incremental live updates on CPU."""
    cfg = _ui_config()
    default_live = "base" if detect_device() == "cpu" else "small"
    cfg.asr.model_size = os.environ.get("RTT_LIVE_ASR_MODEL", default_live)
    cfg.asr.beam_size = 1
    cfg.asr.condition_on_previous_text = False
    cfg.asr.initial_prompt = ""
    live_mt = os.environ.get("RTT_LIVE_MT_BACKEND", "marian")
    cfg.mt.backend = live_mt
    if live_mt == "marian":
        cfg.mt.model_name = ""
    cfg.mt.num_beams = min(cfg.mt.num_beams, 2)
    cfg.mt.max_new_tokens = min(cfg.mt.max_new_tokens, 128)
    return cfg


def _final_config() -> PipelineConfig:
    """Accurate ASR + higher beam MT for the final pass."""
    cfg = _ui_config()
    if not os.environ.get("RTT_ASR_MODEL"):
        cfg.asr.model_size = "medium"
    cfg.asr.beam_size = max(cfg.asr.beam_size, 5)
    cfg.asr.condition_on_previous_text = True
    cfg.mt.num_beams = max(cfg.mt.num_beams, 6)
    cfg.mt.max_new_tokens = max(cfg.mt.max_new_tokens, 512)
    return cfg


def _get_final_pipeline() -> TranslationPipeline:
    """Load the high-quality pipeline only when the user stops recording."""
    global _final_pipeline, _config
    if _final_pipeline is not None:
        return _final_pipeline

    with _init_lock:
        if _final_pipeline is None:
            final_cfg = _final_config()
            logger.info("Loading final ASR: %s (on stop)", final_cfg.asr.model_size)
            _final_pipeline = build_pipeline(final_cfg, include_tts=False)
            _final_pipeline.warmup()
    return _final_pipeline


def get_store() -> LiveSessionStore:
    global _store, _live_pipeline, _config
    if _store is not None:
        return _store

    with _init_lock:
        if _store is None:
            _config = _ui_config(_config)
            live_cfg = _live_config()
            logger.info(
                "Live ASR: %s (beam %d) | Live MT: %s",
                live_cfg.asr.model_size,
                live_cfg.asr.beam_size,
                live_cfg.mt.backend,
            )
            _live_pipeline = build_pipeline(live_cfg, include_tts=False)
            _store = LiveSessionStore(
                _live_pipeline, _get_final_pipeline, _config.second_opinion
            )
            logger.info("Warming up live pipeline…")
            _live_pipeline.warmup()
    return _store


def reset_store() -> None:
    """Drop cached pipelines so the next get_store() rebuilds from env."""
    global _store, _live_pipeline, _final_pipeline, _config
    with _init_lock:
        _store = None
        _live_pipeline = None
        _final_pipeline = None
        _config = PipelineConfig()


def active_session_count() -> int:
    store = _store
    if store is None:
        return 0
    return store.active_count()


def describe_runtime_config() -> dict[str, str]:
    live_cfg = _live_config()
    final_cfg = _final_config()
    return {
        "live_asr": live_cfg.asr.model_size,
        "live_mt": live_cfg.mt.backend,
        "final_asr": final_cfg.asr.model_size,
        "final_mt": final_cfg.mt.backend,
    }


def _display_for_session(session_id: str | None) -> tuple[str, str, str, str]:
    store = get_store()

    if not session_id:
        return "", _last_result["arabic"], _last_result["english"], _last_result["status"]

    state = store.get(session_id)
    if state is None:
        return session_id, _last_result["arabic"], _last_result["english"], _last_result["status"]

    _last_result["arabic"] = state.arabic_text
    _last_result["english"] = state.english_text
    _last_result["status"] = state.status_message

    return session_id, state.arabic_text, state.english_text, state.status_message


def on_start_recording():
    store = get_store()
    state = store.create()
    store.start_processor(state.session_id)
    _last_result["arabic"] = ""
    _last_result["english"] = ""
    _last_result["status"] = "**Recording…** Speak Arabic."
    logger.info("Recording started: %s", state.session_id[:8])
    return state.session_id, "", "", _last_result["status"]


def on_audio_stream(
    session_id: str | None,
    new_chunk: tuple[int, np.ndarray] | None,
):
    store = get_store()

    if session_id is None or store.get(session_id) is None:
        state = store.create()
        session_id = state.session_id
        store.start_processor(session_id)

    if new_chunk is not None:
        sample_rate, samples = new_chunk
        store.append_chunk(session_id, sample_rate, samples)

    return _display_for_session(session_id)


def on_stop_recording(session_id: str | None):
    store = get_store()

    if not session_id or store.get(session_id) is None:
        return session_id or "", _last_result["arabic"], _last_result["english"], _last_result["status"]

    state = store.get(session_id)
    if state is None:
        return session_id, _last_result["arabic"], _last_result["english"], _last_result["status"]

    if state.audio.size == 0:
        store.remove(session_id)
        return session_id, "", "", "No audio captured."

    if os.environ.get("RTT_DEBUG_AUDIO", "").strip().lower() in {"1", "true", "yes"}:
        store.save_debug(session_id, _config.output_dir)

    logger.info("Stop: final pass on %.2fs", state.duration_sec())
    final = store.finalize(session_id)
    if final is None:
        return session_id, "", "", "Processing failed."

    _last_result["arabic"] = final.arabic_text
    _last_result["english"] = final.english_text
    _last_result["status"] = f"**{final.status_message}**"
    return session_id, final.arabic_text, final.english_text, _last_result["status"]


def refresh_display(session_id: str | None):
    return _display_for_session(session_id)


def build_interface(config: PipelineConfig | None = None) -> gr.Blocks:
    global _config
    if config is not None:
        _config = _ui_config(config)

    live_cfg = _live_config()
    final_cfg = _final_config()

    with gr.Blocks(title="Arabic to English Live Translation") as demo:
        gr.Markdown(
            "# Arabic → English Live Translation\n"
            "Speak into the microphone. Text updates every few seconds while you talk.\n\n"
            f"**Live:** Whisper `{live_cfg.asr.model_size}` + {live_cfg.mt.backend} "
            f"(fast incremental) · "
            f"**Final:** Whisper `{final_cfg.asr.model_size}` + NLLB (full quality on stop)\n\n"
            "*On CPU, live text trails speech by ~3–8s. Stop recording for the "
            "high-quality final pass.*"
        )

        session_id_state = gr.State(value=None)

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Microphone — click to speak",
                )
            with gr.Column():
                arabic_box = gr.Textbox(
                    label="Arabic transcript (live)",
                    lines=10,
                    rtl=True,
                    interactive=False,
                )
                english_box = gr.Textbox(
                    label="English translation (live)",
                    lines=10,
                    interactive=False,
                )

        status = gr.Markdown("Ready. Click the microphone to begin.")

        poll_timer = gr.Timer(value=UI_POLL_SEC, active=True)

        audio_input.start_recording(
            fn=on_start_recording,
            outputs=[session_id_state, arabic_box, english_box, status],
            queue=False,
        )

        audio_input.stream(
            fn=on_audio_stream,
            inputs=[session_id_state, audio_input],
            outputs=[session_id_state, arabic_box, english_box, status],
            stream_every=STREAM_INTERVAL_SEC,
            time_limit=STREAM_TIME_LIMIT_SEC,
            queue=False,
            show_progress="hidden",
        )

        audio_input.stop_recording(
            fn=on_stop_recording,
            inputs=[session_id_state],
            outputs=[session_id_state, arabic_box, english_box, status],
            show_progress="full",
        )

        poll_timer.tick(
            fn=refresh_display,
            inputs=[session_id_state],
            outputs=[session_id_state, arabic_box, english_box, status],
            queue=False,
            show_progress="hidden",
        )

    return demo


def launch(config: PipelineConfig | None = None, share: bool = False, port: int = 7860) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if config is not None:
        global _config
        _config = _ui_config(config)
    else:
        _config = _ui_config(_config)

    logger.info("Pre-loading live models…")
    get_store()

    demo = build_interface(_config)
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="127.0.0.1", server_port=port, share=share)


__all__ = ["build_interface", "get_store", "launch"]
