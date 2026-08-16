# Real-Time Arabic → English Speech Translation

Live speech interpretation: **Arabic audio in → Arabic transcript → English translation → English speech out.**

This repo is the working foundation for a **real-time simultaneous interpreter** with speculative
on-screen predictions and adaptive commitment. What runs today is a modular cascade (ASR → MT →
TTS) with a production Next.js UI and a Gradio dev UI. Cloud APIs and hosted models are fine
going forward when they improve latency or quality.

**Not in focus right now:** voice cloning (standard TTS is enough for the current milestone).

---

## Target architecture

Where this is headed — the current code is structured so these layers can be added without a
rewrite.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Production UI (Next.js)          Dev UI (Gradio)                         │
│  word-by-word display,          mic + raw transcript panels               │
│  speculative candidates                                                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ WebSocket / streaming audio
┌───────────────────────────────▼─────────────────────────────────────────┐
│  Session layer (live/session.py)                                          │
│  incremental audio buffer · dual pipeline (fast live + accurate final)    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Streaming ASR │     │ Anticipation /  │     │ MT              │
│ (Whisper now, │────▶│ commitment      │────▶│ Marian / NLLB / │
│  streaming    │     │ engine (future) │     │ LLM APIs later) │
│  models later)│     │ speculative +   │     │                 │
│               │     │ committed spans │     │                 │
└───────────────┘     └─────────────────┘     └────────┬────────┘
                                                       ▼
                                              ┌─────────────────┐
                                              │ TTS (optional)  │
                                              │ SAPI / neural   │
                                              └─────────────────┘
```

**Planned capabilities (in order of priority):**

1. **Low-latency live path** — streaming ASR on short windows, fast MT preview while the user speaks.
2. **Anticipation UI** — show multiple candidate continuations; commit words as confidence grows
   (production UI already has the visual pattern; backend commitment policy comes next).
3. **Accurate final pass** — re-transcribe the full utterance on stop with a larger ASR model.
4. **Better MT** — Levantine-aware models today (`apc_Arab`); LLM or cloud translation when needed.
5. **TTS** — intelligible English output; voice cloning is explicitly **de-prioritized**.

---

## What works today

| Piece | Status |
| --- | --- |
| Live mic → Arabic + English (incremental) | ✅ fast path (small/base Whisper + Marian) |
| Final pass on stop | ✅ medium Whisper + NLLB (Levantine `apc_Arab`) |
| Production interpreter UI | ✅ `run_production.py` (Next.js + FastAPI WebSocket) |
| Dev / debug UI | ✅ `run_ui.py` (Gradio) |
| Pluggable ASR / MT / TTS | ✅ swap via config or env vars |
| Speculative word animation (UI only) | ✅ cosmetic candidates until commitment engine lands |

Models are downloaded from Hugging Face on first run and cached under `models/`. Local inference
is the default, but **network-backed components are allowed** as the system evolves.

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# One-time model download
python scripts/download_models.py

# Smoke test
python scripts/smoke_test.py

# Dev UI — http://127.0.0.1:7860
python run_ui.py

# Production UI — http://127.0.0.1:3000 (requires Node.js)
python run_production.py
```

On macOS/Linux use `source .venv/bin/activate`. SAPI5 TTS is Windows-only; use Kokoro or Piper
from `requirements-optional.txt` on other platforms.

### Production UI

`python run_production.py` starts the FastAPI backend and Next.js frontend. It auto-picks free
ports if defaults are busy, clears stale dev servers, and writes the WebSocket URL for the UI.

Click **Start** to listen. Arabic appears above; English animates below. **Stop** runs the
high-quality final pass.

### Live tuning

```powershell
$env:RTT_LIVE_ASR_MODEL="tiny"       # faster live preview
$env:RTT_LIVE_MT_BACKEND="marian"    # fast live MT
$env:RTT_ASR_MODEL="large-v3-turbo" # best final ASR (slow on CPU)
$env:RTT_NLLB_SOURCE_LANG="apc_Arab" # Levantine Arabic (default)
```

---

## Models (current defaults)

| Stage | Default | Notes |
| --- | --- | --- |
| ASR (live) | Whisper `base` / `tiny`, int8 | Incremental chunks |
| ASR (final) | Whisper `medium` | Full buffer on stop |
| MT (live) | Marian `opus-mt-ar-en` | Fast CPU |
| MT (final) | NLLB distilled 600M, `apc_Arab` | Levantine source |
| TTS | Windows SAPI5 | Optional; not required for interpreter UI |

CUDA is used automatically when available.

---

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RTT_ASR_MODEL` | `medium` | Final-pass Whisper size |
| `RTT_LIVE_ASR_MODEL` | `base` (CPU) | Live Whisper size |
| `RTT_NLLB_SOURCE_LANG` | `apc_Arab` | NLLB dialect code |
| `RTT_MT_BACKEND` | `nllb` | `marian` / `nllb` |
| `RTT_ASR_VAD` | `0` | VAD filter (off for live mic) |
| `RTT_TTS_BACKEND` | `auto` | TTS backend selection |
| `RTT_OUTPUT_DIR` | `outputs/` | Output WAV directory |

See `src/rtt/config.py` for the full list of `RTT_*` overrides.

---

## Project layout

```
run_production.py         Production Next.js UI + WebSocket API
run_ui.py                 Gradio dev UI
production-ui/            Next.js interpreter interface
src/rtt/
  config.py               Pipeline and model settings
  pipeline.py             ASR → MT → TTS orchestration
  live/session.py         Live mic sessions, incremental ASR
  api/production_server.py FastAPI WebSocket bridge
  asr/  mt/  tts/         Pluggable stage backends
  ui/gradio_app.py        Gradio app
  text.py                 Arabic chunking, incremental merge
scripts/
  download_models.py      Pre-fetch Hugging Face weights
  smoke_test.py           End-to-end smoke test
```

Audio I/O uses `soundfile` / PyAV — no system `ffmpeg` binary required.

---

## Extension points

- **Streaming ASR** — `asr.base.ASREngine` + incremental `transcribe_chunk`; segments already
  carry timestamps.
- **Commitment engine** — between ASR and MT; emits committed vs speculative spans for the UI.
- **LLM / cloud MT** — implement `mt.base.Translator` and register in `mt/__init__.py`.
- **Arabic structure rules** — `text.py` for VSO gaps, *iḍāfa*, numerals, TAM particles.
- **TTS** — optional `tts/` backends; cloning is not a current goal.

---

## Troubleshooting

**No speech detected** — speak closer to the mic; keep `RTT_ASR_VAD=0` for live capture.

**Poor Arabic transcription** — try `RTT_ASR_MODEL=large-v3-turbo` for the final pass.

**Production UI won't start** — ensure Node.js is installed; `run_production.py` clears stale
ports and Next.js locks automatically.

**Wrong WebSocket port** — restart via `run_production.py`; it writes `production-ui/public/runtime-config.json`.
