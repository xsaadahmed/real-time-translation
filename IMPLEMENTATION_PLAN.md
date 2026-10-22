# GITEX demo rescue — implementation plan

Audit: 21 October 2026. Base: `88bf893dcb83a2cadcbfa5b3c27d5f6b41f612ac`.

**Status: implementation completed for core demo safeguards; awaiting Sol final review and live rehearsal.**

### Live-accuracy amendment — 21 October 2026

The user observed poor drafts but good Stop results. Sol reproduced this with public Arabic speech and replaced the production incremental algorithm. This amendment supersedes the earlier plan to retain fragment reconciliation in the live API.

- Production now uses `src/rtt/live/context_session.py`: bounded full-recording snapshots, fully revisable Arabic/English drafts, complete-source translation, four seconds of initial context and automatic draining of partial windows after a pause. Old fragment/reconciliation modules remain available to existing research tests, but are not the production API path.
- `src/rtt/live/runtime.py` owns one reviewed CPU model profile: Whisper-small/int8, beam 3, four CPU threads, one worker; Marian ar-en, beam 2 and four torch threads. It publishes the store after warmup and avoids importing Gradio in the API.
- Identical fully processed input is reused on Stop. New tail audio triggers a full pass. Native inference and final result publication share one lock; disconnected/in-flight work retains admission.
- The browser flushes the final PCM tail, cancels late microphone permission results and exits a failed/timed-out finalization. It displays readable, explicitly revisable drafts. The API reports processed-audio progress even when no speech/text changes and finalizes at the 30-second cap.
- Daily UI startup uses the prepared production build, reducing development-server overhead.
- Automatic checks: `scripts/validate_live_accuracy.py` for public-source baseline comparison; `scripts/validate_live_socket.py` for microphone-speed socket replay, noise, silence, repeated sessions and the recording cap. No manual playback is needed.

See the appended accuracy validation for executed results and limitations. General exhibition readiness is still limited by the original outstanding security/licensing/lifecycle gates; this amendment does not declare them completed.

## Checkout and preservation

Implement in `C:\Users\xsaad\Documents\real-time-translation`. This existing checkout matches the remote HEAD audited here. Its existing `production-ui/next-env.d.ts` change points at `.next/dev/types/`; preserve it. No application source was changed during the audit. A clean comparison checkout and scratch probes are under this chat's `work/`. Do not push, deploy, reset the existing checkout, or touch existing recordings/model caches unnecessarily.

## Actual architecture and evidence

- `run_production.py`: starts local uvicorn plus **Next dev**, writes `public/runtime-config.json`, waits for live models. It does not load `.env` itself. `run_demo.py` reuses its startup helpers with a scripted API.
- `production-ui/hooks/use-interpreter-stream.ts`: browser mono PCM16 in base64 JSON, about 500 ms per chunk; WebSocket transport; separate REST readiness/config calls. No English audio playback.
- `src/rtt/api/production_server.py`: creates a session immediately on each socket connection, polls mutable state, finalizes through a two-worker executor.
- `src/rtt/ui/gradio_app.py`: also owns production model/store initialization. CPU live defaults to Whisper **base**, Marian, beam 1; Stop lazily loads Whisper **medium** and NLLB. Importing the API also imports Gradio.
- `src/rtt/live/session.py`: per-session raw audio retained in float32; rolling context inference, heuristic stable/provisional text, shared inference lock; full-buffer second pass on Stop.
- `pipeline.py`, `asr/faster_whisper_asr.py`, `mt/huggingface.py`, `audio.py`: real inference and resampling exist. Research modules exist but do not constitute the README's trained probabilistic simultaneous interpreter.
- Docker uses Python 3.12, includes optional morphology/data, and Caddy exposes `/config`, `/metrics`, `/ws`. Default host port mappings expose all interfaces. Local launcher binds loopback.
- Existing tests cover text reconciliation, guards, VAD decisions, harvesting, branching, second opinion, Python compatibility and metrics. They do not exercise the production WebSocket/microphone lifecycle or launcher's process ownership.
- Recent commits: `88bf893` adds the scripted demo; `70830a6` repairs hypothesis alignment; `b1da9f9` repairs the earlier processing-cursor audio loss. Do not re-fix those repaired bugs. A distinct `merge_audio` bug remains.

## Prioritized findings

| Priority | Finding, evidence and disposition |
|---|---|
| P0 | Default Stop pipeline selects NLLB-200 with CC-BY-NC-4.0 weights. Do not use it for this company exhibition absent separately documented permission. Use Marian in both paths; reject NLLB/custom unreviewed model overrides in the exhibition profile. |
| P0 if exposed | `/ws` and `PUT /config` have no authentication, origin checks or admission controls. CORS does not secure WebSockets. Docker/Caddy maps public-facing ports. Choose local-only service, explicit WebSocket Origin allowlist, loopback Host restrictions and bounded input/session resources. Disable config mutation for the exhibition. Public deployment remains blocked/deferred. |
| P1 | **Measured silence hallucination:** small/int8 ASR with the wrapper's default prompt returned `مرحبا، أنا من لبنان.` for 2 s of digital silence. `_initial_prompt()` substitutes a Lebanon prompt for an empty configured prompt. A runtime-only neutral-prompt probe returned empty text, with and without VAD. Remove this prompt fallback; skip digital silence and handle no-speech input. Noise robustness remains unverified. |
| P1 | **Reproduced audio loss:** `merge_audio` replaces 8,000 existing samples with a 9,000-sample incoming chunk, instead of retaining 17,000; identical consecutive 8,000-sample chunks retain only 8,000 instead of 16,000. WebSocket chunks are deltas. Use explicit append semantics; do not infer cumulative audio from size or waveform similarity. |
| P1 | Stop first-loads medium/NLLB and retranscribes all retained audio; `/ready` only validates the live pipeline. This can produce a long stall or download failure after a seemingly successful live session. Reuse one preloaded pipeline for live and final processing. |
| P1 | `audio` grows without a bound and every append copies it; no wall-clock/idle cap or concurrent-session cap. Final-pass tokenizer truncates at 512 tokens. Limit exhibition recordings to 30 s, retain a bounded buffer, finalize at the cap, limit one live job through completion. |
| P1 | Reconnect creates a new backend session. Frontend keeps old text only temporarily, drops captured audio while disconnected and then replaces output with a new session. Automatic resume is misleading. Stop capture on disconnect, retain visible interrupted text and require explicit new session. |
| P1 | Stop discards unsent browser PCM (up to roughly one chunk). Capture lacks track-ended handling, cancellation generation guards and explicit backpressure. Finalization socket close can leave a permanent Processing state because `runningRef` is already false. Flush before Stop, use explicit cancellation/lifecycle handling and bounded finalization. |
| P1 | Simulation ignores mic audio but the common hook still requests/captures the microphone. Config/runtime carries `mode=demo`, yet the UI never renders it. Script statuses refer to risk scoring/verification as if implemented. Show persistent `SIMULATED — scripted output; microphone off`; do not request mic; remove unsupported technology claims from script statuses. |
| P1 | `_cleanup_stale_api` kills anything listening in a 60-port range. Next lock cleanup can kill a stored PID. UI startup reports success after timeout even without a healthy page; npm child cleanup may leave descendants. Remove indiscriminate cleanup, track owned children, fail clearly on startup error. |
| P1 | `get_store()` publishes `_store` before warmup succeeds; failed warmup can leave a partially initialized cached store. Publish only after successful warmup and expose live loading/error independently from fallback availability. |
| P1 | `apply_config` mutates environment before checking for active sessions. Reload requests also lack serialization around the whole transaction. Disable mutation in exhibition profile; if preserving research configuration, validate all fields and activity before changing any state. |
| P1 | Speech excerpts are logged at INFO in live/final paths. Remove content from default logs; keep timings/counts only; disable audio saving/research logs in exhibition profile. |
| P1, historical evidence | `70830a6` and config comments report ~0.51–0.62 WER and ~3.4–3.7 s median lag on a 52.9 s sample. This supports concern about the old base-model path; these are author-reported results, not independently reproduced measurements or evidence about Gulf Arabic. No reference Arabic audio is tracked. |
| P2 | `apc_Arab` is the configured North Levantine NLLB source, inappropriate as a blanket MSA/Gulf default. Exhibition bypasses NLLB; change research default to `arb_Arab` and document dialect selection if retained. |
| P2 | Build explicitly skips TypeScript errors. Current clean `tsc --noEmit` passes, so this is a missing guard, not a proven current type defect. Enable build type checking. |
| P2 | Google fonts are fetched at build time; fallback currently needs all core Python dependencies through shared imports. Provide an offline-safe prepared build and a small fallback requirements file. |
| Deferred | TTS exists in library/Gradio/file paths but production sockets send text only. No completed live English speech experience. Disable any implication of speech output; no TTS work needed. |
| Deferred | Qwen/LoRA, trained risk head, dual-ASR, cloned voice, future sampling, second-opinion inference, multi-user service, production authentication and cloud infrastructure. |

## Chosen architecture (decisions, not implemented features)

Keep the existing Next UI and API protocol. One local API serves both modes, selected per session, with no automatic covert mode change.

**Live:** browser mic → mono PCM → bounded 16 kHz buffer → faster-whisper `small`, CPU int8, Arabic language, beam 3, temperature 0, no fabricated initial prompt → `Helsinki-NLP/opus-mt-ar-en`, CPU, beam 2 → Arabic/English text. Use the same warmed model objects on Stop; do not mutate their shared settings during inference. Keep current incremental scheduling/context initially; label stable spans **Stable draft**, provisional spans **Updating**, and final output **Final pass**. Stable drafts may be revised on Stop; they are not independently verified. Keep live VAD off initially because old CPU measurements did not justify its scheduling overhead. Add a cheap digital-silence check before ASR; a nonzero noise floor must not be marketed as speech detection. Do not claim measured Gulf accuracy.

`small` is cached on the audited Ryzen 7 / 16 GB / no NVIDIA GPU machine; offline load was exercised. Its real speech accuracy/latency is still an acceptance gate. If it fails rehearsal, use simulated mode for the presentation rather than silently downgrading to tiny. No expensive model search is authorized.

**Simulated:** reuse `demo_server.py`'s script and verified/provisional wire fields, visibly relabeled throughout as scripted interface preview. No microphone or ASR/MT dependency required. This is not prerecorded speech, cached inference or a measured latency demo. Stop should finalize only material reached so far; automatic script completion must send a terminal final event and close cleanly.

**Serving:** add `/ws?mode=live|simulated` dispatch to the existing API, reuse an extracted demo playback helper, remove eager Gradio import from fallback/API startup. `/health` describes service health; `/ready` continues to describe live readiness; `/config` adds explicit available modes, live status/error and fixed model names. UI can select simulation while live initialization fails. Start UI and API without waiting for live models. `--fallback-only` skips live imports/warmup entirely. Do not change Gradio into the product or build a second frontend.

**Limits:** 30 s audio per live session (480,000 float32 samples = 1,920,000 bytes), 60 s recording wall time, 15 s without audio; a configurable 45 s finalization UI deadline. Reject malformed JSON/base64, nonobject messages, unknown commands, odd PCM length, unsupported sample rates, overlarge frames and oversized decoded chunks before expensive resampling. Allow common browser rates 8/16/22.05/24/44.1/48/96 kHz, at most 1 s per chunk and 256 KiB decoded PCM; set uvicorn WebSocket frame max 512 KiB and a bounded queue. At the audio limit finalize the accepted prefix explicitly. On failure retain text with an interrupted/error label. A timeout cannot cancel native inference: retain live admission until the underlying worker exits; fallback must remain available. No queued overlapping live inferences.

**Exposure:** bind local services to `127.0.0.1`; allow only actual launcher-selected UI origins, including port fallback, in HTTP and WebSocket checks. Restrict Host to loopback names. Reject unknown browser origins; CLI tests may use explicitly configured test origins. Disable `PUT /config` in exhibition mode. Docker host mappings must use `127.0.0.1:`; container internal `0.0.0.0` binding is fine. Do not publish via a tunnel.

## Milestones, exact change scope and acceptance

### M1 — fix data correctness and model profile

Files: `src/rtt/live/session.py`, `src/rtt/asr/faster_whisper_asr.py`, `src/rtt/ui/gradio_app.py`, `src/rtt/config.py`, `.env.example`; new `tests/test_live_audio.py`, `tests/test_demo_models.py`.

- Add an explicit chunk-only append path. Preserve Gradio compatibility with explicit API choices instead of heuristic size detection. Lock audio snapshot/append transitions briefly; never hold the state lock through inference.
- Remove Lebanon prompt fallback; empty prompt means no prompt. Skip digital silence before decoding. Build and warm a local exhibition store transactionally; final getter returns its already-loaded pipeline. Enforce reviewed models in exhibition configuration; ensure no leftover env can silently select NLLB.
- Remove transcript content logs by default. Keep research channels off. Do not rewrite reconciliation or unrelated research modules.
- Acceptance: exact sample preservation across varied and repeated chunks; zero/empty input produces no text and no unnecessary inference; failure then retry initializes a fresh valid store; Stop performs no download/model construction. Existing tests pass.

### M2 — bounded server lifecycle and truthful fallback

Files: `src/rtt/api/production_server.py`, `src/rtt/api/demo_server.py`, `src/rtt/api/readiness.py`, `src/rtt/api/settings.py`, `src/rtt/live/session.py`; new `tests/test_demo_api.py`, `tests/test_api_limits.py`.

- Extract reusable script playback and route session mode through the single server; no eager Gradio/model import in fallback-only startup. Separate service health and live readiness. No silent switch to scripted output.
- Implement the limits/origin/Host/admission rules above, automatic limit handling, structured errors and cleanup. Serialize sends or stop/await poll task before final messages. Hold live admission through outstanding native work after disconnect/timeout; release exactly once. Prevent a late increment overwriting a final result.
- Disable config writes in exhibition mode. If preserving non-exhibition updates, make them validate-before-mutate and transactionally serialized.
- Acceptance: fallback-only starts without torch/faster-whisper/Gradio; script start/stop/repeat completes without mic; malformed and oversized input cannot crash API; second live client gets Busy; limits bound allocation; disconnect/Stop races cannot create extra workers or leak slots; untrusted origins rejected; failed model warmup leaves simulation usable.

### M3 — presenter UI lifecycle and focused polish

Read `production-ui/AGENTS.md` and relevant installed `node_modules/next/dist/docs/` before changes.

Files: `production-ui/hooks/use-interpreter-stream.ts`, `components/interpreter/{translation-canvas,session-controls,session-status,recording-indicator,source-text,translation-text}.tsx`, `app/globals.css`, `app/layout.tsx`, `next.config.mjs`. Add `production-ui/tests/session.test.*` only if using the existing toolchain/Node test runner; avoid a large testing framework.

- Replace model selectors with Live / Simulated mode controls, persistent truthful banner, readable Arabic/English labels, larger Start/Stop and status text. Maintain existing design. Expose actual mic-active state, not just WebSocket-open state.
- Simulation never calls getUserMedia or sends audio. Live flushes buffered PCM before sending Stop; resume AudioContext; handle mic denial/track-ended; cap outgoing bufferedAmount. Generation/cancellation guard must stop tracks from late permission resolution after Reset/Stop/mode switch/unmount.
- On disconnect, stop mic and preserve interrupted display; no fake session resumption. Finalization failure/close/timeout must clear busy UI and offer explicit fallback/new session. Start cannot launch a second session while one is connecting/finalizing. Mode switch disposes old socket and capture.
- Remove unsupported risk/accuracy/simultaneous speech claims from demo statuses. Use system fonts or properly licensed bundled fonts to remove build-time font downloads. Enable type checks.
- Acceptance: denied mic does not break simulated mode; mic indicator follows real tracks; no audio after stop/switch; finalization always reaches success/error; zero/repeated sessions behave correctly; no stale callback can alter a later session; large-screen text remains clear and RTL correct; build and tsc pass. Browser lifecycle checks remain mandatory even if unit tests pass.

### M4 — one-window launch and offline preparation

Files: `run_production.py`, `run_demo.py`, `scripts/download_models.py`, `docker-compose.yml`, `docker-compose.dev.yml`, `docker/Caddyfile`, `.env.example`, `README.md`; new `launch_demo.ps1`, `requirements-demo.txt`, `tests/test_demo_launcher.py`. Avoid Docker image redesign; optional research image remains outside recommended path.

- `launch_demo.ps1` uses repo `.venv` Python 3.12 or prints exact setup instructions; never accidentally use Python 3.14. Entrypoint for daily operation is `powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_demo.ps1`.
- Provide explicit `-Prepare` (one-time dependency install, selected model cache and production UI build) and `-FallbackOnly` (no model warmup, minimal requirements) switches. Document prerequisites and internet needed for preparation. Normal launch performs no package installs/model downloads. Missing live packages/models leave the fallback usable and display an actionable error.
- Use one package manager and npm lock (`npm ci` during Prepare); build once, serve `next start` on loopback. No repeated downloads during exhibition. Maintain legacy Python entrypoints as thin compatible wrappers.
- Remove broad port/PID killing and dev lock deletion. Use recorded ownership of the current launch's process tree; busy ports may be skipped safely. Check real UI HTTP health before printing Ready; nonzero exit on failure. Print exact URL, live loading/result, fallback availability and Ctrl+C behavior. Stop only owned processes, including Node descendants, and prevent duplicate launcher instances in the same checkout.
- Acceptance: setup/launch use Python 3.12; missing npm/model/build produce readable errors; fallback works without models/internet/microphone; unrelated port sentinel survives; duplicate launch rejected; Ctrl+C leaves no owned listeners; startup does not claim health after timeout; production Docker mappings are loopback only.

### M5 — acceptance, runbook and Sol review

Files: update `DEMO_RUNBOOK.md`, `DEMO_VALIDATION.md`, `IMPLEMENTATION_PLAN.md`, `TERRA_HANDOFF.md`; add `THIRD_PARTY_NOTICES.md` and a small validation script if useful. Do not introduce a corpus/download service.

- Run tests below once after relevant changes. Rehearse three short MSA phrases and one Gulf phrase with actual microphone/reference audio, plus silence/noise. Use authorized local speech recordings only. Measure warm model load, first display, end-of-speech to final output, and normalized Arabic WER with the existing scorer; record audio origin and reference. Human-check English meaning. Synthetic audio must be labeled as such and cannot establish noisy Gulf-microphone quality.
- Target for the rehearsed 5–10 s MSA phrases: meaningful Arabic/English, final output within 10 s after Stop, no invented text on digital silence, no hangs. This is a **proposed rehearsal threshold**, not a measured performance claim. If missed, record it and recommend simulated mode for the show. Do not silently alter models or claim success.
- Run at least 20 short start/stop cycles plus a 10-minute sequence of bounded sessions, with session/worker/audio counters and RSS snapshots. Large model allocations may remain cached; raw audio/session count must return to zero, no accumulating workers, no growing buffer beyond the cap. Accelerated fake-inference tests verify bounds but do not establish real compute stability.
- Acceptance: presenter runbook tested verbatim, mode always obvious, no public listeners or NC-model load, no retained speech by default, baseline regression checks pass. Record blocked hardware checks honestly. Sol reviews final diff, logs and evidence after Terra; do not self-certify that review.

## Commands

From the implementation checkout in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
.\.venv\Scripts\python.exe -m pytest tests/test_live_audio.py tests/test_demo_models.py
.\.venv\Scripts\python.exe -m pytest tests/test_demo_api.py tests/test_api_limits.py tests/test_demo_launcher.py
Push-Location production-ui
npm.cmd ci
npx.cmd tsc --noEmit
npm.cmd run build
Pop-Location
git diff --check
```

New test filenames above are requirements, not files that already exist. Run the frontend commands after dependency/source changes, not repeatedly without cause. Use fake ASR/MT for routine lifecycle tests; opt in explicitly to real model checks. Avoid running existing launchers during audit: their port cleanup is unsafe.

## Licensing and source rights

Reviewed primary model cards on 21 October 2026:

- [NLLB distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M): CC-BY-NC-4.0; excluded from exhibition profile. No company-use permission inferred.
- [Marian ar-en](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en): Apache-2.0 metadata. Use this exact model, retain applicable notices.
- [Converted Whisper small](https://huggingface.co/Systran/faster-whisper-small): MIT metadata; [faster-whisper code](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) and [Whisper](https://github.com/openai/whisper#license) are MIT. Preserve notices in any distribution.

This repository has no tracked LICENSE/NOTICE identified. The user's instruction authorizes edits but does not establish ownership or a right to redistribute company/source assets. Do not invent a source license. Before distributing an exhibition bundle, obtain owner confirmation and inventory **installed versions**, transitive notices and any fonts/audio assets. Preliminary installed metadata confirms faster-whisper MIT and transformers Apache-2.0; this is not a complete dependency license audit. Torch/Next/React/FastAPI and bundled binary notices still require inventory in M5. Avoid optional TTS/model weights entirely for the demo. No legal clearance is claimed.

## Remaining risks and scope boundary

Actual Arabic microphone accuracy, Gulf speech and ambient noise remain unverified. Small-model CPU latency may still exceed presentation needs. Scripted mode is the reliability backstop and must always disclose that it is simulated. Windows process supervision and microphone cancellation need real lifecycle tests, not code inspection alone. A functioning rescue implementation and final Sol QA have **not** been delivered in this audit-only fallback phase.
