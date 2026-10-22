# Real-Time Arabic → English Speech Translation

## How to run

**Windows / PowerShell — from the repository folder:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_demo.ps1
```

Open the **UI address printed by the launcher**, normally **http://127.0.0.1:3000**. Keep the window open, wait for models to become ready, select **Live**, press **Start**, and allow microphone access. Press **Stop** to finalize. Stop services with **Ctrl+C** in the launcher window.

**First time on a new machine:** install Python **3.12** and Node.js **22 or newer** with npm, then run:

```powershell
git clone https://github.com/xsaadahmed/real-time-translation.git
cd real-time-translation
py -3.12 -m venv .venv
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_demo.ps1 -Prepare
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_demo.ps1
```

Preparation installs dependencies, downloads Whisper Small and Marian, and builds the Next.js interface. It needs internet and sufficient disk space. Repeat `-Prepare` after pulling application changes to rebuild the interface. Daily launch uses the prepared build without reinstalling packages. Missing model weights can still trigger downloads; prepare before going offline.

**Emergency scripted preview:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_demo.ps1 -FallbackOnly
```

This starts explicitly labeled simulated output without loading speech models. It still requires installed dependencies and the prepared frontend. It does **not** recognize microphone speech.

## What runs today

A local Arabic speech recognition and English **text** translation demo. Next.js captures microphone audio; FastAPI transports audio and results over WebSockets. Actual recognition uses faster-whisper, followed by Marian Arabic-to-English translation.

| Mode | Input | Output |
| --- | --- | --- |
| **Live** | Browser microphone | Actual model-generated Arabic transcription and English translation |
| **Simulated** | Prepared script, no microphone | Predictable, visibly labeled interface demonstration |

Live text is a **revisable draft** until Stop. Earlier words and translations can change as more audio arrives. This demo does not provide simultaneous English speech output.

## Giving a live demonstration

1. Use a close-talk headset in a quiet location. Check the selected microphone in Windows and the browser.
2. Select **Live**, wait for readiness, and choose a speech model before starting.
3. Press **Start** and speak Arabic in short, natural sentences. Allow several seconds for the first draft.
4. Pause occasionally to let decoding catch up. Earlier wording can correct itself without pressing Stop.
5. Press **Stop** for the complete-recording final pass. Start a new session for the next example.

Recordings finalize automatically at **30 seconds**. Only one live session is admitted at a time, including outstanding inference work. A disconnected session is interrupted; reconnecting does not restore its audio context.

See [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) for presenter phrases and recovery steps.

## Speech model dropdown

The compact **Speech model** dropdown changes the actual ASR used for both live drafts and finalization. Marian remains the translator.

| Choice | Practical use |
| --- | --- |
| Tiny | Lowest compute cost; Arabic recognition may be insufficient |
| Base | Lightweight alternative; validate accuracy before presenting |
| **Small** | Default and recommended starting point on the tested CPU laptop |
| Medium | More compute and memory; potentially better recognition with slower updates |
| Large turbo | Larger option; not hardware-validated here |
| Large v3 | Larger option; not hardware-validated here |

Finish the current session before switching. The interface displays loading status and disables recording during a change. Previous ASR weights are released before loading the new model; the translator stays loaded. Failure attempts to restore the previous cached ASR.

An uncached model downloads on first selection. Load intended exhibition models beforehand. A restart returns to **Small**. Larger models are not guaranteed to improve your specific speech.

The exhibition profile uses **CPU int8 ASR**, four CPU threads, one ASR worker, and beam size 3. The dropdown changes size, not device or compute settings. AMD integrated graphics are not used by this profile. On a 16 GB laptop, reduce background memory use before trying larger models.

## Live architecture

```text
Browser microphone
    → WebSocket PCM audio
    → bounded in-memory recording
    → Whisper Arabic transcription of the complete recording so far
    → Marian translation of the complete recognized Arabic
    → replaceable Arabic and English drafts
    → final full-context pass on Stop
```

The production path is in `src/rtt/live/context_session.py` and `src/rtt/live/runtime.py`.

- Initial decoding normally starts after approximately four seconds of audio, plus processing time. A short phrase followed by quiet can trigger sooner.
- Updates catch up using the latest context rather than queuing stale audio cuts.
- Whole-context drafts replace previous text instead of permanently stitching uncertain short fragments.
- One quiet-tail recheck is retained, then repeated silence can reuse the draft without repeated inference. This is an energy heuristic, not noise cancellation.
- Live and Stop share one loaded ASR/translation pipeline. Speech synthesis and research channels are excluded from this path.

This favors coherent short demonstrations. Processing time grows with recording length; continuous low-latency interpretation of long speeches is outside the current scope.

## Executed checks and limitations

The inspected laptop has a Ryzen 7 8840HS, 16 GB RAM, and AMD integrated graphics. Tests used local models, without paid APIs.

| Check | Observed result |
| --- | --- |
| Python regression suite | **80 passed, 4 skipped**; optional morphology checks skipped |
| Next.js production build | Passed with TypeScript checking enabled |
| Small, three clean MSA recordings / 53 reference words | Normalized live WER improved from 64.2% to 11.3% in the before/after architecture/configuration comparison |
| Small, paced clean-audio WebSocket replay | About **7.2 seconds** to first update on one sample |
| Medium, one clean MSA sample | **0 normalized word errors**, about **11.9 seconds** to first update |
| Actual model switching | Medium loaded, Small restored; active-session changes rejected |
| Empty sessions and bounds | 20 empty start/stop cycles and the 30-second cap checked |
| Synthetic noisy audio | Substantial degradation remained; one diagnostic measured 50% WER |

These are small development checks, not a general benchmark. The three-sample accuracy comparison used synchronous prefix processing, not real-time latency measurement. Word error rate measures Arabic transcription, not English translation quality. Large/turbo dispatch is unit-tested with injected engines; those weights have not been loaded or benchmarked here.

Gulf dialect, exhibition noise, browser microphone denial/disconnection, prolonged booth operation, and offline clean-machine recovery remain incompletely qualified. No ten-minute resource soak or Docker acceptance test is claimed. [DEMO_VALIDATION.md](DEMO_VALIDATION.md) records dated evidence and remaining risks; later entries supersede the original audit baseline.

## Automatic validation

No speaker playback or manual microphone operation is needed for the audio checks. The scripts download a few public MSA recordings once and reuse a scratch cache.

```powershell
# Unit tests
.\.venv\Scripts\python.exe -m pytest tests/

# Actual model accuracy checks
.\.venv\Scripts\python.exe scripts/validate_live_accuracy.py --work-dir work/audio-check --report outputs/audio-accuracy.json

# With the real demo running: paced WebSocket audio and session checks
.\.venv\Scripts\python.exe scripts/validate_live_socket.py --work-dir work/audio-check --report outputs/socket-check.json
```

Rebuild after frontend changes:

```powershell
cd production-ui
npm ci
npm run build
cd ..
```

Test audio and generated reports are excluded from Git. Recordings come from the Arabic Speech Corpus mirror `tunis-ai/arabic_speech_corpus`, attributed to Nawar Halabi under CC-BY-4.0. They are not bundled with the application. These read-MSA samples do not qualify Gulf speech or booth noise.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Python version or venv error | Create the venv with `py -3.12 -m venv .venv`; avoid a different default Python |
| Dependencies or frontend build missing | Run `launch_demo.ps1 -Prepare` with internet, then launch again |
| Models still loading | Wait for readiness; first downloads/load take time. Avoid repeated restarts |
| Microphone denied or silent | Allow microphone access for the printed localhost address; check browser input and Windows privacy settings |
| Slow drafts | Use Small, short recordings, pauses, and fewer background CPU/RAM consumers |
| Inaccurate drafts | Use a close-talk headset and reduce noise. Try Medium if its delay is acceptable |
| New model fails | Wait for rollback, use the previous size, or restart to Small; inspect the launcher error |
| Connection interrupted | Start a new session; interrupted audio is not resumed |
| Live fails during the talk | Explicitly select Simulated, or restart with `-FallbackOnly`, and explain that it is scripted |
| Default port occupied | Use the exact printed URL; defaults are UI 3000 and API 8765 |

An unusual UI port may prevent model changes under the default Origin allowlist. Stop duplicate launchers and prefer default ports; custom origins require `RTT_CORS_ORIGINS` before startup.

After caching intended models, `HF_HUB_OFFLINE=1` can prevent network model lookups. Validate the prepared laptop offline before relying on it at the exhibition.

## Local operation and deployment scope

The supported Windows launcher binds services to **127.0.0.1**. Model changes require a local client and an allowed browser Origin; overlapping live/model operations are rejected. These are not complete public authentication or abuse controls.

Docker/Compose/Caddy configurations and a Gradio debug UI remain in the repository. They are not the accepted exhibition launch path; Docker has not been revalidated with these changes. Public exposure requires a separate security/deployment review. The Docker-oriented `.env.example` does not override every setting in the fixed exhibition profile: model selection is owned by `live/runtime.py` and the dropdown.

## Licensing

The active translator is **`Helsinki-NLP/opus-mt-ar-en`**, whose [model card lists Apache-2.0](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en). Upstream Whisper publishes an [MIT license](https://github.com/openai/whisper/blob/main/LICENSE). Preserve applicable notices and review the exact converted weights and dependency terms used in your distribution.

NLLB and optional research backends are not selected by this demo. Do not substitute noncommercial weights into a company exhibition without appropriate permission. This checkout has no repository-wide source LICENSE file; dependency/model licenses do not establish ownership or redistribution permission for this project's source.

## Repository guide

| Path | Purpose |
| --- | --- |
| `launch_demo.ps1` | Windows preparation, daily launch, simulated-only launch |
| `run_production.py` | Local API and prepared Next.js UI launcher |
| `run_demo.py` | Scripted-only server and UI launcher |
| `production-ui/` | Arabic/English interface, microphone capture, model dropdown |
| `src/rtt/api/production_server.py` | WebSocket API and runtime configuration |
| `src/rtt/live/context_session.py` | Bounded full-context drafts and finalization |
| `src/rtt/live/runtime.py` | Model ownership and safe switching |
| `src/rtt/asr/`, `src/rtt/mt/` | Recognition and translation backends |
| `scripts/validate_live_*.py` | Automatic real-audio checks |
| `tests/` | Model, session, audio, and API tests |
| [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) | Presenter instructions and recovery |
| [DEMO_VALIDATION.md](DEMO_VALIDATION.md) | Executed checks and remaining risks |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Original prioritized audit and decisions |
| [TERRA_HANDOFF.md](TERRA_HANDOFF.md) | Historical handoff; validation records subsequent progress |

## Deliberately deferred

Voice cloning, simultaneous English speech output, speculative speech synthesis, trained risk scoring, multi-model verification, a new Qwen translation stack, public hosting, and continuous multi-user interpretation are outside this demo. Experimental modules do not imply those capabilities are implemented in the supported interface.
