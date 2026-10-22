# Demo validation — baseline audit and live accuracy repair

21 October 2026. Audited commit `88bf893dcb83a2cadcbfa5b3c27d5f6b41f612ac`.

**Not final exhibition acceptance.** The implementation addressed the audited code defects below; microphone/noise rehearsal and final Sol review remain required.

## Environment

- Windows; AMD Ryzen 7 8840HS, 8 cores / 16 logical processors; 16,423,702,528 bytes RAM (about 15.3 GiB).
- AMD Radeon integrated graphics reported; no NVIDIA GPU identified. Inference explicitly used CPU int8 ASR and CPU MT.
- Default shell `python`: 3.14.2, unsupported by this repo. Installed `py -3.12`: 3.12.10.
- Original checkout `C:\Users\xsaad\Documents\real-time-translation`, existing `.venv` Python 3.12.10, same HEAD as remote. Pre-existing `production-ui/next-env.d.ts` modification preserved.
- Original venv metadata: faster-whisper 1.2.1, torch 2.13.0, transformers 4.57.6, Gradio 6.17.3, FastAPI 0.141.1, httpx 0.28.1, websockets 17.0.1.
- Clean frontend dependency/build checks used this chat's `work/repo/production-ui`, Node 24.12.0, npm 11.6.2, Next 16.3.0. Repository CI specifies Node 22; Node 22 was not separately tested.
- A separate minimal audit venv was created in `work/repo`; full inference checks used the existing original venv and cached original models. No paid API or new model downloads were used.

## Executed results

| Check | Actual result |
|---|---|
| Original venv: `python -m pytest tests/` | **54 passed, 4 skipped in 46.80 s**. Optional morphology guard cases skipped. These tests do not establish production socket/mic lifecycle correctness. |
| Clean frontend `npm.cmd ci --no-audit --no-fund` | Passed; 366 packages installed. No vulnerability audit performed by this command. |
| Clean frontend `npm.cmd run build` | Passed; static `/` and `/_not-found` generated. Build printed `Skipping validation of types` due to current config. |
| Clean frontend `npx.cmd tsc --noEmit` | Passed, exit 0, no diagnostics. |
| Simulated FastAPI TestClient `/health`, `/config` | HTTP 200, config mode `demo`. Deprecation warning about TestClient/httpx observed; did not fail the test. |
| Simulated socket start and immediate Stop | Initial `update`, then `progress`, then `final` containing English. This validates existing script protocol only. It fills the full script even on early Stop; plan changes that behavior. No browser or microphone used. |
| Larger delta audio chunk | **Defect reproduced:** 8,000 existing + 9,000 incoming became 9,000 samples, expected 17,000. |
| Repeated delta audio chunk | **Defect reproduced:** 8,000 + identical 8,000 retained only 8,000, expected 16,000. |
| Empty configured ASR prompt | **Defect reproduced:** wrapper substituted a Lebanon/language/identity prompt. |
| Offline `small` CPU/int8 plus Marian warmup | Passed in **12.99 s**, both models loaded with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. This is one warmup observation; includes imports/load effects and is not a cold-machine SLA. |
| Real Marian Arabic text translation | Passed, **0.61 s**. Input/output below; no speech ASR involved. |
| Actual small ASR, 2 s digital silence, default wrapper prompt | **Failed correctness:** returned `مرحبا، أنا من لبنان.` in **3.10 s**. |
| Same cached ASR, runtime-only neutral-prompt override, VAD off | Empty transcript, **1.33 s**. |
| Same neutral-prompt override, VAD on | Empty transcript, **0.23 s**. |

MT input: `مرحبا بكم في دبي. نعرض اليوم نظاما لترجمة الكلام العربي إلى اللغة الإنجليزية.`

Actual output: `Welcome to Dubai. Today we are offering a system for translating Arabic into English.`

The neutral-prompt override was applied to one scratch probe instance, not saved into application source. Passing digital silence does not establish noise rejection, speech accuracy or acceptable Gulf Arabic performance.

## Post-change validation

- `\.venv\Scripts\python.exe -m pytest tests/`: **60 passed, 4 skipped**. One Starlette TestClient deprecation warning remains in the dependency stack.
- `production-ui\npx.cmd tsc --noEmit`: passed.
- `production-ui\npm.cmd run build`: passed with TypeScript validation enabled.
- Added tests verify chunk delta preservation, neutral prompt behavior, Marian default selection, simulated socket completion, and production API simulated routing with live models disabled.
- Remaining unverified: actual browser microphone lifecycle, live Arabic speech accuracy, booth noise, 30-second resource cap under real audio, and full launcher process-tree cleanup.

## Reproduction

Baseline suite, in original checkout:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

From this chat's directory, executed scratch checks:

```powershell
& 'C:\Users\xsaad\Documents\real-time-translation\.venv\Scripts\python.exe' -u work/audit_probe.py
& 'C:\Users\xsaad\Documents\real-time-translation\.venv\Scripts\python.exe' -u work/silence_probe.py
```

Probe source is retained in `work/` for Terra. It imports the original checkout and uses cached models offline. Generated silence contains no human speech. No speech recordings were retained by these probes.

## Hypotheses resolved

- Scripted demo: confirmed from code and actual socket exchange. It is simulation, not cached or recorded real inference.
- Unauthenticated public-capable routes, shared inference lock, new session on reconnect, unbounded audio: confirmed from source; no public deployment or load attack performed.
- NLLB noncommercial restriction: confirmed against the [publisher's model card](https://huggingface.co/facebook/nllb-200-distilled-600M). Marian's [model card](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en) identifies Apache-2.0; converted Whisper small's [model card](https://huggingface.co/Systran/faster-whisper-small) identifies MIT. This is not blanket legal clearance or a complete dependency inventory.
- `apc_Arab` default and missing production TTS: confirmed. No need to implement TTS for a truthful text demonstration.
- High WER and multisecond lag: historical code comments and commit `70830a6` support the report. Not independently remeasured: the documented benchmark Arabic WAV/reference are not tracked and were absent from the existing samples path inspected. Do not cite old or synthetic measurements as exhibition microphone performance.

## Outstanding at original audit (latest results below supersede completed items)

- Clean complete launcher startup and Ctrl+C process-tree cleanup. Existing launchers deliberately not run because they kill arbitrary nearby port owners.
- Browser capture, mic denial/unplug, actual mic-active indication, Stop tail flush and late permission cancellation.
- Human/prerecorded Arabic speech → recognized Arabic → English. No representative input available in the inspected sample location; no new speech was recorded or fabricated.
- Gulf/MSA WER, noisy booth rehearsal, real end-to-end speech/display latency.
- Live/fallback browser switching, model-failure recovery, disconnect during finalization, concurrency/admission and packet bounds.
- 20-cycle and 10-minute bounded-session stability/memory checks.
- Prepared production build launched with internet disabled; Docker image build/run; frontend visual QA on exhibition monitor.
- Complete dependency notices and source redistribution authority.

Terra must append post-change evidence and milestone status here. Final Sol acceptance remains pending; the project must not yet be advertised as rescued or demonstration-ready.

## Sol live-accuracy repair — 21 October 2026

User reported inaccurate live Arabic/English despite good Stop output, and requested fully automated audio testing. No user microphone/speaker playback was required. The production API now uses revisable full-context snapshots and translates the whole recognized source, instead of permanently stitching short hypotheses and fragment translations.

### Actual model comparison

Downloaded three small natural MSA recordings from [Arabic Speech Corpus, HF mirror tunis-ai](https://huggingface.co/datasets/tunis-ai/arabic_speech_corpus), attributed to Nawar Halabi (dataset metadata CC-BY-4.0). Only scratch copies were saved, not bundled into the app. Reference transcripts use corpus Buckwalter transliteration, converted to Arabic; thaa `^` is converted explicitly. Scoring removes tashkeel/punctuation and normalizes alif and alif maqsura. Original reference spacing/spelling was otherwise retained, so some substitutions count annotation differences.

Same cached small/int8 ASR and Marian models were used for both algorithms. Baseline used beam 1/no previous-text conditioning, the repaired configuration beam 3/context conditioning; both used four CPU threads and one ASR worker. This tests the combined architecture/configuration repair, not an isolated beam-size experiment. Input was fed as 500ms audio deltas with synchronous prefix decoding; these are accuracy checks, not real-time latency measurements.

| Corpus row | Reference words | Before live errors/WER | After live errors/WER | After live vs Stop |
|---|---:|---:|---:|---|
| 0 | 23 | 16 / 69.6% | 2 / 8.7% | Identical Arabic and English |
| 1 | 10 | 2 / 20.0% | 1 / 10.0% | Identical Arabic and English |
| 2 | 20 | 16 / 80.0% | 3 / 15.0% | Identical Arabic and English |
| Weighted total | 53 | 34 / 64.2% | 6 / 11.3% | All three identical |

Total synchronous compute including Stop across the three cases: baseline 76.85s, repaired 45.77s. This is not perceived latency, and the baseline is not a complete recreation of the old eight-thread running process. One offline load took 13.99s. Final outputs and every intermediate draft are in `LIVE_ACCURACY_RESULTS.json` in this chat's outputs directory.

### Real paced WebSocket validation

Ran `scripts/validate_live_socket.py` against the actual restarted live API, sending frames at microphone speed. No scripted demo endpoint was used. The harness waited in listening mode for all accepted audio to be processed **before sending Stop**.

- Natural MSA row 0 at **48 kHz**, resampled in the actual API: first nonempty live update **7.166s** after replay start; latest live WER **8.7%**, Stop WER **8.7%**, identical Arabic and English. Stop took **0.004s** after the stream had drained, because the matching snapshot was reused.
- Row 1 plus seeded white noise at **12 dB SNR**, 16 kHz: first nonempty update **7.049s**; latest live and Stop WER **50%**, identical outputs. This is a diagnostic failure of speech quality, not a passed noise-accuracy gate. White noise is not a real booth recording.
- Digital silence: no Arabic or English in live/final output.
- **20 empty start/stop cycles passed**, followed by another usable session.
- **30 seconds of accepted silence auto-finalized** without a Stop command, with no hallucinated text.
- A first paced check revealed undrained partial windows; added automatic catch-up after the previous decode. A subsequent silence check revealed decoded progress was not published when text/status stayed unchanged; added progress-change polling and a regression test. The final full socket run passed after both fixes. Earlier checks are not represented as successful final runs.

The script and full timestamped messages are retained in `LIVE_SOCKET_RESULTS.json`. First displayed text is not the same as completion of a sentence or word-level latency. The 0.004s Stop measurement does not imply instant transcription of newly captured/unprocessed audio.

### Code/build checks and runtime

- Context regression tests cover revision of earlier words, complete-source translation, cached Stop, arriving audio during inference, Stop/live ordering, digital silence, admission while native work finishes, audio limits, partial-window catch-up and unchanged-text progress updates.
- Frontend production build passed with TypeScript checking enabled after the capture-tail/cancellation changes. Existing generated `next-env.d.ts` content was restored after build.
- Final Python suite: **69 passed, 4 skipped**, one existing TestClient/httpx deprecation warning, in 9.74s. The skipped cases require optional morphology data.
- Production `/ready` returned ready; `/config` reported small/Marian, full_context, beam 3. UI HTTP 200 included the new full-context draft message. Daily startup now serves `next start`, not the development compiler. Next emits an output-standalone compatibility warning; HTTP serving was verified successfully. Docker packaging was not changed or tested in this repair.
- After the final automatic socket run, the actual API reported **24 sessions completed, zero failures/disconnects, zero active sessions, zero finalization errors**. The model-loaded API and production UI were left running for normal use.
- API uses one cached model pair and one ASR worker, CPU int8, without importing the Gradio interface. No NVIDIA GPU was present. Available system RAM remained low due to overall laptop load; no unrelated applications/processes were closed.

### Limits of this result

Three read-MSA samples/one corpus speaker and one synthetic-noise case cannot establish general Arabic or Gulf dialect accuracy. Earlier words remain drafts until the final pass, and meaningful updates require several seconds of context plus CPU time. English examples improved from fragmentary/duplicated translations to coherent sentences; no independent English reference scorer or human translation evaluation was performed. The browser microphone permission/device/acoustic path itself was not exercised; the browser-style PCM and actual backend path were. Original broad exhibition/security/licensing gates and a ten-minute sustained test remain outside this focused accuracy repair. No public deployment, paid services or remote pushes occurred.

## Model selector and pause improvements — 21 October 2026

- Added one minimal native speech-model dropdown (Tiny, Base, Small, Medium, Large turbo, Large v3). Browser inspection confirmed the choices, loading/disabled state and selected model after completion. The computer-use skill supported this UI verification. Small was restored as the default after testing.
- Switching reuses the translator and store, unloads the previous ASR before loading another, blocks new live sessions and parallel reloads, and refuses mutation during an existing session/finalization. A failed new-model load attempts to restore the previous cached ASR. Validation allows only the six known Whisper identifiers and Marian translation; no arbitrary model/path strings.
- Model mutation is restricted to local clients and allowed browser origins. Actual running API checks returned HTTP 409 for switching during a live session and HTTP 403 for an untrusted Origin.
- Actual browser selection loaded the cached Medium model; automatic microphone-speed replay of clean MSA corpus row 1 produced **0 normalized word errors**, identical live/final text, and **11.889s to first output**. Small had one error on this clip in the earlier test. This one recording does not establish overall superiority. Large/turbo weights were not downloaded or loaded during QA; their dispatch/unload behavior is covered with fake engines.
- The actual API restored Small in **1.879s**; readiness returned true. API working-set snapshots were about 1475 MB with Medium and 999 MB after Small was restored (resident snapshots, not peak memory measurements).
- Short voiced phrases followed by half a second of quiet may now trigger before the usual four-second window. One silent-tail recheck is retained, then further silence reuses the draft without repeating inference. This is an energy heuristic, not a noise-robust VAD. Regression tests verify speech after silence resumes decoding.
- Frontend production build passed with TypeScript checks. New model-operation tests cover every offered size using injected engines, old-weight release, failed-load recovery, active/in-flight admission and concurrent changes. Existing optional morphology skips and TestClient warning remain.
- Evidence is saved in this chat's outputs: `MODEL_SELECTOR_VALIDATION.json` and `MODEL_DROPDOWN.jpg`. No microphone, manual playback, paid service or new large-model download was needed.

Final regression check: 80 passed, 4 skipped in 13.63 seconds. Frontend production build passed. Live API confirmed ready with Small selected after validation.
