# Terra implementation handoff

**User action:** switch this chat to GPT-5.6 Terra, Medium reasoning, then ask it to execute this handoff. Terra was unavailable in Sol's subagent list; no delegation has occurred.

## Work location and authority

Repository: `C:\Users\xsaad\Documents\real-time-translation`.
Base HEAD: `88bf893dcb83a2cadcbfa5b3c27d5f6b41f612ac`.
Preserve the pre-existing change in `production-ui/next-env.d.ts` (dev route type paths). Sol added only these documentation files. Confirm current status before editing; preserve any subsequent user changes. No pushes, public deployment, paid APIs or infrastructure.

Read `IMPLEMENTATION_PLAN.md` completely, then implement M1 through M5 sequentially. Architecture, exact file scopes, limits and acceptance criteria are specified there; do not reopen ordinary decisions. One implementation session, one agent. No additional agents or repository-wide rewrite. Read `production-ui/AGENTS.md` and the relevant installed Next docs before frontend edits.

## Nonnegotiable behavior

1. Keep the Next interface and real Whisper→Marian cascade. Exhibition profile uses one preloaded small/int8 Whisper + Marian pipeline for both live and Stop; no NLLB, lazy medium, TTS or research models.
2. Two explicit session modes on one API/UI. Simulation is labeled throughout and never requests a microphone; fallback remains usable when live imports, model warmup or connectivity fail. No silent mode switches.
3. Correct chunk append semantics, neutral/no ASR prompt, no translation from digital silence, bounded recording/input resources, one live inference job through completion.
4. Interrupt on disconnect instead of pretending to resume. Preserve visible output with an interrupted label. Flush browser tail on Stop; handle asynchronous permission cancellation and finalization failure.
5. Remove indiscriminate process killing. Loopback binding, Origin/Host checks, disabled exhibition config mutation, no transcript logging/retention by default.
6. Build once during explicit preparation; one-window daily PowerShell launcher; never install/download during presentation startup. Test process ownership and cleanup.

## Evidence already collected; avoid repeating expensive audit work

- Existing venv Python 3.12.10, faster-whisper 1.2.1, torch 2.13.0, transformers 4.57.6, Gradio 6.17.3, FastAPI 0.141.1.
- Host: Ryzen 7 8840HS, 8 cores/16 threads, about 15.3 GiB RAM, AMD integrated graphics; no NVIDIA GPU identified. Default `python` is 3.14.2; use repo venv explicitly.
- Existing tests: **54 passed, 4 skipped**, 46.80 s; optional morphology skipped. Frontend clean-clone `npm ci`, `npm run build`, `npx tsc --noEmit` passed with Node 24.12.0. CI uses Node 22; build skips types until fixed.
- Cached small/int8 + Marian loaded offline in **12.99 s**; actual Arabic text translation in **0.61 s**. No speech recognition accuracy result exists.
- Digital silence with default wrapper prompt hallucinated `مرحبا، أنا من لبنان.` in 3.10 s. Same cached ASR with runtime neutral-prompt override produced empty output: 1.33 s without VAD, 0.23 s with VAD. Override was scratch-only, not a source fix.
- `merge_audio` retained 9,000 instead of 17,000 samples on larger delta; 8,000 instead of 16,000 on repeated delta.
- Simulated API `/health` 200; config `mode=demo`; socket initial update and Stop→progress→nonempty final passed via TestClient. Browser still requests mic and never shows persistent mode.
- Existing model cache contains base/small/medium/tiny and Marian/NLLB directories. Small/Marian actually loaded offline; other cached files were not verified. Do not download or load NLLB for this project.
- Old benchmark audio isn't tracked/present at documented samples path. Do not call old commit-message WER/lag measurements your own.

Baseline probes are at this chat's `work/audit_probe.py` and `work/silence_probe.py`; they use the original checkout's venv and no paid services. Baseline validation is in `DEMO_VALIDATION.md`.

## Workflow and reporting

Implement one milestone, run its focused tests, record results and move on. Fix ordinary failures independently. After two unsuccessful attempts at the same failure, stop that line of work and report exact errors to Sol; do not consume allowance on repeated speculative changes. Escalate an architecture conflict or missing authorization instead of inventing product behavior.

Update the plan's status as work actually completes. Replace the draft runbook with commands verified from a clean launch. Append tests, outputs, measured timings and limitations to validation; never erase baseline failures. Inventory dependency/model notices without fabricating source ownership. No blanket promise of exhibition readiness if mic/noise/rehearsal gates remain untested.

At the end report: changed files, milestone status, tests passed/failed/skipped, exact launch command, remaining blockers and a concise diff summary. Then request the user switch back to GPT-6.1 Sol for final QA of source diff, lifecycle, licensing/exposure and operational rehearsal. Do not claim that Sol review has occurred.
