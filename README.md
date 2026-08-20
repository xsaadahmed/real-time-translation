# Real-Time Arabic → English Speech Translation

Working toward a **real-time simultaneous interpreter**: Arabic speech in, English text and speech out, with speculative on-screen predictions and calibrated commitment before anything reaches the listener's ears.

**Organizing principle: draft-and-verify everywhere.** A fast model drafts, a better-informed model verifies, and only verified content crosses into output the user trusts. Speculate everywhere; verify before the ear. This applies to ASR (fast transducer drafts, Whisper verifies), translation (incomplete source drafts, more-complete source verifies), and speech (audio synthesized speculatively, playback gated on commitment).

What runs in this repo today is an earlier **cascade prototype** (Whisper → Marian/NLLB → optional TTS) with live incremental updates and a production Next.js UI. The architecture below is the **target system** the codebase is being shaped toward. Cloud APIs and network-backed models are acceptable when they improve latency or quality.

---

## Target architecture

**Verdict on cascade vs end-to-end: hybrid, decisively.** The cascade is the quality spine — Whisper's Arabic and a LoRA-tuned 14B's Arabic reasoning are each far better than anything available in a released end-to-end Arabic→English model. But a direct speech-to-text-translation model runs *alongside* it as an independent channel, because it does not share the cascade's ASR errors — and two architecturally independent systems agreeing is a far stronger safety signal than one system agreeing with itself.

```
 Headset mic (close-talk, cardioid) ─► DeepFilterNet denoise ─► Silero VAD
                                                │
                          ┌─────────────────────┴────────────────────┐
                          │                                          │
                 ═══ DRAFT LANE (fast) ═══              ═══ VERIFY LANE (accurate) ═══
                          │                                          │
              Streaming ASR drafter                        Whisper large-v3
              (whisper-turbo, 240 ms hop)                  (900 ms hop, CTranslate2)
                          │                                          │
                 provisional Arabic ─────► ARABIC STATE ◄──── verified Arabic
                                                │
                                  ┌─────────────┴──────────────┐
                                  │                            │
                     ARABIC STRUCTURAL GUARDS        SeamlessStreaming ar→en
                     (VSO gap, iḍāfa, numerals,      (independent 2nd opinion,
                      TAM particles, proclitics)      no shared ASR errors)
                                  │                            │
                                  ▼                            │
        ┌──────────────────────────────────────────────┐       │
        │  BRANCHED TRANSLATOR  (single batched call)  │       │
        │  Qwen3-14B FP8 + SI-LoRA + EAGLE-3 spec-dec  │       │
        │  shared KV prefix across all branches:       │       │
        │    • 1 branch on observed Arabic only        │       │
        │    • K=8 branches on observed + sampled        │       │
        │      Arabic futures (0.5B Arabic drafter)    │       │
        │  all branches PREFILLED with committed EN    │       │
        └──────────────────────────────────────────────┘       │
                                  │                            │
                                  ▼                            │
              ┌────────────────────────────────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────┐
   │  COMMIT RISK MODEL  (gradient-boosted, <1 ms)       │
   │  features: branch agreement depth, target-branch    │
   │  divergence, guard state, temporal survival, ASR    │
   │  confidence, Seamless agreement, current lag        │
   │  output: P(this English prefix survives to the end) │
   └─────────────────────────────────────────────────────┘
              │                                    │
      P < θ ──┴─► WAIT                   P ≥ θ ────┴─► COMMIT (immutable)
              │                                    │
              ▼                                    ▼
     GREY SPECULATIVE TEXT              BLACK COMMITTED TEXT
     (screen only, never spoken)                   │
              │                                    │
              ▼                                    ▼
     SPECULATIVE TTS (synthesized       ┌──► matches spec buffer? play instantly
     into a shadow buffer, muted) ──────┤
                                        └──► mismatch? synthesize fresh (~90 ms)
                                                     │
                                                     ▼
                                  JITTER BUFFER (600 ms target occupancy)
                                  rate control 0.95×–1.15×, pauses only
                                  at clause boundaries
                                                     │
                                                     ▼
                        Streaming TTS in the SPEAKER'S CLONED VOICE
                        (CosyVoice2-0.5B, persistent session, prosody carried
                         across chunks) ──────► English audio out
```

Two structural properties:

- **No sentence segmentation on the committed path.** Segmentation exists only to chunk text for TTS and trim context — both non-critical. Nothing waits for a sentence boundary (the failure mode on garden-path speech like “I think we should… probably… cancel the meeting”).
- **Grey lane and black lane have different latencies by design.** The eye gets text at ~600 ms; the ear gets audio at ~2.2 s. That gap is intentional — text can be retracted; audio cannot.

### Why not a single sampled future?

Asking “what is the next Arabic phrase?” is the wrong question. The question that matters is: **does any plausible continuation change the English I am about to say?** That is a quantifier over futures; one sampled continuation is a point estimate least informative when the future is most uncertain.

The fix is not to delete anticipation. It is to **marginalize over futures**, add a second orthogonal uncertainty axis anticipation alone cannot see, and **calibrate against ground truth the system harvests from itself** (retrospective labels on prefixes considered vs prefixes that actually survived).

---

## Anticipation mechanism

Replace one sampled future with a **calibrated marginal over futures, measured on two orthogonal axes, then learned rather than hand-tuned.**

### Axis 1 — Future-marginal invariance

A 0.5B Arabic causal LM samples K=8 short continuations (~8 tokens each) in one batched call (~10 ms). The translator runs 9 branches — observed-only plus eight futures — in a **single batched call with a fully shared KV prefix**. With prefix caching and continuous batching, K=8 costs barely more than K=1 (~60–100 ms on an H100). The commit candidate is the longest English prefix appearing in at least 7 of 9 branches; **agreement depth** is a continuous confidence score, not a binary flag.

### Axis 2 — Target-space divergence

Sampling futures cannot detect uncertainty about *how to render* what was already heard. The observed-only branch also runs with a small beam; measure where top hypotheses diverge. Future-agreement and target-agreement fail in different situations — a prefix must clear both.

### Axis 3 — Structural invariance (Arabic guards)

Sampling-based agreement has a blind spot: **correlated errors**, where every branch is confidently wrong the same way. Arabic produces these reliably. After hearing only `قفز`, all nine branches may commit “jumped” — unanimous and wrong, because VSO order means the English subject slot cannot be filled yet.

Guards catch this class: verb with no subject yet, bare noun that may head an *iḍāfa*, noun awaiting postnominal adjectives, partially-heard numerals (`واحد وعشرون` — emitting “one” first is unrecoverable), pre-verbal TAM particles (`كان`, `لم`, `قد`, `سوف`), dangling proclitics. Conversely, pro-drop is free latency: `ذهبتُ` carries person and number, so “I went” is committable once the verb is stable.

### Learned, calibrated risk head

Run the pipeline offline over hours of Arabic speech. For every prefix *considered* for commit, retrospectively label whether it **survived** once the full utterance is decoded — free, exact, self-supervised data. Train a small gradient-boosted classifier on branch agreement depth, target divergence, guard state, ASR posterior, Seamless agreement, current lag, tokens since last commit → P(survives). Inference is under a millisecond.

Commitment becomes a **calibrated risk estimate with a tunable operating point** θ (e.g. “we commit at 99% predicted survival”), not a hand-tuned heuristic. The temporal verifier is both the runtime safety net and the label generator that trains the prospective predictor.

---

## Commitment algorithm

Per tick (240 ms draft lane, 900 ms verify lane):

1. **Update Arabic state.** Drafter tokens feed only the speculative path. Promote to *verified* when Whisper confirms, or drafter and verifier agree across two consecutive frames. Never treat a partial Arabic word as verified — `كتاب` vs `كتابه` differ by one suffix and by “a book” vs “his book.”
2. **Guard check.** If the verified tail sits in a hazardous construction, return WAIT and skip the LLM call.
3. **Branched translation.** One batched call, nine branches, all prefilled with committed English so the model cannot rewrite spoken text.
4. **Score.** Agreement depth, target divergence, temporal survival vs previous tick, Seamless agreement → risk model → P(survives) as a function of prefix length.
5. **Commit.** Longest prefix with P(survives) ≥ θ, truncated to a word boundary. Default θ = 0.97.
6. **Lag governor.** Track lag from Whisper word timestamps. Above 3.0 s, decay θ (0.97 → 0.90 → 0.80); at 3.5 s hard ceiling, force-commit to nearest guard-safe boundary. Below 1.2 s lag, raise θ. Without this, fast speakers make lag grow without bound.
7. **Emit.** Committed words go black and enter the TTS queue. Best speculative branch beyond the commit point renders grey, damped to avoid visual flicker.

In one sentence: **commit the longest English prefix that a calibrated model predicts will survive with ≥97% probability, given eight sampled futures, competing translations of the present, the last 240 ms of audio, and Arabic constructions whose heads have not yet arrived — unless lag exceeds 3 s, in which case lower the bar rather than fall further behind.**

---

## Speculative pipeline

The retraction boundary is the ear; everything upstream should be as aggressive as hardware allows.

**Fully speculative, freely discarded:** provisional ASR from the draft lane; eight sampled Arabic futures; all nine translation branches; grey on-screen text; **synthesized audio** in a muted shadow buffer. On commit, playback is instant; on rejection, audio is discarded silently.

**Must be verified before the ear:** text entering the commit queue (guards + risk threshold). Once unmuted, waveform is final — no rollback, no correction.

**Not speculative:** committed English is prefilled into every branch — immutability by construction.

**Speculative decoding** (EAGLE-3 inside the translator) is a *computational* optimization. **Linguistic speculation** (branches over futures) is a *semantic* mechanism. They stack but solve different problems.

Speculative TTS should run **locally** so session state can be snapshotted and rolled back — a constraint on deployment, not just quality.

---

## Model and technology choices (target)

| Layer | Choice | Notes |
| --- | --- | --- |
| Front end | Cardioid headset, DeepFilterNet3, Silero VAD | Ambient noise destroys ASR before architecture can save it |
| ASR draft | `whisper-large-v3-turbo`, 240 ms hop, greedy | faster-whisper |
| ASR verify | `whisper-large-v3`, 900 ms hop, beam 2, word timestamps | Lag governor needs timestamps |
| Arabic futures | 0.5B causal LM, K=8 continuations | ~10 ms batched |
| Translation | Qwen3-14B FP8, vLLM, SI-LoRA, EAGLE-3 | LoRA on synthetic simultaneous-interp (prefix→prefix) data |
| Second opinion | SeamlessStreaming ar→en | Error-decorrelation sensor, not primary translator |
| TTS | CosyVoice2-0.5B streaming, cloned voice, persistent session | Kokoro fallback |
| Inference | 2× H100 80GB | GPU A: vLLM only. GPU B: Whispers, Seamless, TTS |

**Context:** pinned session memo (topic, entities, terminology glossary) refreshed every 30 s; last ~60 s of committed text appended. Preload glossary with domain proper nouns, company names, and Arabic honorifics.

Optional: NeMo FastConformer-RNNT on MGB-2 as drafter; Qwen3-32B FP8 as quality toggle if VRAM allows.

---

## Latency budget (target)

Three quantities:

- **Linguistic latency** — how much future Arabic the policy needs before safe to speak.
- **Computational latency** — wall-clock after required audio exists.
- **Perceived audio latency** — Arabic spoken to English audible, including jitter buffer.

| Milestone | Target | Composition |
| --- | --- | --- |
| First grey text | 550–700 ms | 240 ms hop + ASR + MT + render |
| First committed black text | 1.3–1.8 s | ~0.9 s linguistic + ~0.4 s compute |
| First spoken English | 2.0–2.5 s | commit + speculative-hit TTS + buffer |
| Steady-state ear-to-ear | 2.0–2.8 s | ~1.0 s linguistic, ~0.7 s compute, ~0.5 s buffer |
| Worst case (guard-heavy) | 3.5 s hard cap | lag governor |

Human simultaneous interpreters work at a 2–4 s ear-voice span; **the eye is served roughly four times faster than the ear** — different risk budgets because text can be retracted and audio cannot.

---

## Why this beats a simple cascade

- Anticipation: nine-branch marginal + **calibrated self-supervised risk model** instead of one sampled future and hand-tuned heuristics.
- Second axis: **target-space divergence** for translation ambiguity futures cannot see.
- Guards: catch **correlated Arabic errors** no amount of sampling fixes.
- Latency: 240 ms draft hop, EAGLE-3, speculative pre-synthesis, variable-length risk-based acceptance.
- Robustness: **error decorrelation** via independent Seamless channel vs single-ASR agreement.
- Output: cloned voice with prosody across chunks vs chunked robotic TTS.

---

## What this repo implements today

| Piece | Status |
| --- | --- |
| Cascade ASR → MT → TTS | ✅ Whisper + Marian/NLLB |
| Live incremental mic path | ✅ fast live + accurate final on stop |
| Production UI (speculative text UX) | ✅ Next.js; candidates cosmetic until commitment engine |
| Verified vs provisional Arabic (live lane) | ✅ `reconcile_provisional()` in `text.py` + session split |
| Production UI wired to verified/provisional | ✅ black = verified, grey = provisional, live WebSocket |
| Draft-and-verify dual ASR lanes (parallel) | 🔲 still single fast lane + final on stop |
| Branched translator + risk model | 🔲 |
| Arabic structural guards | ✅ `check_structural_guards()` in `text.py` (not yet wired into live commit path) |
| Seamless second opinion | 🔲 |
| Speculative TTS + jitter buffer | 🔲 |
| Cloned-voice streaming TTS | 🔲 target (CosyVoice2) |

---

## Quick start (current prototype)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_models.py
python scripts/smoke_test.py

# Dev UI — http://127.0.0.1:7860
python run_ui.py

# Production UI — http://127.0.0.1:3000 (requires Node.js)
python run_production.py
```

`run_production.py` auto-picks free ports, clears stale dev servers, and writes the WebSocket URL for the UI.

### Live tuning

```powershell
$env:RTT_LIVE_ASR_MODEL="tiny"
$env:RTT_LIVE_MT_BACKEND="marian"
$env:RTT_ASR_MODEL="large-v3-turbo"
$env:RTT_NLLB_SOURCE_LANG="apc_Arab"
```

---

## Project layout

```
run_production.py         Production Next.js UI + WebSocket API
run_ui.py                 Gradio dev UI
production-ui/            Interpreter interface (grey/black text UX)
src/rtt/
  pipeline.py             Current cascade orchestrator
  live/session.py         Incremental live sessions (step toward draft lane)
  api/production_server.py WebSocket bridge
  asr/  mt/  tts/         Pluggable backends
  text.py                 Arabic chunking; guard rules land here
scripts/                  Model download, smoke tests
```

---

## Extension points (mapped to target arch)

- `asr.base.ASREngine` — draft lane (`transcribe_chunk`) + verify lane (full re-decode).
- `live/session.py` — verified vs provisional Arabic/English; parallel verify lane still TODO.
- `text.py` — structural guards (VSO, *iḍāfa*, numerals, TAM, proclitics).
- `mt.base.Translator` — branched batched translation with shared prefix + committed EN prefill.
- New `commit/` module — risk model, lag governor, commit policy.
- `tts/` — speculative shadow buffer + jitter-buffered playback.
