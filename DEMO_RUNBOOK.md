# GITEX presenter runbook

The implementation provides explicit Live and Simulated modes. Live drafts now reprocess the complete recording so far and can correct earlier wording. Automated Arabic audio checks are available; no manual audio playback is needed for regression testing. Noisy/dialectal accuracy remains limited.

## Planned one-time preparation (technical owner)

Use the existing repository at `C:\Users\xsaad\Documents\real-time-translation`. Install Python 3.12 and Node LTS if absent, then create the repo's Python 3.12 `.venv` if needed. The machine already inspected has both the venv and cached small/Marian models.

Prepare the laptop before travel with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_demo.ps1 -Prepare
```

This installs dependencies, caches Whisper-small/Marian and builds the frontend while internet is available. Daily launch serves that production build.

## Planned daily launch

From the repository directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_demo.ps1
```

Keep that window open. Open the exact localhost URL it prints. Wait for Live ready, or choose the visible Simulated mode if live is unavailable. The URL may use a different port if one is occupied. Add `-FallbackOnly` to start the no-model scripted demo.

The launcher selects the repository Python 3.12 venv. Do not use the machine's default Python 3.14.

### Speech model dropdown

The compact header dropdown offers Tiny, Base, Small, Medium, Large turbo and Large v3. Small is the default for this CPU laptop. Stop the session before changing it; switching is disabled while recording or finalizing. Both live drafts and Stop use the selected size. The translator remains Marian.

Larger models can improve recognition but may delay updates considerably. In one automated clean-MSA check, Medium produced a correct transcript and took 11.9 seconds to first output; this single sample is not a general accuracy guarantee. Medium loaded from the existing cache. Models not yet cached download on first selection, so load any intended exhibition model beforehand. If loading fails, the previous cached speech model is reloaded. Only one speech model is kept active; live admission pauses during the switch. The dropdown shows loading and errors. A restart returns to Small.

## Planned presentation sequence

1. Use a close-talk headset and a quiet position. Open browser fullscreen; confirm Arabic and English are readable.
2. Select **Live translation**. Confirm the mode banner, model readiness and that no earlier session is processing.
3. Press Start, allow microphone access, then speak in short natural sentences. Initial text needs roughly four seconds of speech plus decoding time (about seven seconds to first text in the automated laptop check). Pause to allow drafts to catch up automatically; there is no need to Stop to correct earlier words.
4. Draft text can change throughout the recording. Stop finalizes the latest complete context and preserves any last microphone samples. Use New session before the next demonstration. The recording finalizes automatically at 30 seconds. Do not promise simultaneous spoken interpretation.
5. If live is slow or inaccurate, stop it and explicitly select **Simulated — scripted output**. Say: “This is a scripted preview of the interface.” Press Start to play the prepared script. Do not speak into the microphone as if it were driving these outputs.

## Arabic rehearsal phrases

These are suggested inputs and meanings, **not verified model outputs**:

| Arabic | Intended meaning |
|---|---|
| مرحباً بكم في دبي. | Welcome to Dubai. |
| نعرض اليوم نظاماً لترجمة الكلام العربي إلى اللغة الإنجليزية. | Today we are showing a system that translates Arabic speech into English. |
| يساعد هذا النظام الناس على التواصل بلغات مختلفة. | This system helps people communicate in different languages. |
| حياكم الله، اليوم بنوريكم كيف نترجم الكلام من العربي للإنجليزي. | Welcome; today we will show you how we translate speech from Arabic into English. (Gulf-style rehearsal; validate with the actual speaker.) |

Use phrases the presenter has successfully rehearsed. No Gulf dialect accuracy claim is approved yet.

## Recovery (to verify after implementation)

- **Mic denied/unavailable:** use browser site settings to allow the headset mic, then start a new session; simulation must work without permission.
- **Model loading/failure or internet loss:** select Simulated visibly. Prepared simulation must need no models or network. Do not attempt package/model downloads during the talk.
- **Connection lost:** the recording must stop, prior text remain labeled Interrupted. Start a new session; do not claim the interrupted session resumed.
- **Processing too long:** show the reported timeout/error and switch to Simulated. Native model work may still be ending; do not repeatedly press Start.
- **Application frozen:** Ctrl+C in the launcher window, then run the daily launch command again. Planned emergency launch: same command with `-FallbackOnly`. These cleanup/restart procedures require testing before presenter use.

## Truthful claims

After acceptance, Live may be described as local Arabic speech recognition followed by English **text** translation, with drafts and a final pass. Accuracy and delay depend on speech, noise and hardware. Simulation demonstrates a scripted interface and does not recognize arbitrary speech. This build does not provide verified simultaneous English speech, voice cloning, trained risk scoring, certified translation or proven Gulf/noise robustness. Do not quote README target latency as measured performance.

## Automatic audio regression

The developer can run the following after launching the real demo. It downloads three small public test recordings once and then reuses its scratch cache; it never plays them through the speakers or requests the microphone.

```powershell
.\.venv\Scripts\python.exe scripts/validate_live_accuracy.py --work-dir work/audio-check --report outputs/audio-accuracy.json
.\.venv\Scripts\python.exe scripts/validate_live_socket.py --work-dir work/audio-check --report outputs/socket-check.json
```

The baseline comparison additionally accepts `--baseline PATH_TO_SAVED_OLD_SESSION_MODULE`. Validation recordings are public MSA; synthetic white noise is a diagnostic, not a booth-noise qualification.
