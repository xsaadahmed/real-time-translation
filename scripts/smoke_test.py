"""Verify each pipeline stage works locally.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --audio samples/arabic.wav

Without an Arabic audio file the ASR stage is exercised by synthesising English
speech and feeding it back into Whisper, which validates the audio plumbing
end to end. Pass --audio to additionally run the real Arabic path.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtt.asr import build_asr  # noqa: E402
from rtt.audio import resample, save_wav  # noqa: E402
from rtt.config import ASR_SAMPLE_RATE, PipelineConfig  # noqa: E402
from rtt.mt import build_translator  # noqa: E402
from rtt.pipeline import TranslationPipeline  # noqa: E402
from rtt.text import chunk_for_translation  # noqa: E402
from rtt.tts import build_tts  # noqa: E402

ARABIC_SAMPLES = [
    "مرحبا بكم في هذا العرض التوضيحي للترجمة الفورية.",
    "قفز الكلب فوق السور في الحديقة.",
    "ذهبت إلى الجامعة أمس لحضور محاضرة عن الذكاء الاصطناعي.",
]

ENGLISH_PROBE = "The quick brown fox jumps over the lazy dog."


def _ok(label: str, detail: str = "") -> None:
    print(f"  [PASS] {label}" + (f" -> {detail}" if detail else ""))


def _fail(label: str, exc: Exception) -> None:
    print(f"  [FAIL] {label}: {exc}")
    traceback.print_exc()


def check_text_utils() -> bool:
    print("\n[1/5] Arabic text utilities")
    try:
        chunks = chunk_for_translation(" ".join(ARABIC_SAMPLES))
        assert len(chunks) == 3, f"expected 3 sentences, got {len(chunks)}"
        _ok("sentence splitting", f"{len(chunks)} chunks")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail("sentence splitting", exc)
        return False


def check_tts(config: PipelineConfig):
    print("\n[2/5] Text to speech")
    try:
        engine = build_tts(config.tts)
        speech = engine.synthesize(ENGLISH_PROBE)
        assert not speech.is_empty(), "backend returned no samples"
        _ok(f"backend '{engine.name}'", f"{speech.duration:.2f}s @ {speech.sample_rate} Hz")
        return engine, speech
    except Exception as exc:  # noqa: BLE001
        _fail("text to speech", exc)
        return None, None


def check_asr(config: PipelineConfig, speech) -> bool:
    print("\n[3/5] Speech recognition")
    try:
        asr = build_asr(config.asr)
        asr.load()
        _ok("model loaded", f"{config.asr.model_size} on {config.asr.resolved_device()}")

        if speech is None:
            print("  [SKIP] no probe audio available")
            return True

        # Feed the synthesised English probe back in to confirm the audio path
        # reaches Whisper intact.
        audio = resample(speech.audio, speech.sample_rate, ASR_SAMPLE_RATE)
        original_language = asr.config.language
        asr.config.language = "en"
        try:
            transcript = asr.transcribe(audio, ASR_SAMPLE_RATE)
        finally:
            asr.config.language = original_language

        recognised = transcript.text.lower()
        hits = sum(word in recognised for word in ("quick", "brown", "fox", "jumps", "dog"))
        if hits >= 3:
            _ok("round-trip recognition", f"{hits}/5 keywords -> {transcript.text!r}")
        else:
            print(f"  [WARN] weak round-trip recognition ({hits}/5): {transcript.text!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail("speech recognition", exc)
        return False


def check_mt(config: PipelineConfig) -> bool:
    print("\n[4/5] Arabic to English translation")
    try:
        translator = build_translator(config.mt)
        translator.load()
        _ok("model loaded", translator.name)
        for arabic in ARABIC_SAMPLES:
            english = translator.translate(arabic)
            assert english.strip(), "empty translation"
            print(f"      {arabic}\n        -> {english}")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail("translation", exc)
        return False


def check_pipeline(config: PipelineConfig, audio_path: str | None) -> bool:
    print("\n[5/5] Full pipeline")
    try:
        pipeline = TranslationPipeline.from_config(config)
        if audio_path:
            result = pipeline.run_file(audio_path)
            print(f"      Arabic:  {result.arabic_text}")
            print(f"      English: {result.english_text}")
        else:
            result = pipeline.translate_text(ARABIC_SAMPLES[1])
            print(f"      Arabic:  {result.arabic_text}")
            print(f"      English: {result.english_text}")

        out_dir = Path(config.output_dir)
        written = pipeline.save_speech(result, out_dir / "smoke_test_en.wav")
        _ok("pipeline run", result.timing_summary())
        if written:
            _ok("audio written", str(written))
        return True
    except Exception as exc:  # noqa: BLE001
        _fail("full pipeline", exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the local pipeline")
    parser.add_argument("--audio", default=None, help="Optional Arabic audio file")
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--mt-backend", default=None, choices=["marian", "nllb"])
    args = parser.parse_args()

    config = PipelineConfig()
    if args.asr_model:
        config.asr.model_size = args.asr_model
    if args.mt_backend:
        config.mt.backend = args.mt_backend
        config.mt.model_name = ""

    print("Configuration:", config.describe())

    results = [check_text_utils()]
    engine, speech = check_tts(config)
    results.append(engine is not None)

    if speech is not None:
        save_wav(Path(config.output_dir) / "smoke_test_probe.wav", speech.audio, speech.sample_rate)

    results.append(check_asr(config, speech))
    results.append(check_mt(config))
    results.append(check_pipeline(config, args.audio))

    passed = sum(1 for r in results if r)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
