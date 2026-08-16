"""End-to-end orchestration: Arabic audio in, English text and audio out.

The pipeline owns no model logic of its own. It wires together an
:class:`~rtt.asr.base.ASREngine`, a :class:`~rtt.mt.base.Translator` and a
:class:`~rtt.tts.base.TTSEngine`, all of which are injected, so an individual
stage can be replaced without touching this file.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .asr import ASREngine, Transcript, build_asr
from .audio import AudioArray, load_audio, save_wav
from .config import PipelineConfig
from .mt import Translator, build_translator
from .text import normalize_arabic
from .tts import SpeechAudio, TTSEngine, build_tts

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything one pass through the pipeline produced."""

    arabic_text: str
    english_text: str
    transcript: Transcript
    speech: SpeechAudio | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def audio_duration(self) -> float:
        return self.transcript.audio_duration

    @property
    def real_time_factor(self) -> float:
        """Processing time divided by audio duration. Below 1.0 is faster than real time."""
        total = self.timings.get("total", 0.0)
        return total / self.audio_duration if self.audio_duration > 0 else 0.0

    def timing_summary(self) -> str:
        parts = [f"{stage} {seconds:.2f}s" for stage, seconds in self.timings.items()]
        if self.audio_duration:
            parts.append(f"audio {self.audio_duration:.2f}s")
            parts.append(f"RTF {self.real_time_factor:.2f}x")
        return " | ".join(parts)


class TranslationPipeline:
    """Arabic speech -> Arabic text -> English text -> English speech."""

    def __init__(
        self,
        asr: ASREngine,
        translator: Translator,
        tts: TTSEngine | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.asr = asr
        self.translator = translator
        self.tts = tts

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig | None = None,
        *,
        include_tts: bool = True,
    ) -> "TranslationPipeline":
        config = config or PipelineConfig()
        return cls(
            asr=build_asr(config.asr),
            translator=build_translator(config.mt),
            tts=build_tts(config.tts) if include_tts else None,
            config=config,
        )

    def warmup(self) -> None:
        """Load every model up front so the first request is not slow."""
        logger.info("Warming up models: %s", self.config.describe())
        self.asr.load()
        self.translator.load()
        if self.tts is not None:
            self.tts.load()

    @contextmanager
    def _stage(self, timings: dict[str, float], name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = time.perf_counter() - started

    def run(
        self,
        audio: AudioArray,
        sample_rate: int,
        synthesize: bool = True,
    ) -> PipelineResult:
        """Run the full pipeline over one chunk of mono float32 audio."""
        timings: dict[str, float] = {}
        started = time.perf_counter()

        with self._stage(timings, "asr"):
            transcript = self.asr.transcribe(audio, sample_rate)

        arabic_text = normalize_arabic(transcript.text)
        if not arabic_text:
            timings["total"] = time.perf_counter() - started
            logger.warning("No speech detected in the input audio")
            return PipelineResult(
                arabic_text="", english_text="", transcript=transcript, timings=timings
            )

        with self._stage(timings, "mt"):
            english_text = self.translator.translate(arabic_text)

        speech: SpeechAudio | None = None
        if synthesize and english_text and self.tts is not None:
            with self._stage(timings, "tts"):
                speech = self.tts.synthesize(english_text)

        timings["total"] = time.perf_counter() - started

        result = PipelineResult(
            arabic_text=arabic_text,
            english_text=english_text,
            transcript=transcript,
            speech=speech,
            timings=timings,
        )
        logger.info("Pipeline complete: %s", result.timing_summary())
        return result

    def transcribe(self, audio: AudioArray, sample_rate: int) -> tuple[str, Transcript, dict[str, float]]:
        """ASR only — returns Arabic text, transcript, and timings."""
        timings: dict[str, float] = {}
        with self._stage(timings, "asr"):
            transcript = self.asr.transcribe(audio, sample_rate)
        arabic_text = normalize_arabic(transcript.text)
        return arabic_text, transcript, timings

    def translate(self, arabic_text: str) -> tuple[str, dict[str, float]]:
        """MT only — returns English text and timings."""
        timings: dict[str, float] = {}
        arabic_text = normalize_arabic(arabic_text)
        if not arabic_text:
            return "", timings
        with self._stage(timings, "mt"):
            english_text = self.translator.translate(arabic_text)
        return english_text, timings

    def run_file(self, path: str | Path, synthesize: bool = True) -> PipelineResult:
        """Run the pipeline over an audio file on disk."""
        from .config import ASR_SAMPLE_RATE

        audio = load_audio(path, ASR_SAMPLE_RATE)
        return self.run(audio, ASR_SAMPLE_RATE, synthesize=synthesize)

    def translate_text(self, arabic_text: str, synthesize: bool = True) -> PipelineResult:
        """Skip ASR and translate Arabic text directly. Useful for testing MT/TTS."""
        timings: dict[str, float] = {}
        started = time.perf_counter()
        arabic_text = normalize_arabic(arabic_text)

        with self._stage(timings, "mt"):
            english_text = self.translator.translate(arabic_text) if arabic_text else ""

        speech: SpeechAudio | None = None
        if synthesize and english_text and self.tts is not None:
            with self._stage(timings, "tts"):
                speech = self.tts.synthesize(english_text)

        timings["total"] = time.perf_counter() - started
        return PipelineResult(
            arabic_text=arabic_text,
            english_text=english_text,
            transcript=Transcript(text=arabic_text),
            speech=speech,
            timings=timings,
        )

    def save_speech(self, result: PipelineResult, path: str | Path) -> Path | None:
        """Write the synthesised English audio to a WAV file."""
        if result.speech is None or result.speech.is_empty():
            return None
        return save_wav(path, result.speech.audio, result.speech.sample_rate)


def build_pipeline(
    config: PipelineConfig | None = None,
    *,
    include_tts: bool = True,
) -> TranslationPipeline:
    """Convenience constructor used by the CLI and the UI."""
    return TranslationPipeline.from_config(config, include_tts=include_tts)


__all__ = ["PipelineResult", "TranslationPipeline", "build_pipeline"]
