"""Arabic ASR backed by faster-whisper (CTranslate2)."""

from __future__ import annotations

import logging

from ..audio import AudioArray, duration_seconds, resample
from ..config import ASR_SAMPLE_RATE, ASRConfig
from .base import ASREngine, Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

_DEFAULT_ARABIC_PROMPT = (
    "مرحبا، أنا من لبنان. نتحدث باللغة العربية عن لبنان واللغة والهوية."
)


class FasterWhisperASR(ASREngine):
    name = "faster-whisper"

    def __init__(self, config: ASRConfig | None = None) -> None:
        self.config = config or ASRConfig()
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        device = self.config.resolved_device()
        compute_type = self.config.resolved_compute_type()
        logger.info(
            "Loading Whisper '%s' on %s (%s)", self.config.model_size, device, compute_type
        )
        self._model = WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=compute_type,
            download_root=self.config.download_root,
        )

    def _initial_prompt(self) -> str | None:
        prompt = (self.config.initial_prompt or _DEFAULT_ARABIC_PROMPT).strip()
        return prompt or None

    def transcribe(self, audio: AudioArray, sample_rate: int) -> Transcript:
        self.load()
        assert self._model is not None

        audio = resample(audio, sample_rate, ASR_SAMPLE_RATE)
        if audio.size == 0:
            return Transcript(text="")

        kwargs: dict = {
            "language": self.config.language,
            "task": "transcribe",
            "beam_size": self.config.beam_size,
            "vad_filter": self.config.vad_filter,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "no_speech_threshold": 0.35,
            "log_prob_threshold": -1.0,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "hallucination_silence_threshold": self.config.hallucination_silence_threshold,
            "repetition_penalty": self.config.repetition_penalty,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "temperature": [0.0, 0.2, 0.4],
        }
        prompt = self._initial_prompt()
        if prompt:
            kwargs["initial_prompt"] = prompt

        segments_iter, info = self._model.transcribe(audio, **kwargs)

        segments = [
            TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_iter
            if s.text.strip()
        ]

        return Transcript(
            text=" ".join(s.text for s in segments).strip(),
            segments=segments,
            language=getattr(info, "language", self.config.language) or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            audio_duration=duration_seconds(audio, ASR_SAMPLE_RATE),
        )


__all__ = ["FasterWhisperASR"]
