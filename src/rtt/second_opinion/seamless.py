"""Seamless second-opinion backend, via transformers.

README's target is SeamlessStreaming - Meta's true streaming ar->en model.
That is a research release (facebookresearch/seamless_communication), not a
pip-installable transformers model, and out of reach on CPU-only hardware
regardless: even the smallest usable checkpoint here,
'facebook/hf-seamless-m4t-medium', is ~4.8 GB and a 1.2B-parameter model -
nowhere near real-time on CPU. This backend runs it in offline (non-
streaming) batch mode as a placeholder for the same *role* (an independent
second opinion with no shared ASR error), not the same latency profile.
Swap in the true streaming model once GPU hardware is available.
"""

from __future__ import annotations

import logging

from ..audio import AudioArray, resample
from ..config import ASR_SAMPLE_RATE, SecondOpinionConfig
from .base import SecondOpinionEngine

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/hf-seamless-m4t-medium"

# Seamless uses FLORES-200 language codes rather than ISO-639-1.
_SOURCE_LANG = "arb"
_TARGET_LANG = "eng"


class SeamlessSecondOpinion(SecondOpinionEngine):
    name = "seamless"

    def __init__(self, config: SecondOpinionConfig | None = None) -> None:
        self.config = config or SecondOpinionConfig()
        self.model_name = self.config.model_name or DEFAULT_MODEL
        self._processor = None
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        device = self.config.resolved_device()
        logger.info("Loading second-opinion model '%s' on %s", self.model_name, device)

        self._processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=self.config.cache_dir
        )
        self._model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            self.model_name, cache_dir=self.config.cache_dir
        )
        self._model.to(device)
        self._model.eval()

    def translate_speech(self, audio: AudioArray, sample_rate: int) -> str:
        self.load()
        assert self._model is not None and self._processor is not None

        import torch

        audio = resample(audio, sample_rate, ASR_SAMPLE_RATE)
        if audio.size == 0:
            return ""

        device = self.config.resolved_device()
        inputs = self._processor(
            audios=audio, sampling_rate=ASR_SAMPLE_RATE, src_lang=_SOURCE_LANG,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            generated = self._model.generate(**inputs, tgt_lang=_TARGET_LANG)[0]

        text = self._processor.decode(generated.cpu().squeeze(), skip_special_tokens=True)
        return text.strip()


__all__ = ["DEFAULT_MODEL", "SeamlessSecondOpinion"]
