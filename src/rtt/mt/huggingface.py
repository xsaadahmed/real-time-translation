"""Seq2seq translation backends running locally through transformers.

Two models are supported, both Arabic -> English:

``marian``
    ``Helsinki-NLP/opus-mt-ar-en``. ~300 MB, comfortably real-time on CPU.
``nllb``
    ``facebook/nllb-200-distilled-600M``. Noticeably better on long or formal
    Arabic, roughly 2.4 GB, several times slower on CPU.
"""

from __future__ import annotations

import logging

from ..config import MTConfig
from .base import Translator

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "marian": "Helsinki-NLP/opus-mt-ar-en",
    "nllb": "facebook/nllb-200-distilled-600M",
}

# NLLB uses FLORES-200 language codes rather than ISO-639-1.
NLLB_TARGET_CODE = "eng_Latn"


class HuggingFaceTranslator(Translator):
    def __init__(self, config: MTConfig | None = None) -> None:
        self.config = config or MTConfig()
        if self.config.backend not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown MT backend '{self.config.backend}'. "
                f"Expected one of: {', '.join(sorted(DEFAULT_MODELS))}"
            )
        self.model_name = self.config.model_name or DEFAULT_MODELS[self.config.backend]
        self.name = f"{self.config.backend}:{self.model_name}"
        self._tokenizer = None
        self._model = None

    @property
    def _is_nllb(self) -> bool:
        return self.config.backend == "nllb"

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self.config.resolved_device()
        logger.info("Loading translation model '%s' on %s", self.model_name, device)

        tokenizer_kwargs = {"cache_dir": self.config.cache_dir}
        if self._is_nllb:
            tokenizer_kwargs["src_lang"] = self.config.nllb_source_code

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=self.config.cache_dir,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self._model.to(device)
        self._model.eval()

    def _target_bos_token_id(self) -> int | None:
        """Resolve NLLB's forced target-language token across transformers versions."""
        if not self._is_nllb:
            return None
        assert self._tokenizer is not None
        # Older transformers exposed lang_code_to_id; newer ones dropped it in
        # favour of plain vocabulary lookup.
        mapping = getattr(self._tokenizer, "lang_code_to_id", None)
        if isinstance(mapping, dict) and NLLB_TARGET_CODE in mapping:
            return int(mapping[NLLB_TARGET_CODE])
        return int(self._tokenizer.convert_tokens_to_ids(NLLB_TARGET_CODE))

    def translate_batch(self, texts: list[str]) -> list[str]:
        texts = [text for text in texts if text.strip()]
        if not texts:
            return []

        self.load()
        assert self._model is not None and self._tokenizer is not None

        import torch

        device = self.config.resolved_device()
        forced_bos_token_id = self._target_bos_token_id()
        outputs: list[str] = []

        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start : start + self.config.batch_size]
            encoded = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)

            generate_kwargs = {
                "num_beams": self.config.num_beams,
                "max_new_tokens": self.config.max_new_tokens,
                # Greedy/beam search without sampling keeps output reproducible,
                # which matters when comparing pipeline variants later.
                "do_sample": False,
            }
            if forced_bos_token_id is not None:
                generate_kwargs["forced_bos_token_id"] = forced_bos_token_id

            with torch.inference_mode():
                generated = self._model.generate(**encoded, **generate_kwargs)

            outputs.extend(
                self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            )

        return [output.strip() for output in outputs]


__all__ = ["DEFAULT_MODELS", "HuggingFaceTranslator"]
