"""Central configuration for the translation pipeline.

Every model choice, device choice and tunable lives here so that swapping a
component never requires touching pipeline code. Values can be overridden with
``RTT_*`` environment variables, which is how the CLI and UI accept flags
without threading arguments through every layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

#: Whisper operates on 16 kHz mono audio; everything upstream resamples to this.
ASR_SAMPLE_RATE = 16_000


def _env(name: str, default: str) -> str:
    return os.environ.get(f"RTT_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_floats(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = _env(name, ",".join(str(value) for value in default))
    try:
        parsed = tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError:
        return default
    return parsed or default


def _default_cpu_threads() -> int:
    """Threads for the CTranslate2 (Whisper) runtime.

    faster-whisper passes ``cpu_threads=0`` straight through to CTranslate2,
    which then defaults to 4 regardless of machine size — on a many-core host
    that leaves most of the CPU idle. Half the logical cores, capped at 8,
    scales without starving the MT stage that runs alongside it.
    """
    cores = os.cpu_count() or 4
    return max(4, min(8, cores // 2))


def _default_torch_threads() -> int:
    """Threads for the torch (MT) runtime. ``0`` leaves torch's own default.

    Measured on a 20-core host (scripts/bench_threads.py): torch's default of
    14 translates a sentence in 0.33s, while pinning 4/8/12 threads costs
    1.3x/2.9x/2.9x respectively. Torch already sizes this well and capping it
    to "leave room" for ASR is a net loss, so do not override it by default.
    """
    return 0


def detect_device() -> str:
    """Return ``"cuda"`` when a usable GPU is present, otherwise ``"cpu"``."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ASRConfig:
    """Arabic speech recognition settings."""

    # tiny / base / small / medium / large-v3 / large-v3-turbo.
    model_size: str = field(default_factory=lambda: _env("ASR_MODEL", "medium"))
    device: str = field(default_factory=lambda: _env("ASR_DEVICE", "auto"))
    compute_type: str = field(default_factory=lambda: _env("ASR_COMPUTE_TYPE", "auto"))
    language: str = field(default_factory=lambda: _env("ASR_LANGUAGE", "ar"))
    beam_size: int = field(default_factory=lambda: _env_int("ASR_BEAM_SIZE", 5))
    # VAD helps clean file uploads but often deletes most of a live mic capture.
    vad_filter: bool = field(default_factory=lambda: _env_bool("ASR_VAD", False))
    condition_on_previous_text: bool = field(
        default_factory=lambda: _env_bool("ASR_CONDITION_ON_PREVIOUS", True)
    )
    initial_prompt: str = field(default_factory=lambda: _env("ASR_PROMPT", ""))
    hallucination_silence_threshold: float = field(
        default_factory=lambda: float(_env("ASR_HALLUCINATION_SILENCE", "2.0"))
    )
    compression_ratio_threshold: float = field(
        default_factory=lambda: float(_env("ASR_COMPRESSION_RATIO", "2.4"))
    )
    repetition_penalty: float = field(
        default_factory=lambda: float(_env("ASR_REPETITION_PENALTY", "1.1"))
    )
    no_repeat_ngram_size: int = field(
        default_factory=lambda: _env_int("ASR_NO_REPEAT_NGRAM", 3)
    )
    # Whisper re-decodes a segment at each temperature in turn whenever the
    # compression-ratio or log-prob check fails, so a 3-entry fallback is up to
    # a 3x tail-latency spike. The live path overrides this to a single 0.0.
    temperatures: tuple[float, ...] = field(
        default_factory=lambda: _env_floats("ASR_TEMPERATURES", (0.0, 0.2, 0.4))
    )
    cpu_threads: int = field(
        default_factory=lambda: _env_int("ASR_CPU_THREADS", _default_cpu_threads())
    )
    # Lets one pass decode while the next is being prepared.
    num_workers: int = field(default_factory=lambda: _env_int("ASR_NUM_WORKERS", 2))
    download_root: str = field(default_factory=lambda: _env("ASR_DOWNLOAD_ROOT", str(MODEL_DIR / "whisper")))

    def resolved_device(self) -> str:
        return detect_device() if self.device == "auto" else self.device

    def resolved_compute_type(self) -> str:
        if self.compute_type != "auto":
            return self.compute_type
        return "float16" if self.resolved_device() == "cuda" else "int8"


@dataclass
class MTConfig:
    """Arabic to English translation settings."""

    # "marian"  -> Helsinki-NLP/opus-mt-ar-en  (~300 MB, fast on CPU)
    # "nllb"    -> facebook/nllb-200-distilled-600M (~2.4 GB, better, slower)
    backend: str = field(default_factory=lambda: _env("MT_BACKEND", "nllb"))
    model_name: str = field(default_factory=lambda: _env("MT_MODEL", ""))
    device: str = field(default_factory=lambda: _env("MT_DEVICE", "auto"))
    num_beams: int = field(default_factory=lambda: _env_int("MT_BEAMS", 4))
    max_new_tokens: int = field(default_factory=lambda: _env_int("MT_MAX_NEW_TOKENS", 256))
    batch_size: int = field(default_factory=lambda: _env_int("MT_BATCH_SIZE", 8))
    torch_threads: int = field(
        default_factory=lambda: _env_int("MT_TORCH_THREADS", _default_torch_threads())
    )
    # NLLB FLORES code. apc_Arab = North Levantine (Lebanon/Syria); arb_Arab = MSA.
    nllb_source_code: str = field(
        default_factory=lambda: _env("NLLB_SOURCE_LANG", "apc_Arab")
    )
    cache_dir: str = field(default_factory=lambda: _env("MT_CACHE_DIR", str(MODEL_DIR / "mt")))

    def resolved_device(self) -> str:
        return detect_device() if self.device == "auto" else self.device


@dataclass
class TTSConfig:
    """English speech synthesis settings."""

    # "auto" walks the backend preference order and picks the first that loads.
    # Explicit values: "kokoro", "piper", "sapi".
    backend: str = field(default_factory=lambda: _env("TTS_BACKEND", "auto"))
    voice: str = field(default_factory=lambda: _env("TTS_VOICE", ""))
    speaking_rate: int = field(default_factory=lambda: _env_int("TTS_RATE", 175))
    piper_voice_path: str = field(default_factory=lambda: _env("PIPER_VOICE", ""))
    cache_dir: str = field(default_factory=lambda: _env("TTS_CACHE_DIR", str(MODEL_DIR / "tts")))


@dataclass
class SecondOpinionConfig:
    """Independent Seamless ar->en channel, run alongside the cascade purely
    to log agreement/disagreement (README "Seamless second-opinion channel").
    Off by default: the smallest usable checkpoint is a multi-GB, 1.2B-param
    model, nowhere near real-time on CPU-only hardware.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("SECOND_OPINION_ENABLED", False))
    model_name: str = field(default_factory=lambda: _env("SECOND_OPINION_MODEL", ""))
    device: str = field(default_factory=lambda: _env("SECOND_OPINION_DEVICE", "auto"))
    cache_dir: str = field(
        default_factory=lambda: _env("SECOND_OPINION_CACHE_DIR", str(MODEL_DIR / "second_opinion"))
    )
    log_path: str = field(
        default_factory=lambda: _env(
            "SECOND_OPINION_LOG", str(OUTPUT_DIR / "second_opinion_log.jsonl")
        )
    )

    def resolved_device(self) -> str:
        return detect_device() if self.device == "auto" else self.device


@dataclass
class PipelineConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    mt: MTConfig = field(default_factory=MTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    second_opinion: SecondOpinionConfig = field(default_factory=SecondOpinionConfig)
    output_dir: str = field(default_factory=lambda: _env("OUTPUT_DIR", str(OUTPUT_DIR)))

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build a config purely from ``RTT_*`` environment variables."""
        return cls()

    def describe(self) -> dict[str, str]:
        """Flat summary suitable for logging or display in the UI."""
        return {
            "asr.model": self.asr.model_size,
            "asr.device": self.asr.resolved_device(),
            "asr.compute_type": self.asr.resolved_compute_type(),
            "mt.backend": self.mt.backend,
            "mt.device": self.mt.resolved_device(),
            "tts.backend": self.tts.backend,
            "second_opinion.enabled": str(self.second_opinion.enabled),
        }


__all__ = [
    "ASR_SAMPLE_RATE",
    "ASRConfig",
    "MODEL_DIR",
    "MTConfig",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "PipelineConfig",
    "SecondOpinionConfig",
    "TTSConfig",
    "detect_device",
]
