"""Pre-download every model so the pipeline can afterwards run fully offline.

    python scripts/download_models.py
    python scripts/download_models.py --asr-model medium --mt-backend nllb

Once this completes, set HF_HUB_OFFLINE=1 to guarantee no network access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtt.config import PipelineConfig  # noqa: E402


def download_asr(config: PipelineConfig) -> None:
    from faster_whisper import WhisperModel

    print(f"[asr] downloading Whisper '{config.asr.model_size}' ...")
    WhisperModel(
        config.asr.model_size,
        device="cpu",
        compute_type="int8",
        download_root=config.asr.download_root,
    )
    print("[asr] done")


def download_mt(config: PipelineConfig) -> None:
    from rtt.mt.huggingface import HuggingFaceTranslator

    translator = HuggingFaceTranslator(config.mt)
    print(f"[mt] downloading '{translator.model_name}' ...")
    translator.load()
    print("[mt] done")


def download_tts(config: PipelineConfig) -> None:
    from rtt.tts import build_tts

    print("[tts] resolving backend ...")
    engine = build_tts(config.tts)
    print(f"[tts] using '{engine.name}'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download models for offline use")
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--mt-backend", default=None, choices=["marian", "nllb"])
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig()
    if args.asr_model:
        config.asr.model_size = args.asr_model
    if args.mt_backend:
        config.mt.backend = args.mt_backend
        config.mt.model_name = ""

    download_asr(config)
    download_mt(config)
    if not args.skip_tts:
        download_tts(config)

    print("\nAll models cached. You can now run with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
