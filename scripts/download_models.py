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


def download_camel_data() -> None:
    """Fetch the dictionary-based MSA morphology database used by the Arabic
    structural guards (text.py). Deliberately NOT the neural disambiguator
    (disambig-*) or dialect/BERT models — those are too slow for the 240ms
    live hot loop; a plain dictionary lookup is enough for POS candidates.
    """
    import subprocess

    from camel_tools.morphology.database import MorphologyDB

    print("[camel-tools] checking for 'calima-msa-r13' morphology database ...")
    try:
        MorphologyDB.builtin_db(db_name="calima-msa-r13", flags="a")
        print("[camel-tools] already installed")
        return
    except FileNotFoundError:
        pass

    subprocess.run(
        [sys.executable, "-m", "camel_tools.cli.camel_data", "-i", "morphology-db-msa-r13"],
        check=True,
    )
    print("[camel-tools] done")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download models for offline use")
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--mt-backend", default=None, choices=["marian", "nllb"])
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-camel-data", action="store_true")
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
    if not args.skip_camel_data:
        download_camel_data()

    print("\nAll models cached. You can now run with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
