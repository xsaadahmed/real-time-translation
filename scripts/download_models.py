"""Pre-download every model so the pipeline can afterwards run fully offline.

Requires Python 3.12.x (see ``.python-version`` / Docker / CI).

    python scripts/download_models.py
    python scripts/download_models.py --asr-model medium --mt-backend nllb
    python scripts/download_models.py --camel-data-only

Once this completes, set HF_HUB_OFFLINE=1 to guarantee no network access.

Optional morphology (VSO / iḍāfa guards):

    pip install -r requirements-guards.txt
    python scripts/download_models.py --camel-data-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtt.config import PipelineConfig  # noqa: E402
from rtt.python_compat import require_supported_python  # noqa: E402


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


def download_camel_data() -> bool:
    """Fetch the dictionary MSA morphology DB. Return False if camel-tools missing."""
    import subprocess

    try:
        from camel_tools.morphology.database import MorphologyDB
    except ImportError:
        print(
            "[camel-tools] package not installed.\n"
            "  pip install -r requirements-guards.txt\n"
            "  python scripts/download_models.py --camel-data-only",
            file=sys.stderr,
        )
        return False

    print("[camel-tools] checking for 'calima-msa-r13' morphology database ...")
    try:
        MorphologyDB.builtin_db(db_name="calima-msa-r13", flags="a")
        print("[camel-tools] already installed")
        return True
    except FileNotFoundError:
        pass

    subprocess.run(
        [sys.executable, "-m", "camel_tools.cli.camel_data", "-i", "morphology-db-msa-r13"],
        check=True,
    )
    print("[camel-tools] done (~40MB dictionary DB)")
    return True


def main() -> int:
    require_supported_python()

    parser = argparse.ArgumentParser(description="Pre-download models for offline use")
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--mt-backend", default=None, choices=["marian", "nllb"])
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument(
        "--skip-camel-data",
        action="store_true",
        help="Skip optional morphology DB (VSO/iḍāfa guards stay disabled)",
    )
    parser.add_argument(
        "--camel-data-only",
        action="store_true",
        help="Only fetch the camel-tools morphology DB (after requirements-guards.txt)",
    )
    args = parser.parse_args()

    if args.camel_data_only:
        if not download_camel_data():
            return 1
        print("\nMorphology DB ready. Lexicon guards already work without it.")
        return 0

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

    if args.skip_camel_data:
        print(
            "[camel-tools] skipped. Lexicon guards still work; for VSO/iḍāfa:\n"
            "  pip install -r requirements-guards.txt\n"
            "  python scripts/download_models.py --camel-data-only"
        )
    elif not download_camel_data():
        print(
            "[camel-tools] optional step skipped (package not installed).\n"
            "  Core pipeline is fine. For VSO/iḍāfa guards later:\n"
            "  pip install -r requirements-guards.txt\n"
            "  python scripts/download_models.py --camel-data-only"
        )

    print("\nAll models cached. You can now run with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
