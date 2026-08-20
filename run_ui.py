"""Debug-only Gradio scratch UI. Not the product.

Use ``python run_production.py`` (or Docker) to run the interpreter.

    python run_ui.py
    python run_ui.py --asr-model large-v3 --mt-backend nllb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rtt.config import PipelineConfig  # noqa: E402
from rtt.ui.gradio_app import launch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Debug-only Gradio UI. For the real interpreter use "
            "`python run_production.py` or Docker."
        )
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--asr-model",
        default=None,
        help="tiny | base | small | medium | large-v3 | large-v3-turbo (default: medium)",
    )
    parser.add_argument(
        "--mt-backend",
        default=None,
        choices=["marian", "nllb"],
        help="default: nllb for better Arabic→English quality",
    )
    args = parser.parse_args()

    print(
        "Gradio debug UI only — not the production interpreter.\n"
        "Use `python run_production.py` (http://127.0.0.1:3000) or Docker.",
        file=sys.stderr,
    )

    config = PipelineConfig()
    if args.asr_model:
        config.asr.model_size = args.asr_model
    if args.mt_backend:
        config.mt.backend = args.mt_backend
        config.mt.model_name = ""

    launch(config, port=args.port)


if __name__ == "__main__":
    main()
