"""Harvest retrospective commit-candidate labels from a directory of Arabic
recordings (README step 7 - "harvest retrospective labels").

    python scripts/harvest_labels.py --audio-dir samples/arabic/
    python scripts/harvest_labels.py --audio-dir samples/arabic/ --second-opinion

This repo has no real Arabic audio corpus checked in - point --audio-dir at
one before running. Output is a JSONL file of CommitCandidateRecord rows,
the training data step 8's risk model consumes.

Draft-lane ASR/MT come from the live-tuned config (fast, matches what the
runtime commit policy will actually see); the final high-quality pass
supplies retrospective ground truth for labeling, exactly like
LiveSessionStore._run_final.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtt.asr import build_asr  # noqa: E402
from rtt.audio import load_audio  # noqa: E402
from rtt.config import ASR_SAMPLE_RATE, PipelineConfig  # noqa: E402
from rtt.harvest import harvest_utterance, log_records  # noqa: E402
from rtt.mt import build_translator  # noqa: E402
from rtt.second_opinion import build_second_opinion  # noqa: E402

logger = logging.getLogger("harvest_labels")

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"}


def _draft_config(config: PipelineConfig) -> PipelineConfig:
    config.asr.beam_size = 1
    config.asr.condition_on_previous_text = False
    return config


def _final_config(config: PipelineConfig) -> PipelineConfig:
    config.asr.beam_size = max(config.asr.beam_size, 5)
    config.asr.condition_on_previous_text = True
    config.mt.num_beams = max(config.mt.num_beams, 6)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, help="Directory of Arabic recordings")
    parser.add_argument(
        "--output", default=str(Path("outputs") / "harvest_labels.jsonl"),
        help="JSONL path to append labeled records to",
    )
    parser.add_argument("--second-opinion", action="store_true", help="Also run the Seamless channel")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    audio_dir = Path(args.audio_dir)
    audio_paths = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    if not audio_paths:
        logger.error("No audio files found under %s", audio_dir)
        return 1

    draft_config = _draft_config(PipelineConfig())
    final_config = _final_config(PipelineConfig())

    draft_asr = build_asr(draft_config.asr)
    draft_translator = build_translator(draft_config.mt)
    final_asr = build_asr(final_config.asr)
    final_translator = build_translator(final_config.mt)
    draft_asr.load()
    draft_translator.load()
    final_asr.load()
    final_translator.load()

    second_opinion = None
    if args.second_opinion:
        second_opinion = build_second_opinion(draft_config.second_opinion)
        second_opinion.load()

    total_records = 0
    for i, audio_path in enumerate(audio_paths):
        logger.info("[%d/%d] %s", i + 1, len(audio_paths), audio_path.name)
        audio = load_audio(audio_path, ASR_SAMPLE_RATE)
        records = harvest_utterance(
            audio,
            ASR_SAMPLE_RATE,
            asr=draft_asr,
            translator=draft_translator,
            final_asr=final_asr,
            final_translator=final_translator,
            session_id=audio_path.stem,
            second_opinion=second_opinion,
        )
        log_records(records, args.output)
        total_records += len(records)
        logger.info("  -> %d candidate records", len(records))

    logger.info("\n%d utterances -> %d labeled records written to %s", len(audio_paths), total_records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
