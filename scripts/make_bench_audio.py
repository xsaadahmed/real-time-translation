"""Generate a reproducible Arabic speech stream for the latency benchmark.

The public Arabic corpora are awkward here: halabi2016 is a script-based
dataset (unsupported by current ``datasets``) and streaming FLEURS buffers a
multi-GB shard before yielding a single clip. For a *latency* A/B the only
requirement is that both runs decode byte-identical audio, so this synthesises
the stream locally with edge-tts and caches it.

Synthetic speech is somewhat easier to recognise than spontaneous conversation,
so treat the absolute ASR times as a floor; the before/after ratio is the
meaningful number.

Usage::

    python scripts/make_bench_audio.py            # ~60s of Levantine Arabic
    python scripts/make_bench_audio.py --seconds 30
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

VOICE = "ar-EG-SalmaNeural"

# Conversational Arabic covering the domain the README targets: Lebanon,
# language and identity. Sentence-length so the live path sees real pauses.
SENTENCES = [
    "مرحبا، أنا من لبنان وأتحدث اللغة العربية منذ الطفولة.",
    "اللغة ليست مجرد وسيلة للتواصل، بل هي جزء أساسي من الهوية.",
    "في بيروت تسمع العربية والفرنسية والإنجليزية في الشارع نفسه.",
    "كثير من الشباب اليوم يخلطون بين اللهجة المحلية والكلمات الأجنبية.",
    "أعتقد أن الحفاظ على اللغة العربية مسؤولية مشتركة بين الجميع.",
    "المدارس لها دور كبير في تعليم الأطفال لغتهم الأم بشكل صحيح.",
    "عندما نفقد لغتنا، نفقد جزءا من تاريخنا وثقافتنا.",
    "لكنني متفائل، لأن هناك اهتماما متزايدا بالأدب العربي الحديث.",
]


async def _synth(text: str, out: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, VOICE).save(str(out))


def _decode_mp3(path: Path) -> tuple[np.ndarray, int]:
    """Decode an mp3 to float32 mono via soundfile, falling back to ffmpeg."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    except Exception:
        wav = path.with_suffix(".wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", str(wav)],
            check=True,
        )
        data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples/arabic/bench_stream.wav")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--gap", type=float, default=0.4, help="pause between sentences")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    clips: list[np.ndarray] = []
    rate: int | None = None
    total = 0.0

    with tempfile.TemporaryDirectory() as tmp:
        for i, sentence in enumerate(SENTENCES):
            if total >= args.seconds:
                break
            mp3 = Path(tmp) / f"s{i}.mp3"
            asyncio.run(_synth(sentence, mp3))
            audio, sr = _decode_mp3(mp3)
            rate = rate or sr
            if sr != rate:
                raise RuntimeError(f"sample-rate mismatch: {sr} vs {rate}")
            clips.append(audio)
            total += len(audio) / sr
            # Windows consoles default to cp1252, so keep Arabic off stdout.
            print(f"  [{i + 1}/{len(SENTENCES)}] {len(audio) / sr:5.2f}s ({len(sentence)} chars)")

    if not clips or rate is None:
        print("no audio produced", file=sys.stderr)
        return 1

    gap = np.zeros(int(args.gap * rate), dtype=np.float32)
    stream = np.concatenate([part for clip in clips for part in (clip, gap)])
    sf.write(str(out), stream, rate)
    out.with_suffix(".txt").write_text("\n".join(SENTENCES[: len(clips)]), encoding="utf-8")
    print(f"\nwrote {out} — {len(stream) / rate:.2f}s @ {rate}Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
