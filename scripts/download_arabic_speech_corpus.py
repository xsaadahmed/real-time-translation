"""Download and extract a slice of the Arabic Speech Corpus (Halabi 2016)
into real .wav files for scripts/harvest_labels.py to run against.

    python scripts/download_arabic_speech_corpus.py                # first 20 train clips
    python scripts/download_arabic_speech_corpus.py --limit 0       # every clip (train + test)
    python scripts/download_arabic_speech_corpus.py --split both

The Hugging Face dataset card (halabi2016/arabic_speech_corpus) can no
longer be loaded via `datasets.load_dataset` - the modern `datasets`
library dropped support for community loading scripts, and this dataset
has no pre-built parquet files, only the script. That script's own source
(readable via `hf_hub_download('halabi2016/arabic_speech_corpus',
'arabic_speech_corpus.py', repo_type='dataset')`) points at the original
corpus zip, which this downloads directly instead:
http://en.arabicspeechcorpus.com/arabic-speech-corpus.zip (~1.2GB).

Note: the corpus's .lab transcript files are Buckwalter transliteration
(ASCII), not Arabic Unicode script - there is no native-script transcript
shipped with this corpus. Saved alongside each .wav as a .txt purely for
human reference; harvest_labels.py doesn't read it (Whisper produces its
own Arabic-script transcription directly from the audio).
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "http://en.arabicspeechcorpus.com/arabic-speech-corpus.zip"
ZIP_PATH = Path("downloads/arabic-speech-corpus.zip")
OUT_DIR = Path("samples/arabic")
ROOT = "arabic-speech-corpus"


def _download_zip() -> None:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ZIP_URL} -> {ZIP_PATH} (~1.2GB, resumable if it drops)")
    # urlretrieve doesn't resume; if this gets interrupted, `curl -L -C -
    # -o downloads/arabic-speech-corpus.zip <ZIP_URL>` will pick up where it
    # left off (used to recover this exact download during development).
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="0 for no limit (every clip)")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train")
    args = parser.parse_args()

    if not ZIP_PATH.exists():
        _download_zip()

    with zipfile.ZipFile(ZIP_PATH) as z:
        if z.testzip() is not None:
            print(f"Corrupt entry found in {ZIP_PATH} - delete it and re-run to redownload.")
            return 1

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        names = z.namelist()

        subdirs = []
        if args.split in ("train", "both"):
            subdirs.append(("train", f"{ROOT}/wav/", f"{ROOT}/lab/"))
        if args.split in ("test", "both"):
            subdirs.append(("test", f"{ROOT}/test set/wav/", f"{ROOT}/test set/lab/"))

        written = 0
        for split_name, wav_prefix, lab_prefix in subdirs:
            wav_names = sorted(n for n in names if n.startswith(wav_prefix) and n.endswith(".wav"))
            if args.limit:
                wav_names = wav_names[: args.limit]

            for i, wav_name in enumerate(wav_names):
                stem = Path(wav_name).stem  # e.g. "ARA NORM  0002"
                lab_name = f"{lab_prefix}{stem}.lab"

                out_stem = f"halabi_{split_name}_{i:04d}"
                (OUT_DIR / f"{out_stem}.wav").write_bytes(z.read(wav_name))

                try:
                    text = z.read(lab_name).decode("utf-8").strip()
                except KeyError:
                    text = ""
                (OUT_DIR / f"{out_stem}.txt").write_text(text, encoding="utf-8")

                written += 1

    print(f"Wrote {written} clips to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
