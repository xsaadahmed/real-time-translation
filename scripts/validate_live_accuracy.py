"""Automatic real-model Arabic regression: no speakers or microphone needed.

Fetches a few CC-BY-4.0 Arabic Speech Corpus recordings via the HF dataset
viewer, not the full corpus. Audio stays in the requested scratch directory.
The corpus is read MSA (one speaker), not an exhibition-noise/Gulf benchmark.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rtt.audio import load_audio
from rtt.config import ASR_SAMPLE_RATE, ASRConfig, MTConfig, PipelineConfig
from rtt.live.context_session import ContextSessionStore
from rtt.pipeline import build_pipeline

BUCKWALTER = dict(zip(
    "'|>&<}AbptvjHxd*rzs$SDTZEgfqklmnhwYyFNKaui~o`{",
    "ءآأؤإئا بةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰٱ".replace(" ", ""),
))
BUCKWALTER["^"] = "ث"  # Corpus uses ^ for thaa (e.g. أبحاث).


def words(text):
    text = text.replace("^", "ث")
    text = re.sub(r"[\u064b-\u065f\u0670\ufeff]", "", text)
    text = re.sub("[أإآٱ]", "ا", text).replace("ى", "ي")
    return re.sub(r"[^\u0621-\u064a\s]", " ", text).split()


def errors(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        new = [i]
        for j, b in enumerate(hyp, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (a != b)))
        row = new
    return row[-1], len(ref)


def fetch_samples(directory):
    manifest = directory / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    url = ("https://datasets-server.huggingface.co/first-rows?"
           "dataset=tunis-ai%2Farabic_speech_corpus&config=default&split=train")
    with urlopen(url + f"&refresh={int(time.time())}", timeout=45) as response:
        data = json.load(response)
    samples = []
    for item in data["rows"][:3]:
        row, idx = item["row"], item["row_idx"]
        path = directory / f"corpus_{idx}.wav"
        with urlopen(row["audio"][0]["src"], timeout=45) as response:
            path.write_bytes(response.read())
        reference = "".join(BUCKWALTER.get(c, c) for c in row["orthographic"])
        samples.append({"path": str(path), "reference": reference, "row": idx})
    manifest.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    if args.rescore:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        for entry in report["cases"]:
            entry["reference"] = entry["reference"].replace("^", "ث")
            count, total = errors(entry["reference"], entry["live_arabic"])
            entry.update(live_errors=count, reference_words=total, live_wer=round(count / total, 4),
                         final_errors=errors(entry["reference"], entry["final_arabic"])[0])
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    samples = fetch_samples(args.work_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cfg = PipelineConfig(
        asr=ASRConfig(model_size="small", device="cpu", compute_type="int8",
                      beam_size=3, temperatures=(0.0,), initial_prompt="",
                      condition_on_previous_text=True, cpu_threads=4, num_workers=1),
        mt=MTConfig(backend="marian", device="cpu", num_beams=2, torch_threads=4),
    )
    pipeline = build_pipeline(cfg, include_tts=False)
    started = time.perf_counter()
    pipeline.warmup()
    report = {"source": "https://huggingface.co/datasets/tunis-ai/arabic_speech_corpus",
              "license": "cc-by-4.0", "attribution": "Arabic Speech Corpus, Nawar Halabi; HF mirror tunis-ai",
              "method": "deterministic 500ms PCM delta replay; synchronous prefix decodes, not a latency benchmark",
              "normalization": "strip tashkeel/punctuation; normalize alif and alif maqsura",
              "load_seconds": round(time.perf_counter() - started, 2), "cases": []}
    classes = [("context", ContextSessionStore, 3)]
    if args.baseline:
        name = "rtt.live._audit_baseline"
        spec = importlib.util.spec_from_file_location(name, args.baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        classes.insert(0, ("baseline", module.LiveSessionStore, 1))
    for sample in samples:
        audio = load_audio(sample["path"], ASR_SAMPLE_RATE)
        # Give the stream a trailing pause; this is identical in both runs.
        audio = np.concatenate((audio, np.zeros(4 * ASR_SAMPLE_RATE, dtype=np.float32)))[:480000]
        for label, cls, beam in classes:
            pipeline.asr.config.beam_size = beam
            pipeline.asr.config.condition_on_previous_text = label == "context"
            store = cls(pipeline) if label == "context" else cls(pipeline, lambda: pipeline)
            state = store.create()
            updates = []
            started = time.perf_counter()
            for start in range(0, audio.size, 8000):
                store.append_chunk(state.session_id, ASR_SAMPLE_RATE, audio[start:start+8000])
                threshold = 4 if label == "context" else 2
                if state.new_audio_sec() >= threshold:
                    store._run_live_increment(state)
                    updates.append({"audio_seconds": state.processed_samples / ASR_SAMPLE_RATE,
                                    "arabic": state.arabic_text, "english": state.english_text})
            # Decode any remaining tail as a live update, before Stop.
            if state.new_audio_sec() > 0:
                store._run_live_increment(state)
            live_ar, live_en = state.arabic_text, state.english_text
            live_errors, reference_words = errors(sample["reference"], live_ar)
            final_start = time.perf_counter()
            final = store.finalize(state.session_id)
            entry = {"row": sample["row"], "mode": label, "beam": beam,
                     "audio_seconds": round(audio.size / ASR_SAMPLE_RATE, 2),
                     "reference": sample["reference"], "live_arabic": live_ar, "live_english": live_en,
                     "final_arabic": final.arabic_text, "final_english": final.english_text,
                     "live_errors": live_errors, "reference_words": reference_words,
                     "live_wer": round(live_errors / max(1, reference_words), 4),
                     "final_errors": errors(sample["reference"], final.arabic_text)[0],
                     "stop_seconds": round(time.perf_counter() - final_start, 3),
                     "compute_seconds": round(time.perf_counter() - started, 2), "updates": updates}
            report["cases"].append(entry)
            if label == "context" and args.baseline:
                before = next(c for c in report["cases"] if c["row"] == sample["row"] and c["mode"] == "baseline")
                assert live_errors <= before["live_errors"], "Live accuracy regressed against baseline"
            print(json.dumps({k: entry[k] for k in ("row", "mode", "live_wer", "stop_seconds", "compute_seconds")}), flush=True)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Silence goes through the actual new store, without bypassing its guards.
    state = store.create()
    store.append_chunk(state.session_id, ASR_SAMPLE_RATE, np.zeros(64000, dtype=np.float32))
    store._run_live_increment(state)
    assert not state.arabic_text and not store.finalize(state.session_id).english_text
    report["digital_silence"] = "passed"
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
