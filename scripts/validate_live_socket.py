"""Replay test audio into the real WebSocket at microphone speed, silently."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys
import time

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtt.audio import load_audio, resample
from validate_live_accuracy import errors, words


async def replay(url, audio, reference, label, rate=16000):
    audio = np.concatenate((audio, np.zeros(64000, np.float32)))
    speech_seconds = (audio.size - 64000) / 16000
    audio = resample(audio, 16000, rate)
    received, done = [], asyncio.Event()
    started = time.perf_counter()
    async with websockets.connect(url, max_size=1048576) as socket:
        async def read():
            try:
                async for text in socket:
                    msg = json.loads(text)
                    msg["received_seconds"] = round(time.perf_counter() - started, 3)
                    received.append(msg)
                    if msg.get("type") in ("final", "error"):
                        done.set()
                        return
            finally:
                done.set()

        reader = asyncio.create_task(read())
        frame_size = rate // 2
        for offset in range(0, len(audio), frame_size):
            chunk = audio[offset:offset+frame_size]
            pcm = (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()
            await socket.send(json.dumps({"type": "audio", "rate": rate,
                                          "data": base64.b64encode(pcm).decode()}))
            delay = started + (offset + chunk.size) / rate - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        # Stay in listening mode until the live worker catches the tail.
        # This specifically verifies improvement without pressing Stop.
        catch_up_start = time.perf_counter()
        target = audio.size / rate
        while not any(m.get("type") == "update" and
                      m.get("decoded_duration_sec", 0) >= target - .01 for m in received):
            if time.perf_counter() - catch_up_start > 15:
                raise AssertionError("Live worker did not drain the final partial window without Stop")
            await asyncio.sleep(.1)
        stop_started = time.perf_counter()
        await socket.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(done.wait(), timeout=45)
        await reader
    final = next((msg for msg in reversed(received) if msg.get("type") == "final"), None)
    assert final is not None, received[-2:]
    updates = [m for m in received if m.get("type") == "update" and m.get("arabic")]
    live = updates[-1] if updates else {"arabic": "", "english": ""}
    return {"label": label, "reference": reference,
            "speech_seconds": round(speech_seconds, 3), "capture_rate": rate,
            "first_update_seconds": updates[0]["received_seconds"] if updates else None,
            "last_live_wer": round(errors(reference, live["arabic"])[0] / max(1, len(words(reference))), 4),
            "last_live_arabic": live["arabic"], "last_live_english": live["english"],
            "final_arabic": final["arabic"], "final_english": final["english"],
            "final_wer": round(errors(reference, final["arabic"])[0] / max(1, len(words(reference))), 4),
            "stop_seconds": round(time.perf_counter() - stop_started, 3), "messages": received}


async def lifecycle(url):
    for _ in range(20):
        async with websockets.connect(url) as socket:
            await socket.recv()
            await socket.send('{"type":"stop"}')
            while json.loads(await socket.recv()).get("type") != "final":
                pass
    async with websockets.connect(url) as socket:
        await socket.recv()
        pcm = base64.b64encode(bytes(16000)).decode()
        for _ in range(60):
            await socket.send(json.dumps({"type": "audio", "rate": 16000, "data": pcm}))
        # No stop command: reaching the 30s cap must finalize automatically.
        async with asyncio.timeout(45):
            while True:
                msg = json.loads(await socket.recv())
                if msg.get("type") == "final":
                    assert msg["duration_sec"] == 30 and not msg["arabic"]
                    break
    return {"empty_start_stop_cycles": 20, "automatic_30s_silence_limit": "passed"}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws?mode=live")
    args = parser.parse_args()
    samples = json.loads((args.work_dir / "manifest.json").read_text(encoding="utf-8"))
    report = {"method": "500ms PCM frames paced at wall-clock microphone speed over the live WebSocket",
              "cases": []}
    for row, noisy in ((0, False), (1, True)):
        sample = samples[row]
        audio = load_audio(sample["path"], 16000)
        if noisy:
            rng = np.random.default_rng(2701)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            audio = audio + rng.standard_normal(audio.size).astype(np.float32) * rms / (10 ** (12 / 20))
        result = await replay(args.url, audio, sample["reference"].replace("^", "ث"),
                              f"MSA row {row}" + (" + synthetic white noise 12dB SNR" if noisy else " clean"),
                              rate=16000 if noisy else 48000)
        report["cases"].append(result)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("label", "first_update_seconds", "last_live_wer", "final_wer", "stop_seconds")}), flush=True)
    result = await replay(args.url, np.zeros(16000, np.float32), "", "digital silence")
    assert not result["final_arabic"] and not result["final_english"]
    report["cases"].append(result)
    report["lifecycle"] = await lifecycle(args.url)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["lifecycle"]), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
