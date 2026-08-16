"""Audio loading, resampling and writing.

Deliberately avoids shelling out to ffmpeg. WAV/FLAC/OGG go through
``soundfile``; anything else (mp3, m4a, webm from a browser mic) falls back to
PyAV, which ``faster-whisper`` already installs. The canonical in-memory format
throughout the pipeline is mono ``float32`` in [-1, 1].
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .config import ASR_SAMPLE_RATE

AudioArray = np.ndarray


def to_mono_float32(audio: np.ndarray) -> AudioArray:
    """Normalise any reasonable array layout/dtype to mono float32 in [-1, 1]."""
    data = np.asarray(audio)

    if data.dtype.kind in "iu":
        # Scale by the dtype's positive range so int16 -> [-1, 1].
        max_value = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_value
    else:
        data = data.astype(np.float32, copy=False)

    if data.ndim > 1:
        # soundfile and gradio both hand back (samples, channels).
        channel_axis = int(np.argmin(data.shape))
        data = data.mean(axis=channel_axis)

    return np.ascontiguousarray(data.reshape(-1), dtype=np.float32)


def resample(audio: AudioArray, source_rate: int, target_rate: int) -> AudioArray:
    """Polyphase resample. Returns the input untouched when rates already match."""
    if source_rate == target_rate:
        return audio
    if audio.size == 0:
        return audio

    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return np.ascontiguousarray(resample_poly(audio, up, down).astype(np.float32))


def _load_with_pyav(path: Path, target_rate: int) -> AudioArray:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        # Flush any samples buffered inside the resampler.
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return to_mono_float32(np.concatenate(chunks))


def load_audio(path: str | Path, target_rate: int = ASR_SAMPLE_RATE) -> AudioArray:
    """Read an audio file from disk as mono float32 at ``target_rate``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        import soundfile as sf

        data, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return resample(to_mono_float32(data), source_rate, target_rate)
    except Exception:
        # Compressed containers that libsndfile cannot open.
        return _load_with_pyav(path, target_rate)


def save_wav(path: str | Path, audio: AudioArray, sample_rate: int) -> Path:
    """Write mono float32 audio as a 16-bit PCM WAV file."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="PCM_16")
    return path


def to_int16(audio: AudioArray) -> np.ndarray:
    """Convert float32 audio to int16, which is what Gradio's player expects."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def duration_seconds(audio: AudioArray, sample_rate: int) -> float:
    return float(audio.size) / float(sample_rate) if sample_rate else 0.0


__all__ = [
    "AudioArray",
    "duration_seconds",
    "load_audio",
    "resample",
    "save_wav",
    "to_int16",
    "to_mono_float32",
]
