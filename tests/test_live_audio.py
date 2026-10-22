import numpy as np

from rtt.live.session import merge_audio


def test_websocket_audio_deltas_are_always_appended():
    first = np.ones(8_000, dtype=np.float32)
    second = np.full(9_000, 0.5, dtype=np.float32)
    merged = merge_audio(first, second, 16_000)
    assert merged.size == 17_000
    assert np.array_equal(merged[:8_000], first)


def test_identical_audio_deltas_are_not_deduplicated():
    frame = np.ones(8_000, dtype=np.float32)
    assert merge_audio(frame, frame, 16_000).size == 16_000
