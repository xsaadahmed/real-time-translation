"""Tests for in-process API metrics."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rtt.api.metrics import ApiMetrics


def test_metrics_session_lifecycle():
    m = ApiMetrics()
    m.session_start()
    m.session_start()
    assert m.snapshot()["active_sessions"] == 2
    m.session_complete()
    m.session_disconnect()
    snap = m.snapshot()
    assert snap["sessions_started"] == 2
    assert snap["sessions_completed"] == 1
    assert snap["sessions_disconnected"] == 1
    assert snap["active_sessions"] == 0


def test_metrics_finalize_latency():
    m = ApiMetrics()
    m.record_finalize(1.5, ok=True)
    m.record_finalize(0.5, ok=False)
    snap = m.snapshot()
    assert snap["finalize_count"] == 2
    assert snap["finalize_errors"] == 1
    assert snap["finalize_latency_avg_sec"] == 1.0
    assert snap["finalize_latency_max_sec"] == 1.5
