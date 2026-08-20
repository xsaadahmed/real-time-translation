"""In-process API metrics for sessions, failures, and finalize latency."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class ApiMetrics:
    sessions_started: int = 0
    sessions_completed: int = 0
    sessions_failed: int = 0
    sessions_disconnected: int = 0
    finalize_count: int = 0
    finalize_errors: int = 0
    finalize_latency_sum_sec: float = 0.0
    finalize_latency_max_sec: float = 0.0
    active_sessions: int = 0
    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def session_start(self) -> None:
        with self._lock:
            self.sessions_started += 1
            self.active_sessions += 1

    def session_complete(self) -> None:
        with self._lock:
            self.sessions_completed += 1
            self.active_sessions = max(0, self.active_sessions - 1)

    def session_fail(self) -> None:
        with self._lock:
            self.sessions_failed += 1
            self.active_sessions = max(0, self.active_sessions - 1)

    def session_disconnect(self) -> None:
        with self._lock:
            self.sessions_disconnected += 1
            self.active_sessions = max(0, self.active_sessions - 1)

    def record_finalize(self, latency_sec: float, *, ok: bool) -> None:
        with self._lock:
            self.finalize_count += 1
            if not ok:
                self.finalize_errors += 1
            self.finalize_latency_sum_sec += max(0.0, latency_sec)
            self.finalize_latency_max_sec = max(self.finalize_latency_max_sec, latency_sec)

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            avg = (
                self.finalize_latency_sum_sec / self.finalize_count
                if self.finalize_count
                else 0.0
            )
            return {
                "uptime_sec": round(time.time() - self.started_at, 1),
                "sessions_started": self.sessions_started,
                "sessions_completed": self.sessions_completed,
                "sessions_failed": self.sessions_failed,
                "sessions_disconnected": self.sessions_disconnected,
                "active_sessions": self.active_sessions,
                "finalize_count": self.finalize_count,
                "finalize_errors": self.finalize_errors,
                "finalize_latency_avg_sec": round(avg, 3),
                "finalize_latency_max_sec": round(self.finalize_latency_max_sec, 3),
            }


metrics = ApiMetrics()

__all__ = ["ApiMetrics", "metrics"]
