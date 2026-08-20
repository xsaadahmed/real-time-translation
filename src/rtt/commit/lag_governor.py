"""Lag governor: adapts the commit threshold theta to how far behind the
speaker the pipeline currently is (README "Commitment algorithm" step 6).

Without this, a fast speaker (or a slow tick) makes lag grow without bound:
theta stays high, fewer prefixes clear the bar, less gets committed per
tick, and the backlog compounds. Decaying theta under lag trades peak
accuracy for staying caught up; raising it back when lag is low recovers
that accuracy the moment there's slack to spend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LagGovernorConfig:
    base_theta: float = 0.97
    #: README's named decay ladder, highest (normal) to lowest (most lag).
    decay_steps: tuple[float, ...] = (0.97, 0.90, 0.80)
    low_lag_sec: float = 1.2
    high_lag_sec: float = 3.0
    hard_ceiling_sec: float = 3.5

    def __post_init__(self) -> None:
        if self.decay_steps[0] != self.base_theta:
            raise ValueError("decay_steps[0] must equal base_theta - it's the normal-operation rung")
        if not all(a > b for a, b in zip(self.decay_steps, self.decay_steps[1:])):
            raise ValueError("decay_steps must be strictly decreasing")
        if not (0 < self.low_lag_sec < self.high_lag_sec < self.hard_ceiling_sec):
            raise ValueError("expected low_lag_sec < high_lag_sec < hard_ceiling_sec")


@dataclass
class LagGovernor:
    """Stateful: theta moves one rung per :meth:`update` call, not straight
    to an extreme, so a single noisy lag reading can't swing the operating
    point instantly. Call once per tick with the current lag.
    """

    config: LagGovernorConfig = field(default_factory=LagGovernorConfig)
    theta: float = field(init=False)

    def __post_init__(self) -> None:
        self.theta = self.config.base_theta

    def _rung_index(self) -> int:
        return self.config.decay_steps.index(self.theta)

    def update(self, lag_sec: float) -> float:
        """Advance theta by at most one rung towards the state ``lag_sec``
        implies, and return the new value.
        """
        steps = self.config.decay_steps
        idx = self._rung_index()

        if lag_sec > self.config.high_lag_sec and idx < len(steps) - 1:
            idx += 1
        elif lag_sec < self.config.low_lag_sec and idx > 0:
            idx -= 1

        self.theta = steps[idx]
        return self.theta

    def force_commit(self, lag_sec: float) -> bool:
        """True once lag has crossed the hard ceiling - the commit policy
        must emit *something* this tick regardless of theta.
        """
        return lag_sec >= self.config.hard_ceiling_sec

    def reset(self) -> None:
        self.theta = self.config.base_theta


__all__ = ["LagGovernor", "LagGovernorConfig"]
