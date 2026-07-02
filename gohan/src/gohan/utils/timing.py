"""Timing utilities for ROS control loops and scripts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def now_s() -> float:
    """Return a monotonic timestamp in seconds."""
    return time.monotonic()


@dataclass
class RateTimer:
    """Simple wall-clock loop-rate helper."""

    hz: float
    _next_time: float = field(default_factory=now_s)

    @property
    def period_s(self) -> float:
        return 1.0 / max(float(self.hz), 1e-9)

    def sleep(self) -> None:
        self._next_time += self.period_s
        delay = self._next_time - now_s()
        if delay > 0.0:
            time.sleep(delay)
        else:
            self._next_time = now_s()
