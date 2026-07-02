"""Small numeric helpers used across GOHAN."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a scalar to an inclusive range."""
    return float(min(max(float(value), float(low)), float(high)))


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def safe_norm(values: Iterable[float]) -> float:
    """Return the Euclidean norm for a short iterable."""
    arr = np.asarray(list(values), dtype=float)
    return float(np.linalg.norm(arr))


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convert a quaternion to roll, pitch, yaw in radians."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def finite_or(value: float, default: float = 0.0) -> float:
    """Return a finite float, or a default if the input is NaN/inf."""
    value = float(value)
    return value if math.isfinite(value) else float(default)
