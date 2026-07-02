"""Map Stable-Baselines3 policy actions to normalized vehicle commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gohan.telemetry.telemetry_state import VehicleCommand
from gohan.utils.math_utils import clamp


@dataclass(slots=True)
class ActionMapper:
    """Convert the GOHAN continuous action vector into a VehicleCommand."""

    steering_limit: float = 1.0
    throttle_limit: float = 1.0
    brake_limit: float = 1.0

    def map_action(self, action: np.ndarray | Sequence[float]) -> VehicleCommand:
        """Map [steering, throttle_brake] in [-1, 1] to steering/throttle/brake."""
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Action contains NaN or infinity: {arr}")

        steering = clamp(float(arr[0]), -self.steering_limit, self.steering_limit)
        throttle_brake = clamp(float(arr[1]), -1.0, 1.0)

        if throttle_brake >= 0.0:
            throttle = clamp(throttle_brake, 0.0, self.throttle_limit)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(abs(throttle_brake), 0.0, self.brake_limit)

        return VehicleCommand(steering=steering, throttle=throttle, brake=brake)


__all__ = ["ActionMapper", "VehicleCommand"]
