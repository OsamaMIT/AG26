"""Episode-level emergency checks."""

from __future__ import annotations

from dataclasses import dataclass

from gohan.telemetry.telemetry_state import TrackFeatures, VehicleTelemetry


@dataclass(slots=True)
class EmergencyState:
    """Counters used by racing environment termination and truncation logic."""

    off_track_steps: int = 0
    stuck_steps: int = 0
    spin_steps: int = 0

    def reset(self) -> None:
        self.off_track_steps = 0
        self.stuck_steps = 0
        self.spin_steps = 0


def update_emergency_state(
    state: EmergencyState,
    telemetry: VehicleTelemetry,
    track_features: TrackFeatures,
    max_yaw_rate_radps: float,
    stuck_speed_mps: float = 0.5,
) -> EmergencyState:
    """Update counters for off-track, stuck, and spin conditions."""
    state.off_track_steps = state.off_track_steps + 1 if (telemetry.off_track or track_features.off_track) else 0
    state.stuck_steps = state.stuck_steps + 1 if telemetry.speed_mps < stuck_speed_mps else 0
    state.spin_steps = state.spin_steps + 1 if abs(telemetry.yaw_rate_radps) > max_yaw_rate_radps else 0
    return state
