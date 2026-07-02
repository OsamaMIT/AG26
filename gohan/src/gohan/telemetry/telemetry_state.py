"""Shared telemetry and command dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VehicleCommand:
    """Normalized vehicle command used inside GOHAN."""

    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def throttle_brake(self) -> float:
        """Return the combined throttle/brake convention used by the policy."""
        return float(self.throttle - self.brake)


@dataclass(slots=True)
class VehicleTelemetry:
    """Best-effort normalized telemetry from AWSIM ROS2 topics."""

    timestamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    speed_mps: float = 0.0
    longitudinal_velocity_mps: float = 0.0
    lateral_velocity_mps: float = 0.0
    yaw_rate_radps: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    steering_angle: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    gear: int = 0
    rpm: float = 0.0
    lap_count: int = 0
    lap_time: float = 0.0
    collision_detected: bool = False
    off_track: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrackFeatures:
    """Track-relative features used by observation and reward code."""

    lateral_error_m: float = 0.0
    heading_error_rad: float = 0.0
    track_curvature: float = 0.0
    distance_to_left_boundary_m: float = 0.0
    distance_to_right_boundary_m: float = 0.0
    lookahead_points_local: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    progress_m: float = 0.0
    lap_progress: float = 0.0
    lap_length_m: float = 0.0
    nearest_index: int = 0
    off_track: bool = False


@dataclass(slots=True)
class ObservationPacket:
    """Observation vector and its supporting track features."""

    observation: Any
    telemetry: VehicleTelemetry
    track_features: TrackFeatures


@dataclass(slots=True)
class EpisodeStepRecord:
    """One row of episode telemetry suitable for logging."""

    timestamp: float
    episode: int
    step: int
    x: float
    y: float
    yaw: float
    speed_mps: float
    progress_m: float
    lap_progress: float
    lateral_error_m: float
    heading_error_rad: float
    steering: float
    throttle: float
    brake: float
    reward: float
    terminated: bool
    truncated: bool
    collision_detected: bool
    off_track: bool
    info: dict[str, Any] = field(default_factory=dict)
