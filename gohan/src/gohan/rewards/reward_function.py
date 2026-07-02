"""Reward function for telemetry-only racing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gohan.telemetry.telemetry_state import TrackFeatures, VehicleCommand, VehicleTelemetry
from gohan.utils.math_utils import clamp


@dataclass
class RacingReward:
    """Compute racing reward components from telemetry and track features."""

    config: dict[str, Any] | None = None
    previous_progress_m: float | None = None
    previous_command: VehicleCommand = field(default_factory=VehicleCommand)
    stuck_speed_mps: float = 0.5

    def __post_init__(self) -> None:
        cfg = dict((self.config or {}).get("reward", self.config or {}))
        self.progress_weight = float(cfg.get("progress_weight", 8.0))
        self.speed_weight = float(cfg.get("speed_weight", 1.0))
        self.lateral_error_weight = float(cfg.get("lateral_error_weight", 2.0))
        self.heading_error_weight = float(cfg.get("heading_error_weight", 1.5))
        self.yaw_rate_weight = float(cfg.get("yaw_rate_weight", 0.2))
        self.action_smoothness_weight = float(cfg.get("action_smoothness_weight", 0.1))
        self.off_track_penalty = float(cfg.get("off_track_penalty", 25.0))
        self.collision_penalty = float(cfg.get("collision_penalty", 50.0))
        self.reverse_penalty = float(cfg.get("reverse_penalty", 5.0))
        self.stuck_penalty = float(cfg.get("stuck_penalty", 10.0))
        self.low_speed_penalty = float(cfg.get("low_speed_penalty", 0.1))

    def reset(self) -> None:
        self.previous_progress_m = None
        self.previous_command = VehicleCommand()

    def compute(
        self,
        telemetry: VehicleTelemetry,
        track_features: TrackFeatures,
        command: VehicleCommand,
        dt: float,
    ) -> tuple[float, dict[str, float]]:
        """Return scalar reward and named reward components."""
        progress_delta = self._progress_delta(track_features)
        progress = self.progress_weight * progress_delta
        speed = self.speed_weight * clamp(telemetry.speed_mps / 90.0, 0.0, 1.0) * max(dt, 1e-6)
        lateral = -self.lateral_error_weight * abs(track_features.lateral_error_m) / 20.0
        heading = -self.heading_error_weight * abs(track_features.heading_error_rad) / 3.14159
        yaw = -self.yaw_rate_weight * abs(telemetry.yaw_rate_radps) / 3.0
        smoothness = -self.action_smoothness_weight * (
            abs(command.steering - self.previous_command.steering)
            + abs(command.throttle - self.previous_command.throttle)
            + abs(command.brake - self.previous_command.brake)
        )
        off_track = -self.off_track_penalty if (telemetry.off_track or track_features.off_track) else 0.0
        collision = -self.collision_penalty if telemetry.collision_detected else 0.0
        reverse = -self.reverse_penalty * abs(progress_delta) if progress_delta < -1e-3 else 0.0
        stuck = -self.stuck_penalty * max(dt, 1e-6) if telemetry.speed_mps < self.stuck_speed_mps else 0.0
        low_speed = -self.low_speed_penalty * max(dt, 1e-6) if telemetry.speed_mps < 2.0 else 0.0

        components = {
            "progress": float(progress),
            "speed": float(speed),
            "lateral_error": float(lateral),
            "heading_error": float(heading),
            "yaw_instability": float(yaw),
            "action_smoothness": float(smoothness),
            "off_track": float(off_track),
            "collision": float(collision),
            "reverse": float(reverse),
            "stuck": float(stuck),
            "low_speed": float(low_speed),
        }
        total = float(sum(components.values()))
        self.previous_progress_m = track_features.progress_m
        self.previous_command = command
        return total, components

    def _progress_delta(self, track_features: TrackFeatures) -> float:
        if self.previous_progress_m is None:
            return 0.0
        delta = float(track_features.progress_m - self.previous_progress_m)
        if track_features.lap_length_m > 0.0:
            if delta < -0.5 * track_features.lap_length_m:
                delta += track_features.lap_length_m
            elif delta > 0.5 * track_features.lap_length_m:
                delta -= track_features.lap_length_m
        return delta
