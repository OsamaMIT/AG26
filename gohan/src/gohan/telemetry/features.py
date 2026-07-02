"""Feature extraction from telemetry and track maps."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from gohan.telemetry.telemetry_state import ObservationPacket, TrackFeatures, VehicleCommand, VehicleTelemetry
from gohan.track.track_map import TrackMap
from gohan.utils.math_utils import clamp

OBSERVATION_SIZE = 20
_warned_non_finite = False


def build_observation(
    telemetry: VehicleTelemetry,
    track_map: TrackMap,
    previous_command: VehicleCommand | None = None,
    config: dict[str, Any] | None = None,
) -> ObservationPacket:
    """Build GOHAN's normalized 20D observation vector from telemetry."""
    previous_command = previous_command or VehicleCommand()
    config = config or {}
    vehicle_cfg = dict(config.get("vehicle", {}))

    max_speed = float(vehicle_cfg.get("max_speed_mps", 90.0))
    max_yaw_rate = float(vehicle_cfg.get("max_yaw_rate_radps", 3.0))
    max_lateral_error = float(vehicle_cfg.get("max_lateral_error_m", 20.0))
    lookahead_norm = float(vehicle_cfg.get("lookahead_norm_m", 50.0))

    x = _finite(telemetry.x)
    y = _finite(telemetry.y)
    yaw = _finite(telemetry.yaw)
    speed = _finite(telemetry.speed_mps)
    longitudinal_velocity = _finite(telemetry.longitudinal_velocity_mps)
    lateral_velocity = _finite(telemetry.lateral_velocity_mps)
    yaw_rate = _finite(telemetry.yaw_rate_radps)

    query = track_map.nearest(x, y)
    heading_error = track_map.heading_error(yaw, query)
    curvature = track_map.local_curvature(query.segment_index)
    left_dist, right_dist = track_map.boundary_distances(query.lateral_error_m)
    lookahead = track_map.lookahead_points_local(
        query,
        x,
        y,
        yaw,
    )
    off_track = telemetry.off_track or track_map.is_off_track(query.lateral_error_m)

    track_features = TrackFeatures(
        lateral_error_m=query.lateral_error_m,
        heading_error_rad=heading_error,
        track_curvature=curvature,
        distance_to_left_boundary_m=left_dist,
        distance_to_right_boundary_m=right_dist,
        lookahead_points_local=lookahead,
        progress_m=query.progress_m,
        lap_progress=query.lap_progress,
        lap_length_m=track_map.length_m,
        nearest_index=query.segment_index,
        off_track=off_track,
    )

    lap_angle = 2.0 * math.pi * query.lap_progress
    values = [
        _norm(query.lateral_error_m, max_lateral_error),
        math.sin(heading_error),
        math.cos(heading_error),
        _norm(speed, max_speed),
        _norm(longitudinal_velocity, max_speed),
        _norm(lateral_velocity, max_speed),
        _norm(yaw_rate, max_yaw_rate),
        clamp(curvature * 100.0, -1.0, 1.0),
        _boundary_norm(left_dist, track_map.track_width_m),
        _boundary_norm(right_dist, track_map.track_width_m),
        _norm(lookahead[0][0], lookahead_norm),
        _norm(lookahead[0][1], lookahead_norm),
        _norm(lookahead[1][0], lookahead_norm),
        _norm(lookahead[1][1], lookahead_norm),
        _norm(lookahead[2][0], lookahead_norm),
        _norm(lookahead[2][1], lookahead_norm),
        clamp(previous_command.steering, -1.0, 1.0),
        clamp(previous_command.throttle_brake(), -1.0, 1.0),
        math.sin(lap_angle),
        math.cos(lap_angle),
    ]
    observation = _sanitize_observation(np.asarray(values, dtype=np.float32))
    return ObservationPacket(observation=observation, telemetry=telemetry, track_features=track_features)


def _norm(value: float, scale: float) -> float:
    if abs(scale) <= 1e-9:
        return 0.0
    return clamp(float(value) / float(scale), -1.0, 1.0)


def _finite(value: float, default: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else default


def _boundary_norm(distance_m: float, track_width_m: float) -> float:
    # 1.0 near a full track width of clearance, 0.0 at center-ish, -1.0 past boundary.
    return clamp((float(distance_m) / max(float(track_width_m), 1e-9)) * 2.0 - 1.0, -1.0, 1.0)


def _sanitize_observation(observation: np.ndarray) -> np.ndarray:
    global _warned_non_finite
    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"Observation must have shape ({OBSERVATION_SIZE},), got {observation.shape}")
    if not np.all(np.isfinite(observation)):
        if not _warned_non_finite:
            warnings.warn("Non-finite observation values detected; replacing NaN/inf with 0.", RuntimeWarning)
            _warned_non_finite = True
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(observation, -1.0, 1.0).astype(np.float32)
