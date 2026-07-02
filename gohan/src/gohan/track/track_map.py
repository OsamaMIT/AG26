"""Track map utilities for feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gohan.track.geometry import (
    curvature_from_points,
    heading_between,
    heading_error,
    project_point_to_segment,
    signed_lateral_error,
    transform_world_to_local,
)
from gohan.track.waypoint_loader import WaypointData, load_waypoints


@dataclass(slots=True)
class TrackQuery:
    """Nearest-track query result."""

    nearest_point: np.ndarray
    segment_index: int
    segment_t: float
    progress_m: float
    lap_progress: float
    heading_rad: float
    lateral_error_m: float
    distance_m: float


class TrackMap:
    """Centerline/raceline track map with progress and lookahead helpers."""

    def __init__(self, waypoint_data: WaypointData) -> None:
        self.name = waypoint_data.name
        self.points = np.asarray(waypoint_data.points, dtype=float)
        self.yaws = waypoint_data.yaws
        self.track_width_m = float(waypoint_data.track_width_m)
        self.closed_loop = bool(waypoint_data.closed_loop)
        self._segments = self._build_segments()
        self._segment_lengths = np.asarray(
            [float(np.linalg.norm(self.points[b] - self.points[a])) for a, b in self._segments],
            dtype=float,
        )
        self._cumulative = np.concatenate([[0.0], np.cumsum(self._segment_lengths)])
        self.length_m = float(self._cumulative[-1])
        if self.length_m <= 1e-9:
            raise ValueError("Track length must be positive")

    @classmethod
    def from_file(cls, path: str | Path) -> "TrackMap":
        return cls(load_waypoints(path))

    def nearest(self, x: float, y: float) -> TrackQuery:
        """Find the nearest projected centerline point."""
        point = np.asarray([float(x), float(y)], dtype=float)
        best_index = 0
        best_projection = None
        for index, (a_idx, b_idx) in enumerate(self._segments):
            projection = project_point_to_segment(point, self.points[a_idx], self.points[b_idx])
            if best_projection is None or projection.distance < best_projection.distance:
                best_projection = projection
                best_index = index
        assert best_projection is not None
        progress = float(self._cumulative[best_index] + best_projection.t * self._segment_lengths[best_index])
        heading = self.segment_heading(best_index)
        lateral = signed_lateral_error(point, best_projection.point, heading)
        return TrackQuery(
            nearest_point=best_projection.point,
            segment_index=best_index,
            segment_t=best_projection.t,
            progress_m=progress,
            lap_progress=(progress / self.length_m) % 1.0,
            heading_rad=heading,
            lateral_error_m=lateral,
            distance_m=best_projection.distance,
        )

    def segment_heading(self, segment_index: int) -> float:
        """Return heading for a segment."""
        a_idx, b_idx = self._segments[segment_index % len(self._segments)]
        if self.yaws is not None and 0 <= a_idx < len(self.yaws):
            return float(self.yaws[a_idx])
        return heading_between(self.points[a_idx], self.points[b_idx])

    def heading_error(self, vehicle_yaw: float, query: TrackQuery) -> float:
        return heading_error(vehicle_yaw, query.heading_rad)

    def local_curvature(self, segment_index: int) -> float:
        """Estimate local signed curvature near a segment."""
        center = segment_index % len(self.points)
        prev_idx = (center - 1) % len(self.points) if self.closed_loop else max(center - 1, 0)
        next_idx = (center + 1) % len(self.points) if self.closed_loop else min(center + 1, len(self.points) - 1)
        return curvature_from_points(self.points[prev_idx], self.points[center], self.points[next_idx])

    def boundary_distances(self, lateral_error_m: float) -> tuple[float, float]:
        """Estimate left and right boundary distances from centerline error."""
        half_width = self.track_width_m / 2.0
        left = half_width - lateral_error_m
        right = half_width + lateral_error_m
        return float(left), float(right)

    def is_off_track(self, lateral_error_m: float) -> bool:
        return abs(float(lateral_error_m)) > self.track_width_m / 2.0

    def point_at_progress(self, progress_m: float) -> np.ndarray:
        """Interpolate a centerline point at a progress distance."""
        progress = float(progress_m)
        if self.closed_loop:
            progress = progress % self.length_m
        else:
            progress = float(np.clip(progress, 0.0, self.length_m))
        segment_index = int(np.searchsorted(self._cumulative, progress, side="right") - 1)
        segment_index = int(np.clip(segment_index, 0, len(self._segments) - 1))
        a_idx, b_idx = self._segments[segment_index]
        length = max(self._segment_lengths[segment_index], 1e-9)
        t = (progress - self._cumulative[segment_index]) / length
        return self.points[a_idx] + float(np.clip(t, 0.0, 1.0)) * (self.points[b_idx] - self.points[a_idx])

    def lookahead_points_local(
        self,
        query: TrackQuery,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        distances_m: tuple[float, float, float] = (10.0, 25.0, 50.0),
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Return three lookahead points in the vehicle frame."""
        origin = np.asarray([float(vehicle_x), float(vehicle_y)], dtype=float)
        local_points: list[tuple[float, float]] = []
        for distance in distances_m:
            world = self.point_at_progress(query.progress_m + distance)
            local_points.append(transform_world_to_local(world, origin, vehicle_yaw))
        return (local_points[0], local_points[1], local_points[2])

    def _build_segments(self) -> list[tuple[int, int]]:
        segments = [(i, i + 1) for i in range(len(self.points) - 1)]
        if self.closed_loop:
            segments.append((len(self.points) - 1, 0))
        return segments
