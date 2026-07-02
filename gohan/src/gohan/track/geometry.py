"""Geometry helpers for track-relative autonomous racing features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gohan.utils.math_utils import wrap_to_pi


@dataclass(slots=True)
class SegmentProjection:
    """Projection of a point onto a segment."""

    point: np.ndarray
    t: float
    distance: float


def heading_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return heading from point a to point b."""
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return float(math.atan2(delta[1], delta[0]))


def project_point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> SegmentProjection:
    """Project point onto segment a-b."""
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        projected = a.copy()
        t = 0.0
    else:
        t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        projected = a + t * ab
    return SegmentProjection(point=projected, t=t, distance=float(np.linalg.norm(point - projected)))


def signed_lateral_error(point: np.ndarray, center: np.ndarray, heading: float) -> float:
    """Return signed lateral error, positive to the left of path heading."""
    dx, dy = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    left_x = -math.sin(heading)
    left_y = math.cos(heading)
    return float(dx * left_x + dy * left_y)


def transform_world_to_local(point: np.ndarray, origin: np.ndarray, yaw: float) -> tuple[float, float]:
    """Transform a world point into vehicle-local coordinates."""
    dx, dy = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    c = math.cos(yaw)
    s = math.sin(yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return float(local_x), float(local_y)


def curvature_from_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Estimate signed curvature through three points."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ab = b - a
    bc = c - b
    ac = c - a
    lab = np.linalg.norm(ab)
    lbc = np.linalg.norm(bc)
    lac = np.linalg.norm(ac)
    denom = lab * lbc * lac
    if denom <= 1e-12:
        return 0.0
    cross = float(ab[0] * bc[1] - ab[1] * bc[0])
    return float(2.0 * cross / denom)


def heading_error(vehicle_yaw: float, track_heading: float) -> float:
    """Return vehicle heading error relative to track heading."""
    return wrap_to_pi(float(vehicle_yaw) - float(track_heading))
