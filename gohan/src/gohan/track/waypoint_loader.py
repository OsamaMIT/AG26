"""Load track waypoints from YAML or CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(slots=True)
class WaypointData:
    """Waypoint data loaded from disk."""

    name: str
    points: np.ndarray
    yaws: np.ndarray | None
    track_width_m: float
    closed_loop: bool


def load_waypoints(path: str | Path) -> WaypointData:
    """Load waypoints from a YAML or CSV file."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Track waypoint file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported track waypoint format: {path.suffix}")


def _load_yaml(path: Path) -> WaypointData:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    waypoints = data.get("waypoints", [])
    points, yaws = _parse_waypoint_rows(waypoints)
    return WaypointData(
        name=str(data.get("name", path.stem)),
        points=points,
        yaws=yaws,
        track_width_m=float(data.get("track_width_m", 12.0)),
        closed_loop=bool(data.get("closed_loop", True)),
    )


def _load_csv(path: Path) -> WaypointData:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    points, yaws = _parse_waypoint_rows(rows)
    return WaypointData(name=path.stem, points=points, yaws=yaws, track_width_m=12.0, closed_loop=True)


def _parse_waypoint_rows(rows: list[Any]) -> tuple[np.ndarray, np.ndarray | None]:
    points: list[tuple[float, float]] = []
    yaws: list[float] = []
    has_yaw = False
    for row in rows:
        if isinstance(row, dict):
            x = row.get("x", row.get("X"))
            y = row.get("y", row.get("Y"))
            yaw = row.get("yaw", row.get("heading"))
        else:
            if len(row) < 2:
                raise ValueError(f"Waypoint rows must include at least x and y: {row}")
            x, y = row[0], row[1]
            yaw = row[2] if len(row) > 2 else None
        if x is None or y is None:
            raise ValueError(f"Waypoint row missing x/y: {row}")
        points.append((float(x), float(y)))
        if yaw is not None:
            has_yaw = True
            yaws.append(float(yaw))
        else:
            yaws.append(0.0)
    if len(points) < 2:
        raise ValueError("Track map requires at least two waypoints")
    point_array = np.asarray(points, dtype=float)
    yaw_array = np.asarray(yaws, dtype=float) if has_yaw else None
    return point_array, yaw_array
