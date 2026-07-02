"""Telemetry summary metrics."""

from __future__ import annotations

from collections.abc import Iterable

from gohan.telemetry.telemetry_state import EpisodeStepRecord


def summarize_episode(records: Iterable[EpisodeStepRecord]) -> dict[str, float | int]:
    """Return basic aggregate metrics for one episode."""
    rows = list(records)
    if not rows:
        return {
            "steps": 0,
            "total_reward": 0.0,
            "mean_speed_mps": 0.0,
            "max_progress_m": 0.0,
            "off_track_events": 0,
            "collision_events": 0,
        }
    return {
        "steps": len(rows),
        "total_reward": float(sum(row.reward for row in rows)),
        "mean_speed_mps": float(sum(row.speed_mps for row in rows) / len(rows)),
        "max_progress_m": float(max(row.progress_m for row in rows)),
        "off_track_events": int(sum(1 for row in rows if row.off_track)),
        "collision_events": int(sum(1 for row in rows if row.collision_detected)),
    }
