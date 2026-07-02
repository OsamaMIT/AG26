"""Episode report formatting."""

from __future__ import annotations

from gohan.telemetry.metrics import summarize_episode
from gohan.telemetry.telemetry_state import EpisodeStepRecord


def format_episode_report(records: list[EpisodeStepRecord]) -> str:
    """Return a short human-readable episode summary."""
    summary = summarize_episode(records)
    return "\n".join(f"{key}: {value}" for key, value in summary.items())
