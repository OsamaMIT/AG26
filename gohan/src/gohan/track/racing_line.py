"""Racing-line helpers reserved for future GOHAN curriculum work."""

from __future__ import annotations

from pathlib import Path

from gohan.track.track_map import TrackMap


def load_racing_line(path: str | Path) -> TrackMap:
    """Load a racing line as a TrackMap."""
    return TrackMap.from_file(path)
