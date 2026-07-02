"""Run directory helpers for training and evaluation."""

from __future__ import annotations

from pathlib import Path

from gohan.utils.paths import make_run_dir


class RunManager:
    """Small run directory wrapper."""

    def __init__(self, run_name: str, project_root: str | Path | None = None) -> None:
        self.run_dir = make_run_dir(run_name, project_root)
        self.models_dir = self.run_dir / "models"
        self.plots_dir = self.run_dir / "plots"
        self.models_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
