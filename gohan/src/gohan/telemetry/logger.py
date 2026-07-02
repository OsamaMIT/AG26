"""CSV-first telemetry logging with optional Parquet export."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from gohan.telemetry.telemetry_state import EpisodeStepRecord


class TelemetryLogger:
    """Log GOHAN runs under runs/<run_name>/."""

    telemetry_fields = [
        "timestamp",
        "episode",
        "step",
        "x",
        "y",
        "yaw",
        "speed_mps",
        "progress_m",
        "lap_progress",
        "lateral_error_m",
        "heading_error_rad",
        "steering",
        "throttle",
        "brake",
        "reward",
        "terminated",
        "truncated",
        "collision_detected",
        "off_track",
    ]

    def __init__(
        self,
        run_dir: str | Path,
        config_paths: list[str | Path] | None = None,
        flush_every: int = 50,
        enable_parquet: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.run_dir / "models"
        self.plots_dir = self.run_dir / "plots"
        self.models_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.flush_every = int(flush_every)
        self.enable_parquet = bool(enable_parquet)
        self._rows_since_flush = 0
        self._reward_fields: list[str] | None = None
        self._copy_configs(config_paths or [])

        self.telemetry_path = self.run_dir / "telemetry.csv"
        self.reward_path = self.run_dir / "reward_components.csv"
        self.summary_path = self.run_dir / "episode_summary.csv"
        self._telemetry_file = self.telemetry_path.open("w", encoding="utf-8", newline="")
        self._telemetry_writer = csv.DictWriter(self._telemetry_file, fieldnames=self.telemetry_fields)
        self._telemetry_writer.writeheader()
        self._reward_file = self.reward_path.open("w", encoding="utf-8", newline="")
        self._reward_writer: csv.DictWriter | None = None

    def log_step(self, record: EpisodeStepRecord, reward_components: dict[str, float] | None = None) -> None:
        """Append one step to telemetry and reward CSV files."""
        self._telemetry_writer.writerow({field: getattr(record, field) for field in self.telemetry_fields})
        if reward_components is not None:
            self._write_reward(record, reward_components)
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every:
            self.flush()

    def log_episode_summary(self, episode: int, summary: dict[str, Any]) -> None:
        """Append one episode summary row."""
        exists = self.summary_path.exists() and self.summary_path.stat().st_size > 0
        with self.summary_path.open("a", encoding="utf-8", newline="") as handle:
            fieldnames = ["episode", *summary.keys()]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({"episode": episode, **summary})

    def flush(self) -> None:
        self._telemetry_file.flush()
        self._reward_file.flush()
        self._rows_since_flush = 0

    def close(self) -> None:
        self.flush()
        self._telemetry_file.close()
        self._reward_file.close()
        if self.enable_parquet:
            self._try_write_parquet()

    def _write_reward(self, record: EpisodeStepRecord, components: dict[str, float]) -> None:
        if self._reward_writer is None:
            self._reward_fields = ["timestamp", "episode", "step", *components.keys()]
            self._reward_writer = csv.DictWriter(self._reward_file, fieldnames=self._reward_fields)
            self._reward_writer.writeheader()
        assert self._reward_writer is not None
        self._reward_writer.writerow(
            {"timestamp": record.timestamp, "episode": record.episode, "step": record.step, **components}
        )

    def _copy_configs(self, config_paths: list[str | Path]) -> None:
        config_dir = self.run_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        for path in config_paths:
            src = Path(path).expanduser().resolve()
            if src.exists():
                shutil.copy2(src, config_dir / src.name)

    def _try_write_parquet(self) -> None:
        try:
            import pandas as pd

            pd.read_csv(self.telemetry_path).to_parquet(self.run_dir / "telemetry.parquet")
            if self.reward_path.exists() and self.reward_path.stat().st_size > 0:
                pd.read_csv(self.reward_path).to_parquet(self.run_dir / "reward_components.parquet")
        except Exception as exc:  # pragma: no cover - depends on optional pyarrow
            print(f"Parquet export skipped: {exc}")
