"""Matplotlib plots for GOHAN telemetry CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_telemetry_csv(input_csv: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate standard GOHAN telemetry plots."""
    input_csv = Path(input_csv).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_csv)
    outputs: list[Path] = []
    outputs.append(_plot_xy(data, output_dir / "xy_trajectory.png"))
    outputs.append(_plot_lines(data, ["speed_mps"], output_dir / "speed_over_time.png", "Speed (m/s)"))
    outputs.append(
        _plot_lines(data, ["steering", "throttle", "brake"], output_dir / "controls_over_time.png", "Command")
    )
    if "reward" in data.columns:
        outputs.append(_plot_lines(data, ["reward"], output_dir / "reward_over_time.png", "Reward"))
    if "lateral_error_m" in data.columns:
        outputs.append(_plot_lines(data, ["lateral_error_m"], output_dir / "lateral_error_over_time.png", "Meters"))
    if "progress_m" in data.columns:
        outputs.append(_plot_lines(data, ["progress_m"], output_dir / "progress_over_time.png", "Meters"))
    return outputs


def _plot_xy(data: pd.DataFrame, path: Path) -> Path:
    plt.figure(figsize=(8, 6))
    plt.plot(data["x"], data["y"], linewidth=1.5)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("XY trajectory")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_lines(data: pd.DataFrame, columns: list[str], path: Path, ylabel: str) -> Path:
    plt.figure(figsize=(10, 4))
    x = data["step"] if "step" in data.columns else data.index
    for column in columns:
        if column in data.columns:
            plt.plot(x, data[column], label=column)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
