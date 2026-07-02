"""Stable-Baselines3 callbacks for GOHAN."""

from __future__ import annotations

from pathlib import Path


def make_checkpoint_callback(save_freq: int, save_path: str | Path):
    """Return an SB3 checkpoint callback."""
    try:
        from stable_baselines3.common.callbacks import CheckpointCallback
    except Exception as exc:
        raise RuntimeError("stable-baselines3 is required for checkpoint callbacks.") from exc
    return CheckpointCallback(save_freq=int(save_freq), save_path=str(save_path), name_prefix="gohan_ppo")
