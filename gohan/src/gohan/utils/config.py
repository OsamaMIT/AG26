"""YAML configuration loading and path resolution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from gohan.utils.paths import project_root, resolve_project_path


class ConfigError(RuntimeError):
    """Raised when a GOHAN configuration file is invalid or unavailable."""


def load_yaml(path: str | Path, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    resolved = resolve_project_path(path, base_dir)
    if not resolved.exists():
        raise ConfigError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a mapping: {resolved}")
    return data


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge updates into a copy of base."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config_bundle(
    awsim_config: str | Path = "configs/awsim.yaml",
    training_config: str | Path | None = None,
    reward_config: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load the AWSIM, training, and reward configs into one dictionary."""
    base = Path(base_dir).resolve() if base_dir is not None else project_root()
    config = load_yaml(awsim_config, base)
    if training_config is not None:
        config = deep_update(config, load_yaml(training_config, base))
    if reward_config is not None:
        config = deep_update(config, load_yaml(reward_config, base))
    return config


def resolve_config_path(config: dict[str, Any], key_path: tuple[str, ...], base_dir: str | Path | None = None) -> Path:
    """Resolve a path value nested in a config mapping."""
    value: Any = config
    for key in key_path:
        if not isinstance(value, dict) or key not in value:
            joined = ".".join(key_path)
            raise ConfigError(f"Missing configuration key: {joined}")
        value = value[key]
    return resolve_project_path(value, base_dir)
