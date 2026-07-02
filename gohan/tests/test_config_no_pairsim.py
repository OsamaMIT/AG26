from pathlib import Path

import pytest

from gohan.utils.config import ConfigError, load_config_bundle, load_yaml, resolve_config_path


def test_load_yaml_and_resolve_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("env:\n  track_config: tracks/test.yaml\n", encoding="utf-8")
    config = load_yaml(config_path)
    resolved = resolve_config_path(config, ("env", "track_config"), tmp_path)
    assert resolved == (tmp_path / "tracks/test.yaml").resolve()


def test_missing_yaml_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "missing.yaml")


def test_default_config_bundle_loads_awsim():
    config = load_config_bundle()
    assert config["env"]["simulator"] == "awsim_racing"
    assert config["env"]["vehicle"] == "dallara_av21r"
    assert config["ros"]["node_name"] == "gohan_awsim_bridge"


def test_no_stale_simulator_references_in_project_text():
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        "PAIR" + "Sim",
        "pair" + "sim",
        "Purdue-" + "AI-Racing-Simulator",
        "Purdue AI Racing " + "Simulator",
        "AV-" + "24",
    ]
    excluded_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "runs",
        "build",
        "dist",
    }
    excluded_suffixes = {".pyc", ".pyo", ".so", ".zip", ".png", ".jpg", ".jpeg", ".parquet"}
    failures: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in excluded_dirs or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix.lower() in excluded_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in forbidden:
            if needle in text:
                failures.append(f"{path.relative_to(root)} contains stale simulator reference")
                break

    assert not failures, "\n".join(failures)
