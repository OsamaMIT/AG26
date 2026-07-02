"""Path helpers for GOHAN."""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    """Return the installed source package root."""
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Return the GOHAN project root when running from source."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path relative to the GOHAN project root or a supplied base."""
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    root = Path(base).expanduser().resolve() if base is not None else project_root()
    return (root / raw).resolve()


def make_run_dir(run_name: str, root: str | Path | None = None) -> Path:
    """Create and return runs/<run_name>."""
    runs_root = resolve_project_path("runs", root) if root else project_root() / "runs"
    path = runs_root / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
