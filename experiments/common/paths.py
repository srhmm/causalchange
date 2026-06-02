from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "CAUSALCHANGE_DATA_DIR"
RESULTS_DIR_ENV = "CAUSALCHANGE_RESULTS_DIR"


def repo_root(start: Path | None = None) -> Path:
    """Return the repository root by walking upward until pyproject/.git is found."""
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate

    return current


def data_dir(*parts: str, create: bool = False) -> Path:
    """Return the configured data directory.

    Default: ``<repo>/data``.
    Override with ``CAUSALCHANGE_DATA_DIR``.
    """
    root = Path(os.environ.get(DATA_DIR_ENV, repo_root() / "data")).expanduser().resolve()
    path = root.joinpath(*parts)

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def results_dir(*parts: str, create: bool = True) -> Path:
    """Return the configured results directory.

    Default: ``<repo>/results``.
    Override with ``CAUSALCHANGE_RESULTS_DIR``.
    """
    root = Path(os.environ.get(RESULTS_DIR_ENV, repo_root() / "results")).expanduser().resolve()
    path = root.joinpath(*parts)

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path


def ensure_dir(path: Path | str) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path | str) -> Path:
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Expected a file, got: {path}")

    return path


def require_dir(path: Path | str) -> Path:
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Required directory does not exist: {path}")

    if not path.is_dir():
        raise FileNotFoundError(f"Expected a directory, got: {path}")

    return path
