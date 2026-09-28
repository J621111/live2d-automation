"""Resolve configurable local editor state without creating directories during import."""

from __future__ import annotations

import os
from pathlib import Path


def state_directory() -> Path:
    """Return LIVE2D_AUTOMATION_STATE_DIR, or output beneath the current directory."""
    return (
        Path(os.environ.get("LIVE2D_AUTOMATION_STATE_DIR") or Path.cwd() / "output")
        .expanduser()
        .absolute()
    )


def session_directory() -> Path:
    """Return an explicit Swing session directory or the local state default."""
    return (
        Path(
            os.environ.get("LIVE2D_CUBISM_SESSION_DIR")
            or state_directory() / "swing-bridge/session"
        )
        .expanduser()
        .absolute()
    )
