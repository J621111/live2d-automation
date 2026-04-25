"""Shared validation helpers for user-controlled paths and identifiers."""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = (PROJECT_ROOT / "output").resolve()
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InputValidationError(ValueError):
    """Raised when a user-controlled input is unsafe or invalid."""


def _configured_output_root() -> Path:
    override = os.getenv("LIVE2D_OUTPUT_ROOT")
    if not override:
        return DEFAULT_OUTPUT_ROOT
    resolved = Path(override).expanduser().resolve()
    if not is_relative_to(resolved, PROJECT_ROOT):
        raise InputValidationError("LIVE2D_OUTPUT_ROOT must stay inside the project directory.")
    return resolved


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


OUTPUT_ROOT = _configured_output_root()


def resolve_output_dir(output_dir: str | os.PathLike[str]) -> Path:
    """Resolve a user output directory under the configured output root."""

    output_text = str(output_dir).strip()
    if not output_text:
        raise InputValidationError("output_dir is required.")

    normalized_output_dir = output_text.replace("\\", "/")
    raw_path = Path(normalized_output_dir)

    if ".." in raw_path.parts:
        raise InputValidationError("output_dir must stay inside the project output directory.")

    output_root = OUTPUT_ROOT
    default_root_name = DEFAULT_OUTPUT_ROOT.name
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    elif raw_path.parts and raw_path.parts[0] in {"output", default_root_name}:
        resolved = (PROJECT_ROOT / raw_path).resolve()
    else:
        resolved = (output_root / raw_path).resolve()

    if not is_relative_to(resolved, output_root):
        raise InputValidationError("output_dir must stay inside the project output directory.")
    return resolved


def validate_model_name(model_name: str) -> str:
    if not MODEL_NAME_RE.match(model_name):
        raise InputValidationError(
            "model_name may only contain letters, numbers, underscores, and hyphens."
        )
    return model_name
