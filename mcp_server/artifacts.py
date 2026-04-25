"""Artifact writing and redaction helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp_server.validation import InputValidationError, is_relative_to, resolve_output_dir

JsonDict = dict[str, Any]

_SENSITIVE_KEY_FRAGMENTS = ("authorization", "api_key", "apikey", "token", "secret", "password")
_SENSITIVE_FLAG_FRAGMENTS = ("key", "token", "secret", "password")


def _is_sensitive_key(key: str) -> bool:
    lower = key.lower()
    return any(fragment in lower for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _is_sensitive_flag(value: str) -> bool:
    lower = value.lower().lstrip("-/")
    return any(fragment in lower for fragment in _SENSITIVE_FLAG_FRAGMENTS)


def _redact_flag_assignment(value: str) -> str | None:
    if "=" not in value:
        return None
    flag, assigned_value = value.split("=", 1)
    if not assigned_value:
        return None
    if _is_sensitive_flag(flag):
        return f"{flag}=<redacted>"
    return None


def redact_command(command: list[Any]) -> list[Any]:
    redacted: list[Any] = []
    redact_next = False
    for item in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if isinstance(item, str):
            redacted_assignment = _redact_flag_assignment(item)
            if redacted_assignment is not None:
                redacted.append(redacted_assignment)
                continue
            if _is_sensitive_flag(item):
                redacted.append(item)
                redact_next = True
                continue
        redacted.append(redact_sensitive(item))
    return redacted


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        payload: JsonDict = {}
        for key, item in value.items():
            key_text = str(key)
            payload[key_text] = (
                "<redacted>" if _is_sensitive_key(key_text) else redact_sensitive(item)
            )
        return payload
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    if isinstance(value, str) and value.lower().startswith("bearer "):
        return "Bearer <redacted>"
    return value


class ArtifactStore:
    """Write artifacts under the configured output root."""

    def __init__(self, output_dir: str | os.PathLike[str]) -> None:
        self.output_dir = resolve_output_dir(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def path(self, relative_path: str | os.PathLike[str]) -> Path:
        candidate = (self.output_dir / Path(relative_path)).resolve()
        if not is_relative_to(candidate, self.output_dir):
            raise InputValidationError("artifact path must stay inside the output directory.")
        return candidate

    def write_json(self, relative_path: str | os.PathLike[str], payload: Any) -> Path:
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(redact_sensitive(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
