"""Private, serialized JSON transport to a locally attached Cubism Swing bridge."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .cubism_paths import session_directory

MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
ACTIONS = {
    "close_dialog",
    "click",
    "focus",
    "capture",
    "set_text",
    "set_number",
    "select_combo",
    "select_tree",
    "find_table_rows",
    "select_table",
    "scroll_table_row",
    "select_list",
    "mouse",
    "drag",
    "scroll_into_view",
}


def _private(info: os.stat_result, *, directory: bool = False) -> None:
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not kind(info.st_mode)
        or info.st_uid != os.getuid()
        or (not directory and info.st_nlink != 1)
    ):
        raise RuntimeError("Unsafe Cubism session transport artifact")
    if info.st_mode & 0o077:
        raise RuntimeError(
            "Cubism session and transport artifacts must be private to the current user"
        )


def _artifact(root: int, name: str) -> os.stat_result | None:
    try:
        info = os.stat(name, dir_fd=root, follow_symlinks=False)
    except FileNotFoundError:
        return None
    _private(info)
    return info


def _read_json(root: int, name: str, limit: int) -> dict[str, Any]:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise RuntimeError("Unsafe Cubism session transport artifact") from exc
    try:
        info = os.fstat(fd)
        _private(info)
        if info.st_size > limit:
            raise RuntimeError("Cubism transport response exceeds its size limit")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise RuntimeError("Cubism transport response exceeds its size limit")
        try:
            value = json.loads(content)
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("Cubism returned a malformed JSON response") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Cubism returned a malformed response")
        return value
    finally:
        os.close(fd)


def request(
    payload: dict[str, Any], timeout: float = 15, *, session_dir: str | Path | None = None
) -> dict[str, Any]:
    """Send one action without overwriting another client; timed-out requests remain pending."""
    if os.name != "posix":
        raise RuntimeError("Cubism Swing transport requires a POSIX local session")
    import fcntl

    if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
        raise ValueError("Expected a JSON object with an action")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    request_id = uuid.uuid4().hex
    encoded = json.dumps({**payload, "request_id": request_id}, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("Cubism request exceeds its size limit")
    session = Path(session_dir) if session_dir is not None else session_directory()
    try:
        root = os.open(session, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeError("Unsafe or missing Cubism session directory") from exc
    lock = None
    temporary = f"request-{request_id}.tmp"
    try:
        _private(os.fstat(root), directory=True)
        for name in ["ready.json", "request.json", "request.lock", "response.tmp"]:
            _artifact(root, name)
        try:
            ready = _read_json(root, "ready.json", 4096)
        except FileNotFoundError as exc:
            raise RuntimeError("The Cubism Swing bridge is not attached") from exc
        if type(ready.get("pid")) is not int or ready["pid"] <= 0:
            raise RuntimeError("Cubism bridge readiness is malformed")
        lock = os.open("request.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=root)
        _private(os.fstat(lock))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A Cubism UI request is already running") from exc
        if _artifact(root, "request.json") is not None:
            raise RuntimeError(
                "A Cubism UI request is already running; inspect the pending request"
            )
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
        # link publishes a complete request atomically and refuses an existing destination.
        try:
            os.link(
                temporary, "request.json", src_dir_fd=root, dst_dir_fd=root, follow_symlinks=False
            )
        except FileExistsError as exc:
            raise RuntimeError("A Cubism UI request is already running") from exc
        finally:
            os.unlink(temporary, dir_fd=root)
        response_name = f"response-{request_id}.json"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = _read_json(root, response_name, MAX_RESPONSE_BYTES)
            except FileNotFoundError:
                time.sleep(min(0.025, max(0, deadline - time.monotonic())))
                continue
            if response.get("request_id") != request_id:
                raise RuntimeError("Cubism response does not match the request")
            os.unlink(response_name, dir_fd=root)
            return response
        raise TimeoutError("Cubism did not complete the UI request; pending request was retained")
    finally:
        if lock is not None:
            os.close(lock)
        os.close(root)


def _flatten(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    for child in node.get("children", []):
        if isinstance(child, dict):
            yield from _flatten(child)


def guarded_action(
    expected_document: str,
    payload: dict[str, Any],
    *,
    sender: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    session_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Require the exact active model and modal ownership, then repeat the guard on Swing."""
    if (
        not isinstance(expected_document, str)
        or not expected_document.endswith(".cmo3")
        or any(char in expected_document for char in "/\\\x00\n\r")
    ):
        raise ValueError("Expected an exact document filename")
    if payload.get("action") not in ACTIONS | {"snapshot"}:
        raise ValueError("Unsupported editor action")
    send = sender or (lambda value: request(value, session_dir=session_dir))
    depth = payload.get("depth", 42) if payload["action"] == "snapshot" else 42
    state = send({"action": "snapshot", "depth": depth})
    windows = state.get("windows", [])
    if (
        not isinstance(windows, list)
        or not windows
        or not all(isinstance(w, dict) for w in windows)
    ):
        raise ValueError("The active Cubism document does not match")
    if str(windows[0].get("text", "")).partition(" - ")[2] != expected_document:
        raise ValueError("The active Cubism document does not match")
    if payload["action"] == "snapshot":
        return state
    widget_id = payload.get("id")
    owners = [
        window
        for window in windows
        if widget_id is not None and any(node.get("id") == widget_id for node in _flatten(window))
    ]
    if len(owners) != 1:
        raise ValueError("Unknown or expired widget")
    owner = owners[0]
    if "blocked_by" not in owner:
        raise ValueError("Missing modal blocking information; reattach the bridge")
    if owner["blocked_by"] is not None:
        raise ValueError("Resolve the active modal before editing the document")
    return send({**payload, "expected_document": expected_document})


def main(argv: list[str] | None = None) -> int:
    """Read one JSON object from stdin and write one JSON response to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--expected-document")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.read(MAX_REQUEST_BYTES + 1)
        if len(raw.encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError("Request JSON exceeds its size limit")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid request JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")

        def sender(value):
            return request(value, args.timeout, session_dir=args.session_dir)

        result = (
            guarded_action(args.expected_document, payload, sender=sender)
            if args.expected_document
            else sender(payload)
        )
        print(json.dumps(result))
        return 0 if result.get("status") != "error" else 1
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Cubism request failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
