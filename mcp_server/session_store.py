"""Thread-safe in-memory session lifecycle management."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, TypeAlias

from mcp_server.validation import InputValidationError

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
SessionRemovalReason: TypeAlias = Literal["manual", "completed", "expired"]
_SESSION_REMOVAL_REASONS = {"manual", "completed", "expired"}


def empty_session_state() -> dict[str, Any]:
    return {
        "input_image": None,
        "output_dir": None,
        "segments": {},
        "face_features": {},
        "face_layers": [],
        "layers": [],
        "meshes": {},
        "rigging": {},
        "physics": {},
        "motions": [],
        "model_files": {},
        "face_output_dir": None,
        "analysis_metadata": {},
        "layer_generation_metadata": {},
        "ai_parts": [],
        "ai_part_layers": [],
        "cubism_template_mapping": {},
        "cubism_psd_path": None,
        "cubism_automation_plan": {},
        "cubism_dispatch_bundle": {},
        "cubism_dispatch_execution": {},
    }


@dataclass
class SessionMetrics:
    created_sessions: int = 0
    rejected_sessions: int = 0
    expired_sessions: int = 0
    closed_sessions: int = 0
    completed_sessions: int = 0


@dataclass
class SessionRecord:
    """In-memory state for a single MCP session."""

    session_id: str
    state: dict[str, Any] = field(default_factory=empty_session_state)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    active_operation: bool = False


class InMemorySessionStore:
    """Own session records, limits, metrics, and operation locking."""

    def __init__(
        self,
        *,
        max_sessions: int,
        ttl_seconds: int,
        max_concurrent_operations: int,
    ) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.max_concurrent_operations = max_concurrent_operations
        self.records: dict[str, SessionRecord] = {}
        self.lock = RLock()
        self.metrics = SessionMetrics()

    def new_session_id(self) -> str:
        return f"job_{uuid.uuid4().hex[:12]}"

    def validate_session_id(self, session_id: str) -> str:
        if not SESSION_ID_RE.match(session_id):
            raise InputValidationError("session_id contains unsupported characters.")
        return session_id

    def prune_expired(self, now: float | None = None) -> None:
        with self.lock:
            self._prune_expired_locked(now)

    def _prune_expired_locked(self, now: float | None = None) -> None:
        current_time = now or time.time()
        expired = [
            session_id
            for session_id, record in self.records.items()
            if current_time - record.last_accessed > self.ttl_seconds
        ]
        for session_id in expired:
            self.records.pop(session_id, None)
            self.metrics.expired_sessions += 1

    def create(self) -> str:
        with self.lock:
            self._prune_expired_locked()
            if len(self.records) >= self.max_sessions:
                self.metrics.rejected_sessions += 1
                raise InputValidationError(
                    "Too many active sessions. Wait for an existing job to expire or finish."
                )
            session_id = self.new_session_id()
            self.records[session_id] = SessionRecord(session_id=session_id)
            self.metrics.created_sessions += 1
            return session_id

    def remove(
        self,
        session_id: str,
        *,
        reason: SessionRemovalReason = "manual",
    ) -> bool:
        if reason not in _SESSION_REMOVAL_REASONS:
            raise ValueError(f"Unsupported session removal reason: {reason!r}")
        with self.lock:
            record = self.records.pop(session_id, None)
            if record is None:
                return False
            if reason == "manual":
                self.metrics.closed_sessions += 1
            elif reason == "completed":
                self.metrics.completed_sessions += 1
            elif reason == "expired":
                self.metrics.expired_sessions += 1
            return True

    def get(self, session_id: str, *, touch: bool = True) -> SessionRecord:
        self.validate_session_id(session_id)
        with self.lock:
            self._prune_expired_locked()
            record = self.records.get(session_id)
            if record is None:
                raise InputValidationError(
                    "Unknown or expired session_id. Run analyze_photo first."
                )
            if touch:
                record.last_accessed = time.time()
            return record

    def get_state(self, session_id: str) -> dict[str, Any]:
        return self.get(session_id).state

    def require_state_field(self, session_id: str, field: str, message: str) -> dict[str, Any]:
        state = self.get_state(session_id)
        if not state.get(field):
            raise InputValidationError(message)
        return state

    def active_operation_count(self) -> int:
        with self.lock:
            return self._active_operation_count_locked()

    def _active_operation_count_locked(self) -> int:
        return sum(1 for record in self.records.values() if record.active_operation)

    @contextmanager
    def operation(self, session_id: str) -> Iterator[SessionRecord]:
        with self.lock:
            self._prune_expired_locked()
            record = self.get(session_id, touch=False)
            if record.active_operation:
                self.metrics.rejected_sessions += 1
                raise InputValidationError(
                    "This session is already busy running another operation."
                )
            if self._active_operation_count_locked() >= self.max_concurrent_operations:
                self.metrics.rejected_sessions += 1
                raise InputValidationError("Server is busy. Retry after another job completes.")
            record.active_operation = True
            record.last_accessed = time.time()

        try:
            yield record
        finally:
            with self.lock:
                existing = self.records.get(session_id)
                if existing is not None:
                    existing.active_operation = False
                    existing.last_accessed = time.time()
