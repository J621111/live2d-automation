from typing import cast

import pytest

from mcp_server.session_store import InMemorySessionStore, SessionRemovalReason
from mcp_server.validation import InputValidationError


def _store(
    *,
    max_sessions: int = 2,
    ttl_seconds: int = 60,
    max_concurrent_operations: int = 1,
) -> InMemorySessionStore:
    return InMemorySessionStore(
        max_sessions=max_sessions,
        ttl_seconds=ttl_seconds,
        max_concurrent_operations=max_concurrent_operations,
    )


def test_session_store_enforces_capacity_and_tracks_metrics() -> None:
    store = _store(max_sessions=1)

    session_id = store.create()

    assert session_id in store.records
    assert store.records[session_id].state["segments"] == {}
    assert store.metrics.created_sessions == 1
    with pytest.raises(InputValidationError, match="Too many active sessions"):
        store.create()
    assert store.metrics.rejected_sessions == 1


def test_session_store_prunes_expired_records() -> None:
    store = _store(ttl_seconds=10)
    session_id = store.create()
    store.records[session_id].last_accessed = 100.0

    store.prune_expired(now=111.0)

    assert session_id not in store.records
    assert store.metrics.expired_sessions == 1


def test_session_store_does_not_prune_active_operations() -> None:
    store = _store(ttl_seconds=10)
    session_id = store.create()

    with store.operation(session_id):
        store.records[session_id].last_accessed = 100.0
        store.prune_expired(now=111.0)

        assert session_id in store.records
        assert store.metrics.expired_sessions == 0

    store.records[session_id].last_accessed = 100.0
    store.prune_expired(now=111.0)

    assert session_id not in store.records
    assert store.metrics.expired_sessions == 1


def test_session_store_rejects_unknown_removal_reason_before_deleting() -> None:
    store = _store()
    session_id = store.create()
    invalid_reason = cast(SessionRemovalReason, "complete")

    with pytest.raises(ValueError, match="Unsupported session removal reason"):
        store.remove(session_id, reason=invalid_reason)

    assert session_id in store.records


def test_session_store_tracks_removal_metrics_by_reason() -> None:
    store = _store(max_sessions=3)
    manual_session = store.create()
    completed_session = store.create()
    expired_session = store.create()

    assert store.remove(manual_session, reason="manual") is True
    assert store.remove(completed_session, reason="completed") is True
    assert store.remove(expired_session, reason="expired") is True

    assert store.records == {}
    assert store.metrics.closed_sessions == 1
    assert store.metrics.completed_sessions == 1
    assert store.metrics.expired_sessions == 1


def test_session_store_rejects_busy_and_concurrent_operations() -> None:
    store = _store(max_concurrent_operations=1)
    first_session = store.create()
    second_session = store.create()

    with store.operation(first_session):
        assert store.records[first_session].active_operation is True
        with (
            pytest.raises(InputValidationError, match="already busy"),
            store.operation(first_session),
        ):
            pass
        with (
            pytest.raises(InputValidationError, match="Server is busy"),
            store.operation(second_session),
        ):
            pass

    assert store.records[first_session].active_operation is False
    assert store.metrics.rejected_sessions == 2
