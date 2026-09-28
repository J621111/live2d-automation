"""Exercise the real local transport with a synthetic, non-editor responder."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture
def session(tmp_path):
    if os.name != "posix":
        pytest.skip("Local Swing transport requires POSIX private files")
    root = tmp_path / "session"
    root.mkdir(mode=0o700)
    ready = root / "ready.json"
    ready.write_text(json.dumps({"pid": os.getpid()}))
    ready.chmod(0o600)
    return root


def respond(session, transform=None, started=None, release=None):
    """Emulate only the external JSON queue endpoint, including its completion cleanup."""
    deadline = time.monotonic() + 3
    while not (session / "request.json").exists():
        if time.monotonic() > deadline:
            raise AssertionError("No bridge request arrived")
        time.sleep(0.002)
    payload = json.loads((session / "request.json").read_text())
    assert (session / "request.json").stat().st_mode & 0o777 == 0o600
    if started:
        started.set()
    if release:
        assert release.wait(3)
    result = {
        "status": "success",
        "request_id": payload["request_id"],
        "value": payload.get("value"),
    }
    if transform:
        result = transform(result)
    path = session / f'response-{payload["request_id"]}.json'
    temporary = session / "synthetic.tmp"
    temporary.write_text(json.dumps(result))
    temporary.chmod(0o600)
    (session / "request.json").unlink()
    temporary.replace(path)
    return payload


def test_request_correlates_response_and_keeps_input_unchanged(session):
    from mcp_server.tools.cubism_swing import request

    original = {"action": "snapshot", "value": "model"}
    with ThreadPoolExecutor() as pool:
        response = pool.submit(respond, session)
        result = request(original, timeout=2, session_dir=session)
        payload = response.result()
    assert result == {"status": "success", "request_id": payload["request_id"], "value": "model"}
    assert original == {"action": "snapshot", "value": "model"}
    assert not list(session.glob("response-*.json"))


def test_concurrent_request_cannot_overwrite_inflight_action(session):
    from mcp_server.tools.cubism_swing import request

    started, release = threading.Event(), threading.Event()
    with ThreadPoolExecutor() as pool:
        responder = pool.submit(respond, session, None, started, release)
        first = pool.submit(
            request, {"action": "snapshot", "value": "first"}, 2, session_dir=session
        )
        assert started.wait(2)
        try:
            with pytest.raises(RuntimeError, match="running"):
                request({"action": "shutdown"}, session_dir=session)
            assert json.loads((session / "request.json").read_text())["value"] == "first"
        finally:
            release.set()
        assert first.result()["value"] == "first"
        responder.result()


def test_timeout_retains_request_and_blocks_a_later_action(session):
    from mcp_server.tools.cubism_swing import request

    with pytest.raises(TimeoutError):
        request({"action": "snapshot"}, timeout=0.03, session_dir=session)
    stale = (session / "request.json").read_bytes()
    with pytest.raises(RuntimeError, match="running"):
        request({"action": "shutdown"}, session_dir=session)
    assert (session / "request.json").read_bytes() == stale


@pytest.mark.parametrize("artifact", ["ready.json", "request.json", "response.tmp", "request.lock"])
def test_transport_refuses_symlink_artifacts_without_touching_target(session, artifact):
    from mcp_server.tools.cubism_swing import request

    target = session.parent / "outside"
    target.write_text("do not overwrite")
    path = session / artifact
    path.unlink(missing_ok=True)
    path.symlink_to(target)
    with pytest.raises(RuntimeError, match="Unsafe|unsafe"):
        request({"action": "snapshot"}, timeout=0.01, session_dir=session)
    assert target.read_text() == "do not overwrite"


def test_transport_refuses_shared_session_and_linked_session(session):
    from mcp_server.tools.cubism_swing import request

    session.chmod(0o755)
    with pytest.raises(RuntimeError, match="private"):
        request({"action": "snapshot"}, session_dir=session)
    session.chmod(0o700)
    alias = session.parent / "alias"
    alias.symlink_to(session, target_is_directory=True)
    with pytest.raises(RuntimeError, match="Unsafe|unsafe"):
        request({"action": "snapshot"}, session_dir=alias)


@pytest.mark.parametrize("transform", [lambda _: [], lambda x: {**x, "request_id": "wrong"}])
def test_transport_preserves_malformed_or_mismatched_reply_for_diagnosis(session, transform):
    from mcp_server.tools.cubism_swing import request

    with ThreadPoolExecutor() as pool:
        responder = pool.submit(respond, session, transform)
        with pytest.raises(RuntimeError, match="malformed|match"):
            request({"action": "snapshot"}, timeout=2, session_dir=session)
        responder.result()
    assert len(list(session.glob("response-*.json"))) == 1


def test_guard_rejects_wrong_model_and_background_widgets():
    from mcp_server.tools.cubism_swing import guarded_action

    calls = []
    state = {
        "status": "success",
        "windows": [
            {"text": "Cubism - portrait.cmo3", "id": 1, "blocked_by": 3, "children": [{"id": 2}]},
            {
                "text": "Confirm",
                "id": 3,
                "modal": True,
                "blocked_by": None,
                "children": [{"id": 4}],
            },
        ],
    }

    def sender(payload):
        calls.append(payload)
        return state if payload["action"] == "snapshot" else {"status": "success"}

    with pytest.raises(ValueError, match="document"):
        guarded_action("other.cmo3", {"action": "click", "id": 4}, sender=sender)
    with pytest.raises(ValueError, match="modal"):
        guarded_action("portrait.cmo3", {"action": "click", "id": 2}, sender=sender)
    assert len(calls) == 2
    assert guarded_action("portrait.cmo3", {"action": "focus", "id": 4}, sender=sender) == {
        "status": "success"
    }
    assert calls[-1]["expected_document"] == "portrait.cmo3"


@pytest.mark.parametrize(
    "expected,allowed", [("Draft - Avatar.cmo3", True), ("Avatar.cmo3", False)]
)
def test_document_guard_compares_complete_filename(expected, allowed):
    from mcp_server.tools.cubism_swing import guarded_action

    calls = []
    state = {
        "windows": [
            {
                "id": "a:1",
                "text": "Cubism - Draft - Avatar.cmo3",
                "blocked_by": None,
                "children": [{"id": "a:2"}],
            }
        ],
    }

    def sender(payload):
        if payload["action"] == "snapshot":
            return state
        calls.append(payload)
        return {"status": "success"}

    if allowed:
        assert (
            guarded_action(expected, {"action": "click", "id": "a:2"}, sender=sender)["status"]
            == "success"
        )
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="document"):
            guarded_action(expected, {"action": "click", "id": "a:2"}, sender=sender)
        assert calls == []


@pytest.mark.parametrize("reverse", [False, True])
def test_modal_guard_uses_blocker_instead_of_creation_order(reverse):
    from mcp_server.tools.cubism_swing import guarded_action

    active = {"id": "a:2", "modal": True, "blocked_by": None, "children": [{"id": "a:3"}]}
    blocked = {"id": "a:4", "modal": True, "blocked_by": "a:2", "children": [{"id": "a:5"}]}
    state = {
        "windows": [
            {"id": "a:1", "text": "Cubism - Avatar.cmo3", "blocked_by": "a:4"},
            *([blocked, active] if reverse else [active, blocked]),
        ]
    }
    calls = []

    def sender(payload):
        if payload["action"] == "snapshot":
            return state
        calls.append(payload)
        return {"status": "success"}

    assert (
        guarded_action("Avatar.cmo3", {"action": "click", "id": "a:3"}, sender=sender)["status"]
        == "success"
    )
    with pytest.raises(ValueError, match="modal"):
        guarded_action("Avatar.cmo3", {"action": "click", "id": "a:5"}, sender=sender)
    assert [p["id"] for p in calls] == ["a:3"]


def test_guard_rejects_missing_modal_metadata():
    from mcp_server.tools.cubism_swing import guarded_action

    def sender(payload):
        assert payload["action"] == "snapshot", "Missing blocker data must prevent mutation"
        return {
            "windows": [{"id": "a:1", "text": "Cubism - Avatar.cmo3", "children": [{"id": "a:2"}]}]
        }

    with pytest.raises(ValueError, match="modal"):
        guarded_action("Avatar.cmo3", {"action": "click", "id": "a:2"}, sender=sender)


def test_stale_attachment_id_is_not_sent_to_new_widget():
    from mcp_server.tools.cubism_swing import guarded_action

    def sender(payload):
        assert payload["action"] == "snapshot"
        return {
            "windows": [
                {
                    "id": "new:1",
                    "text": "Cubism - Avatar.cmo3",
                    "blocked_by": None,
                    "children": [{"id": "new:2"}],
                }
            ]
        }

    with pytest.raises(ValueError, match="expired"):
        guarded_action("Avatar.cmo3", {"action": "click", "id": "old:2"}, sender=sender)


def test_stdin_cli_reports_invalid_json_without_transport(capsys, monkeypatch):
    import io

    from mcp_server.tools.cubism_swing import main

    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert main([]) == 1
    output = capsys.readouterr()
    assert not output.out and "JSON" in output.err


@pytest.mark.parametrize("options,depth", [({"depth": 0}, 0), ({"depth": 8}, 8), ({}, 42)])
def test_guarded_snapshot_honors_depth(options, depth):
    from mcp_server.tools.cubism_swing import guarded_action

    requests = []
    state = {"status": "success", "windows": [{"id": "a:1", "text": "Cubism - Avatar.cmo3"}]}

    def sender(payload):
        requests.append(payload)
        return state

    assert guarded_action("Avatar.cmo3", {"action": "snapshot", **options}, sender=sender) == state
    assert requests == [{"action": "snapshot", "depth": depth}]


def test_mutation_keeps_deep_guard_snapshot_even_with_depth_zero():
    from mcp_server.tools.cubism_swing import guarded_action

    requests = []

    def sender(payload):
        requests.append(payload)
        if payload["action"] == "snapshot":
            return {
                "windows": [
                    {
                        "id": "a:1",
                        "text": "Cubism - Avatar.cmo3",
                        "blocked_by": None,
                        "children": [{"id": "a:2", "children": [{"id": "a:3"}]}],
                    }
                ]
            }
        return {"status": "success"}

    result = guarded_action(
        "Avatar.cmo3", {"action": "click", "id": "a:3", "depth": 0}, sender=sender
    )
    assert result["status"] == "success"
    assert requests[0] == {"action": "snapshot", "depth": 42}
    assert requests[1]["id"] == "a:3"


@pytest.mark.parametrize("depth", [0, 8])
def test_guarded_cli_forwards_snapshot_depth(monkeypatch, capsys, depth):
    import io

    from mcp_server.tools import cubism_swing

    requests = []

    def request(payload, *args, **kwargs):
        requests.append(payload)
        return {"status": "success", "windows": [{"text": "Cubism - Avatar.cmo3"}]}

    monkeypatch.setattr(cubism_swing, "request", request)
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"action": "snapshot", "depth": depth}))
    )
    assert cubism_swing.main(["--expected-document", "Avatar.cmo3"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "success"
    assert requests == [{"action": "snapshot", "depth": depth}]
