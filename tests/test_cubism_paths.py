"""Configurable editor state does not depend on the installed package directory."""

from pathlib import Path


def test_defaults_follow_working_directory_without_creating_output(tmp_path, monkeypatch):
    from mcp_server.tools.cubism_paths import session_directory, state_directory

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIVE2D_AUTOMATION_STATE_DIR", raising=False)
    monkeypatch.delenv("LIVE2D_CUBISM_SESSION_DIR", raising=False)
    assert state_directory() == tmp_path / "output"
    assert session_directory() == tmp_path / "output/swing-bridge/session"
    assert not (tmp_path / "output").exists()


def test_environment_can_separate_state_and_session(tmp_path, monkeypatch):
    from mcp_server.tools.cubism_paths import session_directory, state_directory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LIVE2D_AUTOMATION_STATE_DIR", "custom-state")
    monkeypatch.setenv("LIVE2D_CUBISM_SESSION_DIR", str(tmp_path / "private-session"))
    assert state_directory() == tmp_path / "custom-state"
    assert session_directory() == tmp_path / "private-session"
    assert isinstance(state_directory(), Path)
