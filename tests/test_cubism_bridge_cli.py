"""Installation and attach checks use synthetic app layouts and process listings."""

import importlib
import os
import plistlib
import subprocess

import pytest


def installation(tmp_path):
    root = tmp_path / "Live2D Cubism Test"
    app = root / "Editor.app"
    for relative in ["Contents/MacOS/CubismEditor"]:
        path = app / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (app / "Contents/Info.plist").write_bytes(
        plistlib.dumps({"CFBundleExecutable": "CubismEditor"})
    )
    for relative in ["jre/Contents/Home/lib/server/libjvm.dylib", "res/json-simple-1.1.jar"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root, app


def test_installation_accepts_explicit_root_and_app_bundle(tmp_path):
    from mcp_server.tools.cubism_bridge_cli import discover_installation

    root, app = installation(tmp_path)
    first = discover_installation(root)
    second = discover_installation(app)
    assert first == second
    assert first.executable == app / "Contents/MacOS/CubismEditor"
    assert first.json_jar == root / "res/json-simple-1.1.jar"


def test_installation_environment_override_and_clear_missing_error(tmp_path, monkeypatch):
    from mcp_server.tools.cubism_bridge_cli import discover_installation

    root, _ = installation(tmp_path)
    monkeypatch.setenv("LIVE2D_CUBISM_HOME", str(root))
    assert discover_installation().root == root
    with pytest.raises(RuntimeError, match="installation"):
        discover_installation(tmp_path / "missing")


def test_process_selection_checks_exact_executable_owner_and_explicit_pid(tmp_path, monkeypatch):
    from mcp_server.tools.cubism_bridge_cli import discover_installation, verified_editor_pid

    root, _ = installation(tmp_path)
    editor = discover_installation(root)
    monkeypatch.setattr(os, "getuid", lambda: 501, raising=False)
    listing = (
        f"100 501 {editor.executable}\n101 502 {editor.executable}\n"
        f"102 501 {editor.executable}-fake\n"
    )

    def run(command, **kwargs):
        assert command == ["/bin/ps", "-A", "-o", "pid=,uid=,comm="]
        return subprocess.CompletedProcess(command, 0, listing, "")

    assert verified_editor_pid(editor, run_command=run) == 100
    with pytest.raises(RuntimeError, match="verified"):
        verified_editor_pid(editor, pid=101, run_command=run)
    with pytest.raises(RuntimeError, match="verified"):
        verified_editor_pid(editor, pid=102, run_command=run)


def test_multiple_editors_require_explicit_matching_pid(tmp_path, monkeypatch):
    from mcp_server.tools.cubism_bridge_cli import discover_installation, verified_editor_pid

    root, _ = installation(tmp_path)
    editor = discover_installation(root)
    monkeypatch.setattr(os, "getuid", lambda: 501, raising=False)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, f"100 501 {editor.executable}\n101 501 {editor.executable}\n", ""
        )

    with pytest.raises(RuntimeError, match="exactly one"):
        verified_editor_pid(editor, run_command=run)
    assert verified_editor_pid(editor, pid=101, run_command=run) == 101


def test_cli_help_does_not_import_jpype_or_discover_editor(monkeypatch, capsys):
    from mcp_server.tools.cubism_bridge_cli import main

    original = importlib.import_module

    def importing(name, *args, **kwargs):
        assert name != "jpype", "Help must not import an optional JVM integration"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", importing)
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--editor-root" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "posix", reason="The macOS JVM CLI requires POSIX permissions")
def test_missing_jpype_returns_actionable_failure_without_import_breakage(
    tmp_path, monkeypatch, capsys
):
    from mcp_server.tools.cubism_bridge_cli import main

    root, _ = installation(tmp_path)
    original = importlib.import_module

    def importing(name, *args, **kwargs):
        if name == "jpype":
            raise ModuleNotFoundError("jpype")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", importing)
    assert (
        main(["compile", "--editor-root", str(root), "--build-dir", str(tmp_path / "build")]) == 1
    )
    assert "macos-cubism" in capsys.readouterr().err


def test_source_is_resolved_from_package_not_working_directory(tmp_path, monkeypatch):
    from mcp_server.tools.cubism_bridge_cli import bridge_source

    monkeypatch.chdir(tmp_path)
    path = bridge_source()
    assert path.name == "SwingBridge.java" and path.is_file()
    assert tmp_path not in path.parents


@pytest.mark.skipif(os.name != "posix", reason="The macOS JVM CLI requires POSIX permissions")
def test_attach_never_replays_a_stale_request_after_readiness_disappears(
    tmp_path, monkeypatch, capsys
):
    from mcp_server.tools import cubism_bridge_cli as cli

    root, _ = installation(tmp_path)
    session = tmp_path / "session"
    session.mkdir(mode=0o700)
    stale = session / "request.json"
    stale.write_text('{"action":"click","id":2}')
    stale.chmod(0o600)
    monkeypatch.setattr(cli, "verified_editor_pid", lambda *args: 123)
    monkeypatch.setattr(cli, "_start_java", lambda *args: None)
    result = cli.main(
        [
            "attach",
            "--editor-root",
            str(root),
            "--build-dir",
            str(tmp_path / "build"),
            "--session-dir",
            str(session),
        ]
    )
    assert result == 1 and "pending" in capsys.readouterr().err.lower()
    assert stale.read_text() == '{"action":"click","id":2}'
